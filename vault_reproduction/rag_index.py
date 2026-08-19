#!/usr/bin/env python3
"""Build an offline, provenance-preserving RAG index from Obsidian Markdown.

The Obsidian vault remains the source of truth.  This script creates a
rebuildable SQLite projection with FTS5 retrieval and metadata fields that can
later hold vector embeddings without changing the source notes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import signal
import sqlite3
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


INDEX_VERSION = "1"
DEFAULT_INCLUDE = "05 Music/shape-note/**/*.md"
DEFAULT_MAX_CHARS = 1800
DEFAULT_OVERLAP_CHARS = 240


class UnreadableSourceError(RuntimeError):
    """A source file did not return within the local read timeout."""

BOOK_EDITIONS: dict[str, dict[str, str]] = {
    "sh": {"edition_name": "The Sacred Harp (edition unspecified)", "tradition": "Sacred Harp", "year": ""},
    "sh1991": {"edition_name": "The Sacred Harp (1991 Denson edition)", "tradition": "Sacred Harp", "year": "1991"},
    "sh2025": {"edition_name": "The Sacred Harp (2025 edition)", "tradition": "Sacred Harp", "year": "2025"},
    "shcooper2012": {"edition_name": "The Sacred Harp (Cooper, 2012)", "tradition": "Sacred Harp Cooper", "year": "2012"},
    "ch7": {"edition_name": "The Christian Harmony (7th edition)", "tradition": "Christian Harmony", "year": ""},
    "shenandoah": {"edition_name": "Shenandoah Harmony (2013)", "tradition": "Shenandoah Harmony", "year": "2013"},
    "southernharmony": {"edition_name": "The Southern Harmony", "tradition": "Southern Harmony", "year": ""},
    "kentucky": {"edition_name": "A Supplement to the Kentucky Harmony", "tradition": "Kentucky Harmony", "year": ""},
    "mnharmony": {"edition_name": "Minnesota Harmony", "tradition": "Minnesota Harmony", "year": ""},
    "sacredharptunes": {"edition_name": "Sacred Harp Tunes online corpus", "tradition": "Sacred Harp Tunes", "year": ""},
    "trumpet": {"edition_name": "The Trumpet online corpus", "tradition": "The Trumpet", "year": ""},
}


def _fallback_frontmatter(text: str) -> dict[str, Any]:
    """Parse the small YAML subset needed when PyYAML is unavailable."""

    result: dict[str, Any] = {}
    current_list: str | None = None
    current_map: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        stripped = line.strip()
        if stripped.startswith("-") and current_list:
            result.setdefault(current_list, []).append(_scalar(stripped[1:].strip()))
            continue
        if ":" not in stripped:
            continue
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if raw_value:
            result[key] = _scalar(raw_value)
            current_list = None
            current_map = None
        else:
            current_list = key
            current_map = None
            result[key] = []
    return result


def _scalar(value: str) -> Any:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    if value.lower() in {"null", "~"}:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        try:
            return json.loads(value.replace("'", '"'))
        except json.JSONDecodeError:
            return [part.strip() for part in value[1:-1].split(",") if part.strip()]
    return value


def parse_frontmatter(markdown: str) -> tuple[dict[str, Any], str]:
    """Return frontmatter and body, tolerating malformed legacy notes."""

    if not markdown.startswith("---"):
        return {}, markdown
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n?", markdown, re.DOTALL)
    if not match:
        return {}, markdown
    raw = match.group(1)
    try:
        import yaml  # type: ignore

        parsed = yaml.safe_load(raw) or {}
        metadata = parsed if isinstance(parsed, dict) else {}
    except (ImportError, Exception):
        metadata = _fallback_frontmatter(raw)
    return metadata, markdown[match.end() :]


def clean_markdown(body: str) -> str:
    """Make Markdown retrieval-friendly while preserving the source meaning."""

    # Dataview is a presentation layer, not corpus evidence.  Excluding it
    # avoids embedding JavaScript and generated tables into text-hub chunks.
    body = re.sub(r"```dataview(?:js)?\s*.*?```", "", body, flags=re.IGNORECASE | re.DOTALL)
    body = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)
    body = re.sub(r"!\[([^]]*)\]\([^)]*\)", r"\1", body)
    body = re.sub(r"\[([^]]+)\]\([^)]*\)", r"\1", body)
    body = re.sub(r"\[\[([^]|]+)\|([^]]+)\]\]", r"\2", body)
    body = re.sub(r"\[\[([^]]+)\]\]", r"\1", body)
    body = re.sub(r"^[ \t]*[-*_]{3,}[ \t]*$", "", body, flags=re.MULTILINE)
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)]


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _json_dumps(value: Any) -> str:
    """Serialize YAML-derived metadata without losing date-like frontmatter."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _first_text(metadata: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, list):
            if value:
                return _as_text(value[0])
        elif value not in (None, ""):
            return _as_text(value)
    return ""


def _slug_identity(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", normalized.lower()).strip("-")
    return normalized or "unknown"


def infer_kind(metadata: dict[str, Any], relative_path: str) -> str:
    declared = _as_text(metadata.get("type")).strip().lower()
    if declared in {"song", "text", "book", "note"}:
        return declared
    if "/texts/" in f"/{relative_path}":
        return "text"
    if "/songs/" in f"/{relative_path}:":
        return "song"
    return "note"


def stable_document_id(kind: str, metadata: dict[str, Any], relative_path: str) -> str:
    if kind == "text" and metadata.get("text_key"):
        return f"text:{_slug_identity(_as_text(metadata['text_key']))}"
    if kind == "song" and metadata.get("source_note_ids"):
        first_id = _as_list(metadata["source_note_ids"])[0]
        return f"song:{_slug_identity(first_id)}"
    digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:24]
    return f"{kind}:{digest}"


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = text.rfind(" ", start + max_chars // 2, end)
            if boundary > start:
                end = boundary
        part = text[start:end].strip()
        if part:
            parts.append(part)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return parts


def chunk_markdown(body: str, max_chars: int = DEFAULT_MAX_CHARS, overlap: int = DEFAULT_OVERLAP_CHARS) -> list[dict[str, str]]:
    """Chunk by headings and paragraphs, retaining a heading on every chunk."""

    sections: list[tuple[str, list[str]]] = []
    heading = ""
    paragraphs: list[str] = []
    for line in body.splitlines():
        if re.match(r"^#{1,6}\s+", line):
            if paragraphs:
                sections.append((heading, paragraphs))
                paragraphs = []
            heading = re.sub(r"^#{1,6}\s+", "", line).strip()
        elif line.strip():
            paragraphs.append(line.strip())
        elif paragraphs and paragraphs[-1] != "":
            paragraphs.append("")
    if paragraphs:
        sections.append((heading, paragraphs))
    if not sections:
        return []

    chunks: list[dict[str, str]] = []
    for section_heading, lines in sections:
        section_text = "\n".join(lines).strip()
        if not section_text:
            continue
        paragraphs_for_section = [p.strip() for p in re.split(r"\n\s*\n", section_text) if p.strip()]
        current = ""
        for paragraph in paragraphs_for_section:
            for piece in _split_long(paragraph, max_chars, overlap):
                candidate = f"{current}\n\n{piece}".strip() if current else piece
                if current and len(candidate) > max_chars:
                    chunks.append({"heading": section_heading, "text": current})
                    tail = current[-overlap:].lstrip() if overlap else ""
                    current = f"{tail}\n\n{piece}".strip() if tail else piece
                else:
                    current = candidate
        if current:
            chunks.append({"heading": section_heading, "text": current})
    return chunks


def _iter_source_files(vault: Path, includes: Iterable[str], excludes: Iterable[str]) -> list[Path]:
    excluded = {path.resolve() for pattern in excludes for path in vault.glob(pattern) if path.is_file()}
    files: set[Path] = set()
    for pattern in includes:
        for path in vault.glob(pattern):
            if path.is_file() and path.suffix.lower() == ".md" and path.resolve() not in excluded:
                files.add(path.resolve())
    return sorted(files, key=lambda path: path.relative_to(vault).as_posix().lower())


def _record_for_file(path: Path, vault: Path, read_timeout_seconds: float = 5.0) -> dict[str, Any]:
    raw = _read_source_text(path, read_timeout_seconds)
    metadata, body = parse_frontmatter(raw)
    relative_path = path.relative_to(vault).as_posix()
    kind = infer_kind(metadata, relative_path)
    clean_body = clean_markdown(body)
    title = _as_text(metadata.get("title")) or next(
        (line.lstrip("# ").strip() for line in clean_body.splitlines() if line.startswith("#")),
        path.stem,
    )
    normalized_metadata = {
        "type": kind,
        "title": title,
        "text_key": _as_text(metadata.get("text_key")),
        "book_family": _as_text(metadata.get("book_family")),
        "display_family": _as_text(metadata.get("display_family")),
        "song_no": _as_text(metadata.get("song_no")),
        "books": _as_list(metadata.get("books")),
        "source_families": _as_list(metadata.get("source_families")),
        "source_note_ids": _as_list(metadata.get("source_note_ids")),
        "urls": _as_list(metadata.get("urls")),
        "tags": _as_list(metadata.get("tags")),
        # These are intentionally explicit fields. Empty values mean the
        # source does not currently assert the fact; they are not guesses.
        "tune": _first_text(metadata, "tune", "tune_name"),
        "meter": _first_text(metadata, "meter", "metre"),
        "key_signature": _first_text(metadata, "key_signature", "key"),
        "composer": _first_text(metadata, "composer", "music_by"),
        "lyricist": _first_text(metadata, "lyricist", "lyrics_by", "author"),
        "book_edition": _first_text(metadata, "book_edition", "edition"),
        "frontmatter": metadata,
    }
    if kind == "song" and not normalized_metadata["tune"]:
        # Current generated song titles are tune titles. Keep the inference
        # visible so a later authoritative tune field can supersede it.
        normalized_metadata["tune"] = title
        normalized_metadata["tune_source"] = "inferred from song title"
    return {
        "document_id": stable_document_id(kind, metadata, relative_path),
        "kind": kind,
        "title": title,
        "relative_path": relative_path,
        "source_path": str(path),
        "checksum": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "metadata": normalized_metadata,
        "body": clean_body,
    }


def _read_source_text(path: Path, timeout_seconds: float = 5.0) -> str:
    """Read a local note without hanging forever on an offloaded iCloud file."""

    def _timeout(_signum: int, _frame: Any) -> None:
        raise UnreadableSourceError(f"timed out reading {path}")

    previous_handler = signal.signal(signal.SIGALRM, _timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return path.read_text(encoding="utf-8")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE documents (
            document_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            title TEXT NOT NULL,
            relative_path TEXT NOT NULL UNIQUE,
            source_path TEXT NOT NULL,
            checksum TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            indexed_at TEXT NOT NULL
        );
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            heading TEXT NOT NULL,
            text TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            UNIQUE(document_id, ordinal)
        );
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            chunk_id UNINDEXED,
            document_id UNINDEXED,
            title,
            heading,
            text,
            keywords
        );
        CREATE TABLE embeddings (
            chunk_id TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
            model TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            vector_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE song_metadata (
            document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
            book_id TEXT NOT NULL,
            song_no TEXT NOT NULL,
            tune TEXT NOT NULL,
            meter TEXT NOT NULL,
            key_signature TEXT NOT NULL,
            composer TEXT NOT NULL,
            lyricist TEXT NOT NULL,
            book_edition TEXT NOT NULL,
            time_signature TEXT NOT NULL,
            source_url TEXT NOT NULL,
            tradition TEXT NOT NULL,
            edition_year TEXT NOT NULL,
            metadata_source TEXT NOT NULL,
            confidence TEXT NOT NULL,
            notes TEXT NOT NULL,
            PRIMARY KEY(document_id, book_id)
        );
        CREATE TABLE book_editions (
            book_id TEXT PRIMARY KEY,
            edition_name TEXT NOT NULL,
            tradition TEXT NOT NULL,
            edition_year TEXT NOT NULL
        );
        CREATE TABLE manifest (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE INDEX chunks_document_idx ON chunks(document_id);
        CREATE INDEX documents_kind_idx ON documents(kind);
        CREATE INDEX documents_text_key_idx ON documents(json_extract(metadata_json, '$.text_key'));
        """
    )


def _load_enrichments(paths: Iterable[Path]) -> dict[tuple[str, str], dict[str, str]]:
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                book_id = _as_text(row.get("book_id") or row.get("source_book_id")).strip().lower()
                song_no = _as_text(row.get("song_no") or row.get("source_song_no")).strip().lower()
                if book_id and song_no:
                    rows[(book_id, song_no)] = {key: _as_text(value).strip() for key, value in row.items()}
    return rows


def _book_ids(metadata: dict[str, Any]) -> list[str]:
    values = metadata.get("books") or metadata.get("source_families") or metadata.get("book_family")
    return sorted(set(_as_list(values)))


def _song_metadata_rows(record: dict[str, Any], enrichments: dict[tuple[str, str], dict[str, str]]) -> list[dict[str, str]]:
    metadata = record["metadata"]
    if record["kind"] != "song":
        return []
    book_ids = _book_ids(metadata) or ["unknown"]
    rows: list[dict[str, str]] = []
    for book_id in book_ids:
        book = BOOK_EDITIONS.get(book_id, {"edition_name": book_id, "tradition": book_id, "year": ""})
        enrichment = enrichments.get((book_id.lower(), _as_text(metadata.get("song_no")).lower()), {})
        explicit_edition = _as_text(metadata.get("book_edition"))
        tune = _as_text(metadata.get("tune")) or _as_text(enrichment.get("tune") or enrichment.get("tune_name")) or record["title"]
        meter = _as_text(metadata.get("meter")) or _as_text(enrichment.get("meter") or enrichment.get("metre"))
        key_signature = _as_text(metadata.get("key_signature")) or _as_text(enrichment.get("key_signature") or enrichment.get("key"))
        composer = _as_text(metadata.get("composer")) or _as_text(enrichment.get("composer"))
        lyricist = _as_text(metadata.get("lyricist")) or _as_text(enrichment.get("lyricist") or enrichment.get("author"))
        book_edition = explicit_edition or _as_text(enrichment.get("book_edition") or enrichment.get("edition")) or book["edition_name"]
        time_signature = _as_text(metadata.get("time_signature")) or _as_text(enrichment.get("time_signature") or enrichment.get("time"))
        source_url = _as_text(enrichment.get("source_url") or enrichment.get("url"))
        provided_fields = [meter, key_signature, composer, lyricist, time_signature]
        source = "frontmatter" if any(metadata.get(field) for field in ("meter", "key_signature", "composer", "lyricist", "book_edition", "time_signature")) else "enrichment CSV" if enrichment else "derived from book ID/title"
        confidence = "source" if source == "frontmatter" else _as_text(enrichment.get("confidence")) or "enriched" if source == "enrichment CSV" else "derived"
        rows.append({
            "document_id": record["document_id"],
            "book_id": book_id,
            "song_no": _as_text(metadata.get("song_no")),
            "tune": tune,
            "meter": meter,
            "key_signature": key_signature,
            "composer": composer,
            "lyricist": lyricist,
            "book_edition": book_edition,
            "time_signature": time_signature,
            "source_url": source_url,
            "tradition": book["tradition"],
            "edition_year": book["year"],
            "metadata_source": source,
            "confidence": confidence,
            "notes": "Tune is inferred from the song title." if not metadata.get("tune") and not enrichment.get("tune") and tune == record["title"] else "",
        })
    return rows


def build_index(vault: Path, database: Path, includes: list[str], excludes: list[str], max_chars: int = DEFAULT_MAX_CHARS, enrichment_paths: Iterable[Path] = (), read_timeout_seconds: float = 5.0) -> dict[str, Any]:
    vault = vault.resolve()
    files = _iter_source_files(vault, includes, excludes)
    records: list[dict[str, Any]] = []
    skipped_files: list[dict[str, str]] = []
    for path in files:
        try:
            records.append(_record_for_file(path, vault, read_timeout_seconds))
        except (OSError, UnicodeError, UnreadableSourceError) as error:
            skipped_files.append({"relative_path": path.relative_to(vault).as_posix(), "error": str(error)})
    enrichments = _load_enrichments(enrichment_paths)
    generated_at = datetime.now(timezone.utc).isoformat()
    database.parent.mkdir(parents=True, exist_ok=True)
    temp_database = database.with_name(f".{database.name}.tmp")
    if temp_database.exists():
        temp_database.unlink()
    connection = sqlite3.connect(temp_database)
    try:
        _create_schema(connection)
        document_count = 0
        chunk_count = 0
        kind_counts: dict[str, int] = {}
        song_metadata_count = 0
        field_coverage = {field: 0 for field in ("tune", "meter", "key_signature", "composer", "lyricist", "book_edition", "time_signature")}
        for book_id, book in BOOK_EDITIONS.items():
            connection.execute("INSERT INTO book_editions VALUES (?, ?, ?, ?)", (book_id, book["edition_name"], book["tradition"], book["year"]))
        for record in records:
            metadata_json = _json_dumps(record["metadata"])
            connection.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    record["document_id"],
                    record["kind"],
                    record["title"],
                    record["relative_path"],
                    record["source_path"],
                    record["checksum"],
                    metadata_json,
                    generated_at,
                ),
            )
            document_count += 1
            kind_counts[record["kind"]] = kind_counts.get(record["kind"], 0) + 1
            chunks = chunk_markdown(record["body"], max_chars=max_chars)
            for ordinal, chunk in enumerate(chunks):
                chunk_id = f"{record['document_id']}:{ordinal}"
                chunk_metadata = dict(record["metadata"])
                chunk_metadata.update({"document_id": record["document_id"], "chunk_ordinal": ordinal})
                chunk_metadata_json = _json_dumps(chunk_metadata)
                connection.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                    (chunk_id, record["document_id"], ordinal, chunk["heading"], chunk["text"], chunk_metadata_json),
                )
                keywords = " ".join(
                    [
                        record["kind"],
                        record["title"],
                        _as_text(record["metadata"].get("text_key")),
                        _as_text(record["metadata"].get("book_family")),
                        " ".join(record["metadata"].get("books", [])),
                        " ".join(record["metadata"].get("source_families", [])),
                    ]
                )
                connection.execute(
                    "INSERT INTO chunks_fts VALUES (?, ?, ?, ?, ?, ?)",
                    (chunk_id, record["document_id"], record["title"], chunk["heading"], chunk["text"], keywords),
                )
                chunk_count += 1
            for song_row in _song_metadata_rows(record, enrichments):
                connection.execute(
                    "INSERT INTO song_metadata VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    tuple(song_row[field] for field in (
                        "document_id", "book_id", "song_no", "tune", "meter", "key_signature", "composer", "lyricist",
                        "book_edition", "time_signature", "source_url", "tradition", "edition_year", "metadata_source", "confidence", "notes",
                    )),
                )
                song_metadata_count += 1
                for field in field_coverage:
                    if song_row[field]:
                        field_coverage[field] += 1
        manifest = {
            "index_version": INDEX_VERSION,
            "generated_at": generated_at,
            "vault": str(vault),
            "includes": includes,
            "excludes": excludes,
            "documents": document_count,
            "chunks": chunk_count,
            "skipped_files": skipped_files,
            "skipped_count": len(skipped_files),
            "song_metadata_rows": song_metadata_count,
            "field_coverage": field_coverage,
            "structured_fields": ["tune", "meter", "key_signature", "composer", "lyricist", "book_edition", "time_signature"],
            "kind_counts": kind_counts,
            "embedding_status": "empty; populate embeddings table with a local model when desired",
            "chunking": {"max_chars": max_chars, "overlap_chars": DEFAULT_OVERLAP_CHARS},
        }
        for key, value in manifest.items():
            connection.execute(
                "INSERT INTO manifest VALUES (?, ?)",
                (key, json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value),
            )
        connection.commit()
    finally:
        connection.close()
    temp_database.replace(database)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vault", type=Path, required=True, help="Obsidian vault root")
    parser.add_argument("--database", type=Path, required=True, help="SQLite database to create")
    parser.add_argument("--include", action="append", dest="includes", default=None, help="Vault-relative glob; repeatable")
    parser.add_argument("--exclude", action="append", dest="excludes", default=[], help="Vault-relative glob to exclude; repeatable")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--read-timeout", type=float, default=5.0, help="Seconds allowed per Markdown file; protects against offloaded iCloud files")
    parser.add_argument("--enrichment-csv", action="append", type=Path, default=[], help="CSV with book_id, song_no, and optional tune/meter/time_signature/key_signature/composer/lyricist/book_edition/source_url fields; repeatable")
    args = parser.parse_args()
    includes = args.includes or [DEFAULT_INCLUDE]
    manifest = build_index(args.vault.resolve(), args.database.resolve(), includes, args.excludes, args.max_chars, args.enrichment_csv, args.read_timeout)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
