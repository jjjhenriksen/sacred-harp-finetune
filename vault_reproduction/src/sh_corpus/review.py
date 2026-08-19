from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .extractor import (
    apply_text_equivalence_override,
    build_text_record,
    find_matching_text_id,
    index_stanza_match_metadata,
    load_text_equivalence_overrides,
    stanza_id_for_text,
)
from .storage import (
    DEFAULT_EXTRACTION_REVIEW_PATH,
    DEFAULT_STANZAS_PATH,
    load_data_file,
    save_data_file,
)

DEFAULT_EXTRACTION_REVIEW_RESOLVED_PATH = Path("data/extraction_review_resolved.yaml")
DEFAULT_REVIEW_TRIAGE_PATH = Path("data/review_triage.yaml")
JUNK_MARKERS = (
    "index of first lines",
    "songs (general index)",
    "songs",
    "lieder / songs",
    "sorry, this song is not available",
    "blackletter",
    "recordings",
    "the pdf can be found",
)
RELIGIOUS_MARKERS = (
    "lord",
    "jesus",
    "god",
    "grace",
    "heaven",
    "savior",
    "soul",
    "lamb",
    "zion",
    "grave",
    "prayer",
)


def review_item_id(item: dict[str, Any]) -> str:
    basis = "\n".join(
        [
            str(item.get("book_id", "")),
            str(item.get("song_no", "")),
            str(item.get("title", "")),
            str(item.get("first_line", "")),
            str(item.get("raw_lyrics", "")),
            str(item.get("review_reason", "")),
        ]
    )
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:12]


def load_review_items(path: str | Path = DEFAULT_EXTRACTION_REVIEW_PATH) -> list[dict[str, Any]]:
    rows = load_data_file(path, default=[])
    return [dict(row) for row in rows]


def load_review_resolutions(path: str | Path = DEFAULT_EXTRACTION_REVIEW_RESOLVED_PATH) -> list[dict[str, Any]]:
    rows = load_data_file(path, default=[])
    return [dict(row) for row in rows]


def save_review_resolutions(rows: list[dict[str, Any]], path: str | Path = DEFAULT_EXTRACTION_REVIEW_RESOLVED_PATH) -> None:
    save_data_file(path, rows)


def load_review_triage(path: str | Path = DEFAULT_REVIEW_TRIAGE_PATH) -> list[dict[str, Any]]:
    rows = load_data_file(path, default=[])
    if isinstance(rows, dict):
        rows = rows.get("items", [])
    return [dict(row) for row in rows]


def upsert_review_resolution(
    resolution: dict[str, Any],
    path: str | Path = DEFAULT_EXTRACTION_REVIEW_RESOLVED_PATH,
) -> list[dict[str, Any]]:
    rows = load_review_resolutions(path)
    review_id = resolution["review_id"]
    updated = [row for row in rows if row.get("review_id") != review_id]
    updated.append(dict(resolution))
    updated.sort(key=lambda row: row["review_id"])
    save_review_resolutions(updated, path)
    return updated


def classify_review_item(item: dict[str, Any]) -> tuple[str, str]:
    reason = (item.get("review_reason") or "").strip().lower()
    confidence = (item.get("segmentation_confidence") or "").strip().lower()
    raw = (item.get("raw_lyrics") or "").strip().lower()
    title = (item.get("title") or "").strip().lower()
    proposed_blocks = [str(block).strip().lower() for block in item.get("proposed_blocks", [])]
    combined = "\n".join([raw, title] + proposed_blocks)

    if reason == "suspected-editorial-paratext":
        return "probable_junk", "editorial-paratext-reason"

    if any(marker in combined for marker in JUNK_MARKERS):
        return "probable_junk", "junk-marker-match"

    if raw and sum(1 for line in raw.splitlines() if line.strip()[:1].isdigit()) >= 3:
        return "probable_junk", "index-like-numbered-lines"

    if reason == "irregular-line-count":
        if any(marker in combined for marker in RELIGIOUS_MARKERS) and not any(marker in combined for marker in JUNK_MARKERS):
            return "probable_safe_irregular_lyric", "irregular-but-lyric-looking"
        return "truly_ambiguous", "irregular-without-clear-lyric-signal"

    if confidence == "medium" and proposed_blocks:
        if not any(marker in combined for marker in JUNK_MARKERS):
            return "probable_safe_irregular_lyric", "clean-proposed-segmentation"

    if len(proposed_blocks) == 1 and "chorus" in raw and "chorus" not in proposed_blocks[0]:
        return "truly_ambiguous", "possible-chorus-structure-confusion"

    return "truly_ambiguous", "default-ambiguous"


def build_review_triage(review_items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    triage_rows: list[dict[str, Any]] = []
    summary = {
        "probable_junk": 0,
        "probable_safe_irregular_lyric": 0,
        "truly_ambiguous": 0,
    }
    for item in review_items:
        bucket, rationale = classify_review_item(item)
        review_id = review_item_id(item)
        triage_rows.append(
            {
                "review_id": review_id,
                "book_id": item["book_id"],
                "song_no": item["song_no"],
                "title": item["title"],
                "first_line": item.get("first_line", ""),
                "review_reason": item.get("review_reason", ""),
                "segmentation_confidence": item.get("segmentation_confidence", ""),
                "bucket": bucket,
                "triage_rationale": rationale,
            }
        )
        summary[bucket] += 1
    triage_rows.sort(key=lambda row: (row["bucket"], row["book_id"], row["song_no"], row["review_id"]))
    return triage_rows, summary


def generate_review_triage(
    review_path: str | Path = DEFAULT_EXTRACTION_REVIEW_PATH,
    triage_path: str | Path = DEFAULT_REVIEW_TRIAGE_PATH,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    review_items = load_review_items(review_path)
    triage_rows, summary = build_review_triage(review_items)
    save_data_file(
        triage_path,
        {
            "summary": summary,
            "items": triage_rows,
        },
    )
    return triage_rows, summary


def accepted_blocks_for_resolution(
    review_item: dict[str, Any],
    resolution: dict[str, Any],
) -> list[str]:
    decision = resolution["decision"]
    if decision == "accept":
        return list(review_item.get("proposed_blocks", []))
    if decision == "edit":
        return list(resolution.get("resolved_blocks", []))
    return []


def merge_resolution_into_stanzas(
    stanza_rows: list[dict[str, Any]],
    review_item: dict[str, Any],
    resolution: dict[str, Any],
    text_equivalence_overrides: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    blocks = accepted_blocks_for_resolution(review_item, resolution)
    if not blocks:
        return stanza_rows

    by_id = {row["id"]: dict(row) for row in stanza_rows}
    match_index: dict[tuple[object, ...], set[str]] = {}
    match_meta: dict[str, dict[str, object]] = {}
    for row in stanza_rows:
        text_id = str(row.get("id", ""))
        text = str(row.get("text", ""))
        if not text_id or not text.strip():
            continue
        index_stanza_match_metadata(text_id, text, match_index=match_index, match_meta=match_meta)
    associated_tune = {
        "book": review_item["book_id"],
        "page": review_item["song_no"],
        "tune": review_item.get("title", ""),
        "url": "",
    }
    for block_text in blocks:
        text = (block_text or "").strip()
        if not text:
            continue
        first_line = text.splitlines()[0].strip()
        text, first_line = apply_text_equivalence_override(text, first_line, text_equivalence_overrides or {})
        stanza_id = find_matching_text_id(
            by_id,
            text,
            first_line,
            match_index=match_index,
            match_meta=match_meta,
        ) or stanza_id_for_text(text, first_line)
        existing = by_id.get(stanza_id)
        if existing is None:
            by_id[stanza_id] = build_text_record(
                text_id=stanza_id,
                first_line=first_line,
                text=text,
                associated_tunes=[associated_tune],
            )
            index_stanza_match_metadata(stanza_id, text, match_index=match_index, match_meta=match_meta)
            continue
        tunes = list(existing.get("associated_tunes", []))
        if associated_tune not in tunes:
            tunes.append(associated_tune)
            tunes.sort(key=lambda item: (item.get("book", ""), item.get("page", ""), item.get("tune", "")))
            existing["associated_tunes"] = tunes
        existing["associated_songs"] = sorted(
            {f"{item['book']}:{item['page']}" for item in existing.get("associated_tunes", []) if item.get("book") and item.get("page")}
        )
        existing["source_books"] = sorted({item["book"] for item in existing.get("associated_tunes", []) if item.get("book")})
    return sorted(by_id.values(), key=lambda row: (row["first_line"].lower(), row["id"]))


def apply_review_resolutions(
    *,
    stanzas_path: str | Path = DEFAULT_STANZAS_PATH,
    review_path: str | Path = DEFAULT_EXTRACTION_REVIEW_PATH,
    resolutions_path: str | Path = DEFAULT_EXTRACTION_REVIEW_RESOLVED_PATH,
    text_equivalence_overrides_path: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stanza_rows = load_data_file(stanzas_path, default=[])
    review_items = load_review_items(review_path)
    resolutions = {row["review_id"]: row for row in load_review_resolutions(resolutions_path)}
    text_equivalence_overrides = load_text_equivalence_overrides(text_equivalence_overrides_path)

    remaining_review: list[dict[str, Any]] = []
    for item in review_items:
        item_id = review_item_id(item)
        resolution = resolutions.get(item_id)
        if resolution is None or resolution.get("decision") == "skip":
            remaining_review.append(item)
            continue
        if resolution["decision"] in {"accept", "edit"}:
            stanza_rows = merge_resolution_into_stanzas(
                stanza_rows,
                item,
                resolution,
                text_equivalence_overrides=text_equivalence_overrides,
            )
        elif resolution["decision"] == "reject":
            pass
        else:
            remaining_review.append(item)

    save_data_file(stanzas_path, stanza_rows)
    save_data_file(review_path, remaining_review)
    return stanza_rows, remaining_review
