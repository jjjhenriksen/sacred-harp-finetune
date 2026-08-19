from __future__ import annotations

import csv
import hashlib
import html as html_lib
from difflib import SequenceMatcher
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

try:
    from bs4 import BeautifulSoup
    from bs4.element import Tag
except Exception:  # pragma: no cover
    BeautifulSoup = None
    Tag = object

from .shape_note_normalization import norm_first_line
from .storage import DEFAULT_EXTRACTION_REVIEW_PATH, DEFAULT_STANZAS_PATH, load_data_file, save_data_file

CHORUS_MARKERS = {"chorus:", "refrain:"}
SIMILARITY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "my", "of", "on", "or", "our", "that", "the", "their", "this",
    "thy", "to", "we", "with", "will", "your",
}
PAREN_CHORUS_RE = re.compile(r"^\((chorus|refrain)\)$", re.IGNORECASE)
VERSE_HEADER_RE = re.compile(r"^(verse|stanza|v)\s*\d+\s*[:.\-]?\s*$", re.IGNORECASE)
NUMBERED_VERSE_RE = re.compile(r"^\d+[.)]?\s*$")
SONG_HEADER_RE = re.compile(r"^\d+[a-z]?\s+.+$")
STOP_LINE_PREFIXES = ("recordings", "midi", "video", "pdf")
STOP_LINE_EXACT = {"none", "sorry, this song is not available"}
INDEX_HEADER_PATTERNS = (
    "songs",
    "lieder / songs",
    "songs (general index)",
    "songs (general-index)",
    "index of first lines",
)
TITLE_LIST_RE = re.compile(r"^\d+[a-z]?\s+[\w“\"'‘(].+$", re.IGNORECASE)
EDITORIAL_MARKERS = (
    "blackletter",
    "you may want to sing",
    "text sei",
    "heisst es",
    "heißt es",
    "singt",
    "modern days",
    "looks like",
    "the pdf can be found",
    "not available",
)


def is_non_song_row(raw: dict[str, str]) -> bool:
    book_id = (raw.get("book_id") or "").strip().lower()
    song_no = (raw.get("song_no") or "").strip().lower()
    title = (raw.get("title") or "").strip().lower()
    url = (raw.get("url") or "").strip()

    if "edition" in title:
        return True
    if not url:
        return False

    parsed = urlparse(url)
    path = (parsed.path or "").strip()
    is_root = path in ("", "/")
    match = re.search(r"(\d{4})", book_id)
    book_year = match.group(1) if match else ""
    return bool(is_root and book_year and song_no == book_year)


def strip_control_characters(text: str) -> str:
    kept: list[str] = []
    for ch in unicodedata.normalize("NFKC", text or ""):
        if ch in "\n\t":
            kept.append(ch)
            continue
        if unicodedata.category(ch).startswith("C"):
            continue
        kept.append(ch)
    return "".join(kept)


def normalize_line(line: str) -> str:
    text = strip_control_characters(line).strip()
    text = re.sub(r"[ ]+", " ", text)
    text = re.sub(r"\s+([,;:.!?])", r"\1", text)
    return text


def normalize_text(text: str) -> str:
    lowered = unicodedata.normalize("NFKD", text or "").lower()
    lowered = lowered.replace("’", "'")
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def clean_lyrics_text(row: dict[str, str]) -> str:
    song_no = normalize_text(row.get("song_no", ""))
    title = normalize_text(row.get("title", ""))
    raw_lines = [normalize_line(raw_line) for raw_line in strip_control_characters(row.get("lyrics", "")).splitlines()]
    raw_lines = [line for line in raw_lines if line]
    cleaned_lines: list[str] = []
    for raw_line in raw_lines:
        line = normalize_line(raw_line)
        if not line:
            continue
        lowered = normalize_text(line)
        if any(lowered.startswith(prefix) for prefix in STOP_LINE_PREFIXES):
            break
        if lowered in STOP_LINE_EXACT:
            break
        if lowered == title:
            continue
        if lowered == f"{song_no} {title}".strip():
            continue
        if SONG_HEADER_RE.match(line) and title and lowered.endswith(title):
            continue
        cleaned_lines.append(line)
    cleaned_lines = trim_leading_index_block(cleaned_lines)
    return "\n".join(cleaned_lines)


def is_index_header_line(line: str) -> bool:
    lowered = normalize_text(line)
    if lowered == "songs":
        return True
    if "index of first lines" in lowered:
        return True
    if "general index" in lowered:
        return True
    if lowered.startswith("lieder songs"):
        return True
    return lowered in {normalize_text(pattern) for pattern in INDEX_HEADER_PATTERNS}


def is_title_list_line(line: str) -> bool:
    return bool(TITLE_LIST_RE.match(line))


def trim_leading_index_block(lines: list[str]) -> list[str]:
    if not lines:
        return lines
    header_hits = 0
    title_hits = 0
    for line in lines[:8]:
        if is_index_header_line(line):
            header_hits += 1
        elif is_title_list_line(line):
            title_hits += 1
    if header_hits == 0 and title_hits < 3:
        return lines

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if is_index_header_line(line) or is_title_list_line(line):
            idx += 1
            continue
        break
    return lines[idx:]


def slugify(text: str, limit: int = 48) -> str:
    slug = normalize_text(text).replace(" ", "-")
    slug = re.sub(r"-+", "-", slug).strip("-")
    return (slug[:limit].rstrip("-") or "stanza")


def stanza_id_for_text(text: str, first_line: str) -> str:
    digest = hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()[:8]
    return f"stanza_{slugify(first_line)}_{digest}"


def normalized_text_lines(text: str) -> list[str]:
    return [normalize_text(line) for line in text.splitlines() if normalize_text(line)]


def stanza_match_metadata(text: str) -> dict[str, object]:
    lines = normalized_text_lines(text)
    joined = " ".join(lines)
    first_line = lines[0] if lines else ""
    last_line = lines[-1] if lines else ""
    return {
        "lines": lines,
        "joined": joined,
        "line_count": len(lines),
        "keys": stanza_candidate_keys_from_lines(lines),
        "first_line": first_line,
        "last_line": last_line,
        "first_line_prefix": tuple(_content_tokens(first_line)[:4]) if first_line else (),
        "last_line_prefix": tuple(_content_tokens(last_line)[:4]) if last_line else (),
    }


def _content_tokens(line: str) -> list[str]:
    tokens = [token for token in line.split() if token]
    content = [token for token in tokens if token not in SIMILARITY_STOPWORDS]
    return content or tokens


def _fingerprint(line: str) -> tuple[str, ...]:
    return tuple(_content_tokens(line)[:3])


def stanza_candidate_keys_from_lines(lines: list[str]) -> set[tuple[object, ...]]:
    if not lines:
        return set()
    line_count = len(lines)
    first_fp = _fingerprint(lines[0])
    last_fp = _fingerprint(lines[-1])
    keys: set[tuple[object, ...]] = {
        ("last", line_count, last_fp),
        ("first", line_count, first_fp),
        ("first-last", line_count, first_fp[:2], last_fp[:2]),
    }
    if line_count > 2:
        middle_fp = _fingerprint(lines[line_count // 2])
        keys.add(("middle", line_count, middle_fp))
    return keys


def _line_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


def stanzas_are_equivalent(
    left_text: str,
    right_text: str,
    *,
    left_meta: dict[str, object] | None = None,
    right_meta: dict[str, object] | None = None,
) -> bool:
    left_info = left_meta or stanza_match_metadata(left_text)
    right_info = right_meta or stanza_match_metadata(right_text)
    left_lines = list(left_info["lines"])
    right_lines = list(right_info["lines"])
    if not left_lines or not right_lines:
        return False
    if len(left_lines) != len(right_lines):
        return False

    left_joined = str(left_info["joined"])
    right_joined = str(right_info["joined"])
    whole_ratio = _line_similarity(left_joined, right_joined)
    if whole_ratio >= 0.97:
        return True
    if whole_ratio < 0.9:
        return False

    per_line = [_line_similarity(a, b) for a, b in zip(left_lines, right_lines)]
    average_ratio = sum(per_line) / len(per_line)
    min_ratio = min(per_line)
    return average_ratio >= 0.92 and min_ratio >= 0.8


def find_matching_text_id(
    aggregated: dict[str, dict[str, object]],
    text: str,
    first_line: str,
    *,
    match_index: dict[tuple[object, ...], set[str]] | None = None,
    match_meta: dict[str, dict[str, object]] | None = None,
    stats: dict[str, int] | None = None,
) -> str | None:
    candidate_meta = stanza_match_metadata(text)
    exact_id = stanza_id_for_text(text, first_line)
    if exact_id in aggregated:
        if stats is not None:
            stats["exact_id_hits"] = stats.get("exact_id_hits", 0) + 1
        return exact_id
    candidate_ids: list[str]
    if match_index is not None:
        candidate_sets = [match_index.get(key, set()) for key in candidate_meta["keys"] if match_index.get(key)]
        candidate_sets.sort(key=len)
        if candidate_sets:
            combined = set(candidate_sets[0])
            for bucket in candidate_sets[1:]:
                overlap = combined.intersection(bucket)
                if overlap:
                    combined = overlap
                if len(combined) <= 8:
                    break
            candidate_ids = sorted(combined)
        else:
            candidate_ids = []
    else:
        candidate_ids = list(aggregated.keys())
    if stats is not None:
        stats["candidate_lookups"] = stats.get("candidate_lookups", 0) + 1
        stats["candidate_total"] = stats.get("candidate_total", 0) + len(candidate_ids)
        stats["candidate_max"] = max(stats.get("candidate_max", 0), len(candidate_ids))
    for text_id in candidate_ids:
        row = aggregated[text_id]
        existing_text = str(row.get("text", ""))
        existing_meta = match_meta.get(text_id) if match_meta is not None else None
        if existing_meta is not None:
            if existing_meta.get("first_line_prefix") != candidate_meta.get("first_line_prefix"):
                if existing_meta.get("last_line_prefix") != candidate_meta.get("last_line_prefix"):
                    continue
        if stanzas_are_equivalent(existing_text, text, left_meta=existing_meta, right_meta=candidate_meta):
            if stats is not None:
                stats["fuzzy_matches"] = stats.get("fuzzy_matches", 0) + 1
            return text_id
    return None


def index_stanza_match_metadata(
    text_id: str,
    text: str,
    *,
    match_index: dict[tuple[object, ...], set[str]],
    match_meta: dict[str, dict[str, object]],
) -> None:
    meta = stanza_match_metadata(text)
    match_meta[text_id] = meta
    for key in meta["keys"]:
        match_index.setdefault(key, set()).add(text_id)


def normalize_multiline_text(text: str) -> str:
    return "\n".join(normalized_text_lines(text))


def load_text_equivalence_overrides(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    candidate = Path(path)
    if not candidate.exists():
        return {}
    rows = load_data_file(candidate, default=[])
    if not rows:
        return {}

    mapping: dict[str, str] = {}
    for row in rows:
        canonical_text = str(row.get("canonical_text", "")).strip()
        if not canonical_text:
            continue
        variants = [canonical_text] + [str(item).strip() for item in row.get("equivalent_texts", []) or [] if str(item).strip()]
        for variant in variants:
            mapping[normalize_multiline_text(variant)] = canonical_text
    return mapping


def apply_text_equivalence_override(
    text: str,
    first_line: str,
    overrides: dict[str, str],
) -> tuple[str, str]:
    canonical_text = overrides.get(normalize_multiline_text(text), text)
    canonical_first_line = canonical_text.splitlines()[0].strip() if canonical_text.strip() else first_line
    return canonical_text, canonical_first_line


def build_near_duplicate_report(
    stanza_rows: list[dict[str, object]],
    *,
    report_threshold: float = 0.88,
) -> list[dict[str, object]]:
    report: list[dict[str, object]] = []
    rows_by_id = {str(row.get("id", "")): row for row in stanza_rows}
    match_index: dict[tuple[object, ...], set[str]] = {}
    match_meta: dict[str, dict[str, object]] = {}
    for row in stanza_rows:
        text_id = str(row.get("id", ""))
        text = str(row.get("text", ""))
        if not text_id or not text.strip():
            continue
        index_stanza_match_metadata(text_id, text, match_index=match_index, match_meta=match_meta)

    compared_pairs: set[tuple[str, str]] = set()
    for key, ids in match_index.items():
        candidate_ids = sorted(ids)
        if len(candidate_ids) < 2:
            continue
        for index, left_id in enumerate(candidate_ids):
            left = rows_by_id[left_id]
            left_meta = match_meta[left_id]
            left_text = str(left.get("text", ""))
            left_lines = list(left_meta["lines"])
            if not left_lines:
                continue
            for right_id in candidate_ids[index + 1 :]:
                pair = (left_id, right_id)
                if pair in compared_pairs:
                    continue
                compared_pairs.add(pair)
                right = rows_by_id[right_id]
                right_meta = match_meta[right_id]
                right_text = str(right.get("text", ""))
                right_lines = list(right_meta["lines"])
                if not right_lines or len(left_lines) != len(right_lines):
                    continue
                if stanzas_are_equivalent(left_text, right_text, left_meta=left_meta, right_meta=right_meta):
                    continue
                whole_ratio = _line_similarity(str(left_meta["joined"]), str(right_meta["joined"]))
                if whole_ratio < report_threshold:
                    continue
                report.append(
                    {
                        "left_id": left.get("id", ""),
                        "right_id": right.get("id", ""),
                        "left_incipit": left.get("incipit") or left.get("first_line", ""),
                        "right_incipit": right.get("incipit") or right.get("first_line", ""),
                        "similarity": round(whole_ratio, 4),
                        "left_associated_songs": list(left.get("associated_songs", []) or []),
                        "right_associated_songs": list(right.get("associated_songs", []) or []),
                    }
                )
    report.sort(key=lambda row: (-float(row["similarity"]), str(row["left_id"]), str(row["right_id"])))
    return report


def split_on_blank_lines(raw_text: str) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []
    saw_blank = False
    for raw_line in raw_text.splitlines():
        line = normalize_line(raw_line)
        if not line:
            saw_blank = True
            if current:
                groups.append(current)
                current = []
            continue
        current.append(line)
    if current:
        groups.append(current)
    return groups if saw_blank and groups else []


def _flush_buffer(blocks: list[list[str]], buffer: list[str]) -> None:
    if buffer:
        blocks.append(list(buffer))
        buffer.clear()


def split_on_markers(lines: list[str]) -> list[list[str]]:
    blocks: list[list[str]] = []
    buffer: list[str] = []
    for line in lines:
        lowered = line.lower()
        if lowered in CHORUS_MARKERS or PAREN_CHORUS_RE.match(line):
            _flush_buffer(blocks, buffer)
            if lowered in CHORUS_MARKERS:
                buffer.append(line)
            elif blocks:
                blocks[-1].append(line)
            else:
                buffer.append(line)
            continue
        if VERSE_HEADER_RE.match(line) or NUMBERED_VERSE_RE.match(line):
            _flush_buffer(blocks, buffer)
            continue
        buffer.append(line)
    _flush_buffer(blocks, buffer)
    return blocks


def split_with_meter_fallback(lines: list[str]) -> tuple[list[list[str]], str, str | None]:
    content_lines = [line for line in lines if line.lower() not in CHORUS_MARKERS and not PAREN_CHORUS_RE.match(line)]
    if not content_lines:
        return [], "low", "no-content-lines"
    count = len(content_lines)
    if count % 4 == 0:
        return [content_lines[i : i + 4] for i in range(0, count, 4)], "medium", None
    if count % 2 == 0 and count <= 8:
        return [content_lines[i : i + 2] for i in range(0, count, 2)], "medium", None
    return [content_lines], "low", "irregular-line-count"


def propose_blocks(raw_text: str) -> tuple[list[list[str]], str, str | None]:
    explicit = split_on_blank_lines(raw_text)
    if explicit:
        return explicit, "high", None
    lines = [normalize_line(line) for line in raw_text.splitlines() if normalize_line(line)]
    marker_blocks = split_on_markers(lines)
    if len(marker_blocks) > 1:
        return marker_blocks, "high", None
    return split_with_meter_fallback(lines)


def build_review_entry(row: dict[str, str], blocks: list[list[str]], confidence: str, reason: str | None) -> dict[str, object]:
    return {
        "book_id": row["book_id"],
        "song_no": row["song_no"],
        "title": row["title"],
        "first_line": row.get("first_lyric_line") or "",
        "raw_lyrics": row.get("lyrics", ""),
        "proposed_blocks": ["\n".join(block) for block in blocks],
        "segmentation_confidence": confidence,
        "review_reason": reason or "",
    }


def review_entry_for_block(
    row: dict[str, str],
    block: list[str],
    reason: str,
    confidence: str = "low",
) -> dict[str, object]:
    return build_review_entry(row, [block], confidence, reason)


def review_entry_for_song(
    row: dict[str, str],
    reason: str,
    confidence: str = "low",
) -> dict[str, object]:
    return {
        "book_id": row["book_id"],
        "song_no": row["song_no"],
        "title": row["title"],
        "first_line": row.get("first_lyric_line") or row.get("first_line") or "",
        "raw_lyrics": row.get("lyrics", ""),
        "proposed_blocks": [],
        "segmentation_confidence": confidence,
        "review_reason": reason,
    }


def classify_candidate_block(row: dict[str, str], block: list[str]) -> tuple[str, str | None]:
    if not block:
        return "exclude", "empty-block"
    lowered_lines = [normalize_text(line) for line in block]
    joined = "\n".join(lowered_lines)

    if all(is_title_list_line(line) or is_index_header_line(line) for line in block):
        return "exclude", "index-or-title-list"
    if len(block) == 1:
        only = lowered_lines[0]
        song_title = normalize_text(f"{row.get('song_no', '')} {row.get('title', '')}")
        if only == song_title or only == normalize_text(row.get("title", "")):
            return "exclude", "title-only-header"
    if any(marker in joined for marker in EDITORIAL_MARKERS):
        return "review", "suspected-editorial-paratext"
    if sum(1 for line in block if is_title_list_line(line)) >= max(3, len(block) // 2):
        return "review", "mixed-title-list-paratext"
    return "keep", None


def split_editorial_tail(block: list[str]) -> tuple[list[str], list[str]]:
    editorial_start = None
    for idx, line in enumerate(block):
        lowered = normalize_text(line)
        if any(marker in lowered for marker in EDITORIAL_MARKERS):
            editorial_start = idx
            break
    if editorial_start is None:
        return block, []
    return block[:editorial_start], block[editorial_start:]


def _cooper_cache_path(cache_dir: Path, url: str) -> Path:
    return cache_dir / url.rstrip("/").split("/")[-1]


def cooper_poetry_url_candidates(url: str) -> list[str]:
    basename = url.rstrip("/").split("/")[-1]
    candidates = [url]
    index_url = f"http://resources.texasfasola.org/index/poetry/{basename}"
    if index_url not in candidates:
        candidates.append(index_url)
    return candidates


def extract_cooper_poetry_lyrics(html: str) -> str:
    if BeautifulSoup is None:
        return _extract_cooper_poetry_lyrics_fallback(html)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()
    pieces: list[str] = []
    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue
        label = normalize_line(tds[0].get_text(" ", strip=True))
        stanza_td = tds[2]
        stanza_lines = [normalize_line(x) for x in stanza_td.get_text("\n", strip=True).splitlines()]
        stanza_lines = [line for line in stanza_lines if line]
        if not stanza_lines:
            continue
        if label.lower() in {"chorus", "refrain"}:
            pieces.append("Chorus:")
        elif re.fullmatch(r"\d+[.)]?", label):
            pieces.append(f"{label}.")
        pieces.extend(stanza_lines)
    if pieces:
        return "\n".join(pieces)

    body = soup.body
    if body is None:
        return ""

    hr_count = 0
    collected: list[str] = []
    for child in body.children:
        if isinstance(child, Tag) and child.name == "hr":
            hr_count += 1
            if hr_count >= 3:
                break
            continue
        if hr_count < 2:
            continue
        collected.append(str(child))

    if not collected:
        return ""

    fragment = "".join(collected)
    fragment = re.sub(r"(?i)<br\s*/?>", "\n", fragment)
    text = BeautifulSoup(fragment, "html.parser").get_text(" ", strip=False)
    raw_lines = text.splitlines()
    lines: list[str] = []
    blank_streak = 0
    for raw_line in raw_lines:
        line = normalize_line(raw_line)
        if not line:
            blank_streak += 1
            continue
        if blank_streak >= 2 and lines and lines[-1] != "":
            lines.append("")
        blank_streak = 0
        lines.append(line)

    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def _html_to_text_preserving_breaks(fragment: str) -> str:
    text = re.sub(r"(?i)<br\s*/?>", "\n", fragment)
    text = re.sub(r"(?i)</?(?:p|div|h\d|tr|table|body|html|td|th)>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", "", text)
    return html_lib.unescape(text)


def _extract_table_poetry_rows(html: str) -> list[tuple[str, list[str]]]:
    rows: list[tuple[str, list[str]]] = []
    for tr_match in re.finditer(r"(?is)<tr\b[^>]*>(.*?)</tr>", html):
        cells = re.findall(r"(?is)<td\b[^>]*>(.*?)</td>", tr_match.group(1))
        if len(cells) < 3:
            continue
        label = normalize_line(_html_to_text_preserving_breaks(cells[0]).replace("\n", " "))
        stanza_lines = [
            normalize_line(line)
            for line in _html_to_text_preserving_breaks(cells[2]).splitlines()
            if normalize_line(line)
        ]
        if stanza_lines:
            rows.append((label, stanza_lines))
    return rows


def _extract_cooper_poetry_lyrics_fallback(html: str) -> str:
    pieces: list[str] = []
    for label, stanza_lines in _extract_table_poetry_rows(html):
        if label.lower() in {"chorus", "refrain"}:
            pieces.append("Chorus:")
        elif re.fullmatch(r"\d+[.)]?", label):
            pieces.append(f"{label.rstrip('.')}.")
        pieces.extend(stanza_lines)
    if pieces:
        return "\n".join(pieces)

    body_match = re.search(r"(?is)<body\b[^>]*>(.*?)</body>", html)
    fragment = body_match.group(1) if body_match else html
    parts = re.split(r"(?is)<hr\b[^>]*\/?>", fragment)
    if len(parts) >= 3:
        fragment = parts[2]
    text = _html_to_text_preserving_breaks(fragment)
    raw_lines = text.splitlines()
    lines: list[str] = []
    blank_streak = 0
    for raw_line in raw_lines:
        line = normalize_line(raw_line)
        if not line:
            blank_streak += 1
            continue
        if blank_streak >= 2 and lines and lines[-1] != "":
            lines.append("")
        blank_streak = 0
        lines.append(line)
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def choose_denson_lyrics_row(
    cooper_row: dict[str, str],
    matches: list[dict[str, str]],
) -> dict[str, str] | None:
    if not matches:
        return None
    title_norm = normalize_text(cooper_row.get("title", ""))
    song_no = (cooper_row.get("song_no") or "").strip().lower()
    same_title = [row for row in matches if normalize_text(row.get("title", "")) == title_norm]
    same_song = [row for row in matches if (row.get("song_no") or "").strip().lower() == song_no]
    pool = same_song or same_title or matches
    text_counts = Counter((row.get("lyrics") or "").strip() for row in pool if (row.get("lyrics") or "").strip())
    if text_counts:
        preferred_text = text_counts.most_common(1)[0][0]
        pool = [row for row in pool if (row.get("lyrics") or "").strip() == preferred_text]
    pool.sort(key=lambda row: (0 if row.get("book_id") == "sh2025" else 1, row.get("song_no", "")))
    return pool[0]


def build_cooper_lyrics_rows(
    cooper_first_lines_csv: str | Path,
    denson_rows: list[dict[str, str]],
    cache_dir: str | Path = ".cache/texasfasola_cooper/resources.texasfasola.org/poetry/cooper",
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    cache_root = Path(cache_dir)
    denson_by_first_line: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in denson_rows:
        if row.get("book_id") not in {"sh1991", "sh2025"}:
            continue
        key = norm_first_line(row.get("first_lyric_line") or "")
        if key:
            denson_by_first_line[key].append(row)

    cooper_rows: list[dict[str, str]] = []
    review_entries: list[dict[str, object]] = []
    with Path(cooper_first_lines_csv).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            first_line_norm = row.get("first_line_norm") or norm_first_line(row.get("first_line_raw") or "")
            denson_match = choose_denson_lyrics_row(row, denson_by_first_line.get(first_line_norm, []))
            if denson_match is not None:
                cooper_rows.append(
                    {
                        "book_id": row["book_id"],
                        "song_no": row["song_no"],
                        "title": row["title"],
                        "url": row["url"],
                        "first_lyric_line": denson_match.get("first_lyric_line") or row.get("first_line_raw") or "",
                        "lyrics": denson_match.get("lyrics") or "",
                    }
                )
                continue

            cache_path = _cooper_cache_path(cache_root, row["url"])
            if not cache_path.exists():
                review_entries.append(
                    review_entry_for_song(
                        {
                            "book_id": row["book_id"],
                            "song_no": row["song_no"],
                            "title": row["title"],
                            "url": row["url"],
                            "first_line": row.get("first_line_raw") or "",
                            "lyrics": "",
                        },
                        "missing-cooper-poetry-page",
                    )
                )
                continue

            html = cache_path.read_text(encoding="utf-8", errors="ignore")
            lyrics = extract_cooper_poetry_lyrics(html).strip()
            if not lyrics:
                review_entries.append(
                    review_entry_for_song(
                        {
                            "book_id": row["book_id"],
                            "song_no": row["song_no"],
                            "title": row["title"],
                            "url": row["url"],
                            "first_line": row.get("first_line_raw") or "",
                            "lyrics": "",
                        },
                        "empty-cooper-poetry-page",
                    )
                )
                continue

            first_line = next(
                (
                    line
                    for line in lyrics.splitlines()
                    if line and line.lower() not in CHORUS_MARKERS and not NUMBERED_VERSE_RE.match(line)
                ),
                row.get("first_line_raw") or "",
            )
            cooper_rows.append(
                {
                    "book_id": row["book_id"],
                    "song_no": row["song_no"],
                    "title": row["title"],
                    "url": row["url"],
                    "first_lyric_line": first_line,
                    "lyrics": lyrics,
                }
            )
    return cooper_rows, review_entries


def _representative_books(books: Iterable[str]) -> list[str]:
    return sorted(set(books))


def _associated_tune_from_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "book": row["book_id"],
        "page": row["song_no"],
        "tune": row.get("title", ""),
        "url": row.get("url", ""),
    }


def _associated_song_keys(associated_tunes: list[dict[str, str]]) -> list[str]:
    return sorted({f"{item['book']}:{item['page']}" for item in associated_tunes if item.get("book") and item.get("page")})


def _source_books_for_tunes(associated_tunes: list[dict[str, str]]) -> list[str]:
    return sorted({item["book"] for item in associated_tunes if item.get("book")})


def build_text_record(
    *,
    text_id: str,
    first_line: str,
    text: str,
    associated_tunes: list[dict[str, str]],
) -> dict[str, object]:
    return {
        "id": text_id,
        "text_id": text_id,
        "incipit": first_line,
        "first_line": first_line,
        "text": text,
        "associated_tunes": associated_tunes,
        "associated_songs": _associated_song_keys(associated_tunes),
        "source_books": _source_books_for_tunes(associated_tunes),
        "theme_tags": [],
        "tone_tags": [],
        "operational_fit": [],
        "tag_rationale": "",
    }


def extract_stanzas(
    lyrics_csv_path: str | Path,
    stanzas_path: str | Path = DEFAULT_STANZAS_PATH,
    review_path: str | Path = DEFAULT_EXTRACTION_REVIEW_PATH,
    extra_lyrics_csv_paths: list[str | Path] | None = None,
    cooper_first_lines_csv: str | Path | None = "cooper2012_first_lines_raw_texasfasola.csv",
    cooper_cache_dir: str | Path = ".cache/texasfasola_cooper/resources.texasfasola.org/poetry/cooper",
    text_equivalence_overrides_path: str | Path | None = None,
    progress: bool = False,
    progress_every: int = 250,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    started_at = time.perf_counter()

    def log(message: str) -> None:
        if not progress:
            return
        elapsed = time.perf_counter() - started_at
        print(f"[extract +{elapsed:0.2f}s] {message}", file=sys.stderr, flush=True)

    aggregated: dict[str, dict[str, object]] = {}
    aggregated_match_index: dict[tuple[object, ...], set[str]] = {}
    aggregated_match_meta: dict[str, dict[str, object]] = {}
    match_stats: dict[str, int] = {}
    review_entries: list[dict[str, object]] = []

    phase_started = time.perf_counter()
    text_equivalence_overrides = load_text_equivalence_overrides(text_equivalence_overrides_path)
    log(f"loaded overrides: {len(text_equivalence_overrides)} variant entries in {time.perf_counter() - phase_started:0.2f}s")

    phase_started = time.perf_counter()
    base_rows: list[dict[str, str]] = []
    lyrics_paths = [Path(lyrics_csv_path), *(Path(path) for path in (extra_lyrics_csv_paths or []))]
    for path in lyrics_paths:
        with path.open("r", encoding="utf-8") as handle:
            base_rows.extend(list(csv.DictReader(handle)))
    log(f"loaded base lyrics rows: {len(base_rows)} in {time.perf_counter() - phase_started:0.2f}s")

    source_rows = list(base_rows)
    if cooper_first_lines_csv and Path(cooper_first_lines_csv).exists():
        phase_started = time.perf_counter()
        cooper_rows, cooper_review = build_cooper_lyrics_rows(cooper_first_lines_csv, base_rows, cooper_cache_dir)
        source_rows.extend(cooper_rows)
        review_entries.extend(cooper_review)
        log(
            "built cooper rows: "
            f"{len(cooper_rows)} rows, {len(cooper_review)} review items in {time.perf_counter() - phase_started:0.2f}s"
        )

    log(f"processing source rows: {len(source_rows)} total")

    processed_rows = 0
    kept_blocks = 0
    for row in source_rows:
        processed_rows += 1
        if is_non_song_row(row):
            continue
        raw_lyrics = clean_lyrics_text(row).strip()
        if not raw_lyrics:
            continue
        blocks, confidence, reason = propose_blocks(raw_lyrics)
        if confidence != "high":
            review_entries.append(build_review_entry(row, blocks, confidence, reason))
        associated_tune = _associated_tune_from_row(row)
        for block in blocks:
            block, editorial_tail = split_editorial_tail(block)
            if editorial_tail:
                review_entries.append(review_entry_for_block(row, editorial_tail, "suspected-editorial-paratext"))
            stanza_lines = [
                line
                for line in block
                if line and not PAREN_CHORUS_RE.match(line) and line.lower() not in CHORUS_MARKERS
            ]
            if not stanza_lines:
                continue
            disposition, junk_reason = classify_candidate_block(row, stanza_lines)
            if disposition == "exclude":
                continue
            if disposition == "review":
                review_entries.append(review_entry_for_block(row, stanza_lines, junk_reason or "suspected-junk"))
                continue
            kept_blocks += 1
            first_line = stanza_lines[0]
            text = "\n".join(stanza_lines)
            text, first_line = apply_text_equivalence_override(text, first_line, text_equivalence_overrides)
            stanza_id = find_matching_text_id(
                aggregated,
                text,
                first_line,
                match_index=aggregated_match_index,
                match_meta=aggregated_match_meta,
                stats=match_stats,
            ) or stanza_id_for_text(text, first_line)
            existing = aggregated.get(stanza_id)
            if existing is None:
                aggregated[stanza_id] = build_text_record(
                    text_id=stanza_id,
                    first_line=first_line,
                    text=text,
                    associated_tunes=[associated_tune],
                )
                index_stanza_match_metadata(
                    stanza_id,
                    text,
                    match_index=aggregated_match_index,
                    match_meta=aggregated_match_meta,
                )
            else:
                existing_tunes = list(existing.get("associated_tunes", []))
                deduped = {
                    (item.get("book", ""), item.get("page", ""), item.get("tune", ""), item.get("url", "")): item
                    for item in existing_tunes
                }
                deduped[(associated_tune["book"], associated_tune["page"], associated_tune["tune"], associated_tune["url"])] = associated_tune
                merged_tunes = sorted(
                    deduped.values(),
                    key=lambda item: (item.get("book", ""), item.get("page", ""), item.get("tune", "")),
                )
                existing["associated_tunes"] = merged_tunes
                existing["associated_songs"] = _associated_song_keys(merged_tunes)
                existing["source_books"] = _representative_books(existing["source_books"] + [row["book_id"]])
        if progress and processed_rows % max(progress_every, 1) == 0:
            lookups = match_stats.get("candidate_lookups", 0)
            avg_candidates = (match_stats.get("candidate_total", 0) / lookups) if lookups else 0.0
            log(
                f"processed {processed_rows}/{len(source_rows)} rows; "
                f"canonical_texts={len(aggregated)} kept_blocks={kept_blocks} review_items={len(review_entries)} "
                f"avg_candidates={avg_candidates:0.2f} max_candidates={match_stats.get('candidate_max', 0)} "
                f"fuzzy_matches={match_stats.get('fuzzy_matches', 0)} exact_id_hits={match_stats.get('exact_id_hits', 0)}"
            )
    stanza_rows = sorted(
        aggregated.values(),
        key=lambda item: (str(item["first_line"]).lower(), str(item["id"])),
    )
    phase_started = time.perf_counter()
    save_data_file(stanzas_path, stanza_rows)
    stanzas_save_elapsed = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    save_data_file(review_path, review_entries)
    review_save_elapsed = time.perf_counter() - phase_started
    log(
        "finished: "
        f"canonical_texts={len(stanza_rows)} review_items={len(review_entries)} "
        f"save_stanzas={stanzas_save_elapsed:0.2f}s save_review={review_save_elapsed:0.2f}s "
        f"avg_candidates={((match_stats.get('candidate_total', 0) / match_stats.get('candidate_lookups', 1)) if match_stats.get('candidate_lookups', 0) else 0.0):0.2f} "
        f"max_candidates={match_stats.get('candidate_max', 0)}"
    )
    return stanza_rows, review_entries
