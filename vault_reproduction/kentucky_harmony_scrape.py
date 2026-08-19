#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

try:
    import fitz  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyMuPDF is required for kentucky_harmony_scrape.py. Install with: python3 -m pip install PyMuPDF") from exc


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sh_corpus.bremen.scrape import canonicalize_cache_rows
from sh_corpus.shape_note_normalization import norm_first_line


BOOK_ID = "kentucky"
INDEX_URL = "https://www.shapenote.net/berkley/SKyH3.htm"
BASE_URL = "https://www.shapenote.net/berkley/"
USER_AGENT = "sh-corpus-kentucky/1.0 (+personal research)"

FIRST_LINE_FIELDS = [
    "book_id",
    "song_no",
    "title",
    "first_line",
    "first_line_raw",
    "first_line_norm",
    "text_key",
    "url",
]

LYRICS_FIELDS = [
    "book_id",
    "song_no",
    "title",
    "url",
    "first_lyric_line",
    "lyrics",
    "source_site",
    "lyrics_source",
]

FILE_NAME_RE = re.compile(r"^(?P<num>\d+)(?P<suffix>[a-z]?)\s+(?P<stem>.+)$", re.IGNORECASE)
STANZA_START_RE = re.compile(r"(?<!\d)(?P<num>\d{1,2})\.\s+")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*")


@dataclass
class IndexEntry:
    song_no: str
    title: str
    href: str
    url: str
    search_aliases: list[str]


def fetch_bytes(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=60) as response:
        return response.read()


def fetch_html(url: str) -> str:
    return fetch_bytes(url).decode("utf-8", "ignore")


def clean_space(value: str) -> str:
    value = value.replace("\xa0", " ")
    value = value.replace("\u2019", "'")
    value = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def normalize_search_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "").upper()
    value = value.replace("&", " AND ")
    value = value.replace("’", "'")
    value = value.replace("'", "")
    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(r"\[[^]]*\]", " ", value)
    value = re.sub(r"[^A-Z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def clean_title_label(value: str) -> str:
    value = clean_space(value)
    value = value.lstrip("*").strip()
    value = re.sub(r"\s*\[\s*\d+\s*\]\s*$", "", value)
    value = re.sub(r"\s*\([^)]*\)", "", value)
    value = value.replace("*", "")
    # Many entries are hand-spaced like "C amden"; collapse that form only.
    value = re.sub(r"\b([A-Z])\s+(?=[a-z])", r"\1", value)
    return clean_space(value)


def parse_stem_parts(href: str) -> tuple[int, str, list[str]]:
    stem = Path(href).stem
    match = FILE_NAME_RE.match(stem)
    if not match:
        raise ValueError(f"Could not parse song number from {href}")
    number = int(match.group("num"))
    suffix = match.group("suffix").lower()
    parts = [clean_space(part) for part in match.group("stem").split("-")]
    return number, suffix, parts


def build_song_no(number: int, file_suffix: str, occurrence: int) -> str:
    if file_suffix:
        return f"{number}{file_suffix}"
    if occurrence == 1:
        return str(number)
    suffix_rank = occurrence - 1
    return f"{number}{chr(ord('a') + suffix_rank)}"


def parse_index_entries(html: str) -> list[IndexEntry]:
    soup = BeautifulSoup(html, "html.parser")
    entries: list[IndexEntry] = []
    occurrence_by_href: Counter[str] = Counter()

    for li in soup.select("li"):
        a = li.find("a", href=True)
        if not a:
            continue
        href = a["href"]
        if not href.lower().endswith(".pdf"):
            continue
        raw_label = " ".join(li.get_text(" ", strip=True).split())
        title = clean_title_label(raw_label)
        number, file_suffix, stem_parts = parse_stem_parts(href)
        occurrence_by_href[href] += 1
        occurrence = occurrence_by_href[href]
        song_no = build_song_no(number, file_suffix, occurrence)

        aliases = [title]
        aliases.extend(re.findall(r'"([^"]+)"', raw_label))
        if not file_suffix and len(stem_parts) >= occurrence:
            aliases.append(stem_parts[occurrence - 1])
        elif file_suffix and stem_parts:
            aliases.append(stem_parts[-1])
        search_aliases = [normalize_search_text(alias) for alias in aliases if normalize_search_text(alias)]

        entries.append(
            IndexEntry(
                song_no=song_no,
                title=title,
                href=href,
                url=urljoin(BASE_URL, quote(href)),
                search_aliases=list(dict.fromkeys(search_aliases)),
            )
        )

    return entries


def extract_pdf_text(pdf_bytes: bytes) -> tuple[str, list[list[tuple[float, float, float, float, str]]]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_texts: list[str] = []
    page_words: list[list[tuple[float, float, float, float, str]]] = []
    for page in doc:
        page_texts.append(page.get_text("text"))
        words = [(x0, y0, x1, y1, text) for x0, y0, x1, y1, text, *_rest in page.get_text("words")]
        page_words.append(words)
    return "\n".join(page_texts), page_words


def locate_sections(cleaned_text: str, entries: list[IndexEntry]) -> list[tuple[IndexEntry, str]]:
    search_text = cleaned_text.upper()

    def find_alias_position(alias: str, start: int) -> int:
        words = [word for word in alias.split() if word]
        if not words:
            return -1
        pattern = r"\b" + r"\W+".join(re.escape(word) for word in words) + r"\b"
        match = re.search(pattern, search_text[start:])
        if match:
            return start + match.start()
        match = re.search(pattern, search_text)
        return match.start() if match else -1

    positions: list[tuple[int, int]] = []
    cursor = 0
    for entry in entries:
        found = -1
        for alias in entry.search_aliases:
            idx = find_alias_position(alias, cursor)
            if idx >= 0 and (found < 0 or idx < found):
                found = idx
        positions.append((max(found, cursor), found))
        cursor = max(found, cursor)

    if len(entries) == 1:
        return [(entries[0], cleaned_text)]

    sections: list[tuple[IndexEntry, str]] = []
    for i, entry in enumerate(entries):
        start_guess, start_found = positions[i]
        next_start = len(cleaned_text)
        for later_guess, later_found in positions[i + 1 :]:
            if later_found >= 0 and later_guess > start_guess:
                next_start = later_guess
                break
        if start_found < 0:
            section_text = cleaned_text
        else:
            section_text = cleaned_text[start_guess:next_start]
        sections.append((entry, section_text))
    return sections


def stitch_hyphenated_syllables(value: str) -> str:
    out = value
    prev = None
    while out != prev:
        prev = out
        out = re.sub(r"([A-Za-z])\s*-\s*([A-Za-z])", r"\1-\2", out)
    return out


def clean_lyric_text(value: str) -> str:
    value = clean_space(value)
    value = stitch_hyphenated_syllables(value)
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    value = re.sub(r"\(\s+", "(", value)
    value = re.sub(r"\s+\)", ")", value)
    return clean_space(value)


def is_viable_lyric_line(value: str) -> bool:
    letters = WORD_RE.findall(value)
    junk_chars = sum(1 for ch in value if not (ch.isalnum() or ch.isspace() or ch in ".,;:!?'-()&"))
    return len(letters) >= 4 and junk_chars <= max(6, len(value) // 12)


def extract_stanzas_from_text(section_text: str) -> list[str]:
    text = clean_lyric_text(section_text)
    matches = list(STANZA_START_RE.finditer(text))
    if not matches:
        return []

    stanzas: list[str] = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        stanza = clean_lyric_text(text[start:end])
        stanza = re.sub(r"^[^A-Za-z]+", "", stanza)
        stanza = re.sub(r"\s+[A-Z][A-Z .'-]{2,}\.\s+[A-Z].*$", "", stanza)
        if is_viable_lyric_line(stanza):
            stanzas.append(stanza)

    deduped: list[str] = []
    for stanza in stanzas:
        if not deduped or deduped[-1] != stanza:
            deduped.append(stanza)
    return deduped


def group_word_lines(page_words: list[tuple[float, float, float, float, str]], tolerance: float = 3.5) -> list[str]:
    if not page_words:
        return []
    words = sorted(page_words, key=lambda word: (word[1], word[0]))
    groups: list[list[tuple[float, float, float, float, str]]] = []
    current: list[tuple[float, float, float, float, str]] = []
    current_y = None

    for word in words:
        y = word[1]
        if current and current_y is not None and abs(y - current_y) > tolerance:
            groups.append(current)
            current = [word]
            current_y = y
        else:
            if not current:
                current_y = y
            else:
                current_y = (current_y + y) / 2 if current_y is not None else y
            current.append(word)
    if current:
        groups.append(current)

    lines: list[str] = []
    for group in groups:
        parts = [clean_space(word[4]) for word in sorted(group, key=lambda item: item[0])]
        line = clean_lyric_text(" ".join(part for part in parts if part))
        if line:
            lines.append(line)
    return lines


def extract_stanzas_from_words(page_words_by_page: list[list[tuple[float, float, float, float, str]]]) -> list[str]:
    lines: list[str] = []
    for page_words in page_words_by_page:
        lines.extend(group_word_lines(page_words))

    stanza_lines = [line for line in lines if re.match(r"^\d+\.\s+", line)]
    stanzas: list[str] = []
    for line in stanza_lines:
        stanza = clean_lyric_text(re.sub(r"^\d+\.\s*", "", line))
        if is_viable_lyric_line(stanza):
            stanzas.append(stanza)
    return stanzas


def choose_first_line(stanza: str) -> str:
    stanza = clean_lyric_text(re.sub(r"^\d+\.\s*", "", stanza))
    candidates = [match.end() for match in re.finditer(r"[,;:!?]", stanza)]
    for end in candidates:
        prefix = stanza[:end].strip()
        if len(WORD_RE.findall(prefix)) >= 4:
            return prefix
    words = stanza.split()
    return " ".join(words[: min(len(words), 10)]).strip()


def load_unique_title_fallbacks(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_title: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_title[normalize_search_text(row.get("title", ""))].append(row)

    out: dict[str, dict[str, str]] = {}
    for title_key, matches in by_title.items():
        text_key_counts: Counter[str] = Counter(row.get("text_key", "").strip() for row in matches if row.get("text_key", "").strip())
        if not text_key_counts:
            continue
        ranked = text_key_counts.most_common()
        if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
            preferred_key = ranked[0][0]
            preferred = sorted(
                [row for row in matches if row.get("text_key", "").strip() == preferred_key],
                key=lambda row: (row.get("book_id", ""), row.get("song_no", "")),
            )[0]
            out[title_key] = preferred
    return out


def lookup_title_fallback(entry: IndexEntry, title_fallbacks: dict[str, dict[str, str]]) -> dict[str, str] | None:
    for alias in entry.search_aliases:
        fallback = title_fallbacks.get(alias)
        if fallback:
            return fallback
    return None


def has_strong_first_line(first_line: str) -> bool:
    first_line = clean_lyric_text(first_line)
    if not first_line:
        return False
    words = first_line.split()
    alpha_words = [word for word in words if re.search(r"[A-Za-z]", word)]
    long_words = [word for word in alpha_words if len(re.sub(r"[^A-Za-z]", "", word)) >= 3]
    short_words = [word for word in alpha_words if len(re.sub(r"[^A-Za-z]", "", word)) <= 2]
    weird_chars = sum(1 for ch in first_line if not (ch.isalnum() or ch.isspace() or ch in ".,;:!?'-()"))
    upper = first_line.upper()

    if len(alpha_words) < 4:
        return False
    if len(long_words) < 3:
        return False
    if any(ch.isdigit() for ch in first_line):
        return False
    if " MAJOR." in f" {upper}" or " MINOR." in f" {upper}":
        return False
    if weird_chars > 2:
        return False
    if len(short_words) > max(4, len(alpha_words) // 2):
        return False
    if alpha_words and sum(ch.isalpha() for ch in first_line) / max(len(first_line), 1) < 0.6:
        return False
    return True


def load_lyrics_fallbacks(first_lines_path: Path, lyrics_paths: Iterable[Path]) -> dict[str, dict[str, str]]:
    text_key_to_first_line: dict[str, str] = {}
    with first_lines_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            text_key = row.get("text_key", "").strip()
            first_line = row.get("first_line_raw", "").strip()
            if text_key and first_line and text_key not in text_key_to_first_line:
                text_key_to_first_line[text_key] = first_line

    fallback_rows: dict[str, dict[str, str]] = {}
    for path in lyrics_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                first_line = row.get("first_lyric_line", "").strip()
                lyrics = row.get("lyrics", "").strip()
                if not first_line or not lyrics:
                    continue
                text_key = norm_first_line(first_line)
                if text_key not in text_key_to_first_line:
                    continue
                fallback_rows.setdefault(
                    text_key,
                    {
                        "first_lyric_line": first_line,
                        "lyrics": lyrics,
                        "source_site": row.get("source_site", ""),
                        "lyrics_source": row.get("lyrics_source", ""),
                    },
                )
    return fallback_rows


def build_rows(
    entries: list[IndexEntry],
    title_fallbacks: dict[str, dict[str, str]],
    lyrics_fallbacks: dict[str, dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]], Counter[str]]:
    first_line_rows_raw: list[dict[str, str]] = []
    lyric_rows: list[dict[str, str]] = []
    counts: Counter[str] = Counter()

    entries_by_href: dict[str, list[IndexEntry]] = defaultdict(list)
    for entry in entries:
        entries_by_href[entry.href].append(entry)

    for href, href_entries in entries_by_href.items():
        pdf_bytes = fetch_bytes(href_entries[0].url)
        raw_text, page_words_by_page = extract_pdf_text(pdf_bytes)
        cleaned_text = clean_space(raw_text)
        sections = locate_sections(cleaned_text, href_entries)

        for entry, section_text in sections:
            stanzas = extract_stanzas_from_text(section_text)
            source = "shapenote_pdf_text"
            first_line = choose_first_line(stanzas[0]) if stanzas else ""
            fallback = lookup_title_fallback(entry, title_fallbacks)

            if (not stanzas or not has_strong_first_line(first_line)) and len(href_entries) == 1:
                stanzas = extract_stanzas_from_words(page_words_by_page)
                if stanzas:
                    source = "shapenote_pdf_words"
                    first_line = choose_first_line(stanzas[0])

            if fallback:
                first_line = fallback.get("first_line_raw", "") or fallback.get("first_line", "")
                counts["first_line_title_fallback"] += 1
            elif first_line:
                counts[source] += 1

            if not first_line:
                counts["first_line_missing"] += 1
                continue

            first_line_rows_raw.append(
                {
                    "book_id": BOOK_ID,
                    "song_no": entry.song_no,
                    "title": entry.title,
                    "first_line": first_line,
                    "first_line_raw": first_line,
                    "url": entry.url,
                }
            )

            lyrics = "\n\n".join(stanzas).strip()
            lyrics_source = source
            source_site = "shapenote.net"
            first_lyric_line = first_line

            if fallback:
                text_key = fallback.get("text_key", "").strip()
                fallback_lyrics = lyrics_fallbacks.get(text_key)
                if fallback_lyrics:
                    first_lyric_line = fallback_lyrics["first_lyric_line"]
                    lyrics = fallback_lyrics["lyrics"]
                    source_site = fallback_lyrics["source_site"] or "fallback_existing_corpus"
                    lyrics_source = f"title_match_existing_witness:{fallback_lyrics['lyrics_source'] or 'unknown'}"
                    counts["lyrics_title_fallback"] += 1
                elif not lyrics:
                    counts["lyrics_missing"] += 1
            elif lyrics:
                counts[f"lyrics_{source}"] += 1
            else:
                counts["lyrics_missing"] += 1

            lyric_rows.append(
                {
                    "book_id": BOOK_ID,
                    "song_no": entry.song_no,
                    "title": entry.title,
                    "url": entry.url,
                    "first_lyric_line": first_lyric_line,
                    "lyrics": lyrics,
                    "source_site": source_site,
                    "lyrics_source": lyrics_source,
                }
            )

    first_line_rows, _alias_map = canonicalize_cache_rows(first_line_rows_raw)
    first_line_rows.sort(key=lambda row: song_no_sort_key(row["song_no"]))
    lyric_rows.sort(key=lambda row: song_no_sort_key(row["song_no"]))
    return first_line_rows, lyric_rows, counts


def song_no_sort_key(song_no: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"(?P<num>\d+)(?P<suffix>[a-z]?)", (song_no or "").strip().lower())
    if not match:
        return (10**9, 99, song_no or "")
    suffix = match.group("suffix")
    if suffix == "":
        suffix_rank = 0
    elif suffix == "t":
        suffix_rank = 1
    else:
        suffix_rank = 2 + (ord(suffix) - ord("a"))
    return (int(match.group("num")), suffix_rank, song_no or "")


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(prog="kentucky_harmony_scrape.py")
    parser.add_argument("--first-lines-out", default="kentucky_first_lines.csv")
    parser.add_argument("--lyrics-out", default="kentucky_lyrics.csv")
    parser.add_argument("--combined-first-lines", default="combined_first_lines_canonicalized.csv")
    parser.add_argument(
        "--lyrics-fallback-csv",
        action="append",
        default=["bremen_lyrics.csv", "modern_shape_note_lyrics.csv", "southernharmony_lyrics.csv"],
        help="Existing lyrics CSVs used only for unique title-match fallback.",
    )
    args = parser.parse_args()

    html = fetch_html(INDEX_URL)
    entries = parse_index_entries(html)

    combined_path = Path(args.combined_first_lines).expanduser().resolve()
    title_fallbacks = load_unique_title_fallbacks(combined_path) if combined_path.exists() else {}
    lyrics_fallback_paths = [Path(path).expanduser().resolve() for path in args.lyrics_fallback_csv]
    lyrics_fallbacks = load_lyrics_fallbacks(combined_path, lyrics_fallback_paths) if combined_path.exists() else {}

    first_line_rows, lyric_rows, counts = build_rows(entries, title_fallbacks, lyrics_fallbacks)

    first_lines_out = Path(args.first_lines_out).expanduser().resolve()
    lyrics_out = Path(args.lyrics_out).expanduser().resolve()
    write_rows(first_lines_out, FIRST_LINE_FIELDS, first_line_rows)
    write_rows(lyrics_out, LYRICS_FIELDS, lyric_rows)

    print(f"[ok] source: {INDEX_URL}")
    print(f"[ok] first-lines: {first_lines_out} ({len(first_line_rows)} rows)")
    print(f"[ok] lyrics: {lyrics_out} ({len(lyric_rows)} rows)")
    for key in sorted(counts):
        print(f"[ok] {key}: {counts[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
