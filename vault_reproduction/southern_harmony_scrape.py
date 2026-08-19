#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sh_corpus.bremen.scrape import canonicalize_cache_rows


BOOK_ID = "southernharmony"
SOURCE_URL = "https://ccel.org/ccel/walker/harmony/cache/harmony.html3"
USER_AGENT = "sh-corpus-southernharmony/1.0 (+personal research)"

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

HYMN_ID_RE = re.compile(r"^(H(?P<song_no>\d+[a-z]?))-p0\.1$", re.IGNORECASE)
METADATA_CLASSES = {"meter", "author", "authorRel", "tune", "music", "scripRef"}
SECTION_MARKERS = {"refrain:", "chorus:"}


def fetch_html(url: str) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=30) as response:
        return response.read().decode("utf-8", "ignore")


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


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


def build_entry_url(anchor: str) -> str:
    return f"https://ccel.org/ccel/walker/harmony/harmony.{anchor}.html"


def extract_stanzas(hymn: BeautifulSoup) -> list[str]:
    stanzas: list[str] = []
    current_lines: list[str] = []

    for p in hymn.find_all("p"):
        classes = set(p.get("class") or [])
        text = clean_text(p.get_text(" ", strip=True))
        if not text:
            continue
        if classes & METADATA_CLASSES:
            continue
        if "verseFloatLeft" in classes:
            if current_lines:
                stanzas.append("\n".join(current_lines))
                current_lines = []
            continue
        if text.lower() in SECTION_MARKERS:
            if current_lines:
                stanzas.append("\n".join(current_lines))
                current_lines = []
            continue
        if "l" in classes:
            current_lines.append(text)

    if current_lines:
        stanzas.append("\n".join(current_lines))
    return stanzas


def parse_rows(html: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    soup = BeautifulSoup(html, "html.parser")
    first_line_rows_raw: list[dict[str, str]] = []
    lyric_rows: list[dict[str, str]] = []

    for hymn in soup.select("div.hymn[id]"):
        raw_id = hymn.get("id", "")
        match = HYMN_ID_RE.fullmatch(raw_id)
        if not match:
            continue
        anchor = match.group(1)
        song_no = match.group("song_no").lower()
        title_node = hymn.find("h2")
        title = clean_text(title_node.get_text(" ", strip=True) if title_node else "")
        stanzas = extract_stanzas(hymn)
        if not title or not stanzas:
            continue
        first_line = clean_text(stanzas[0].splitlines()[0])
        url = build_entry_url(anchor)

        first_line_rows_raw.append(
            {
                "book_id": BOOK_ID,
                "song_no": song_no,
                "title": title,
                "first_line": first_line,
                "first_line_raw": first_line,
                "url": url,
            }
        )
        lyric_rows.append(
            {
                "book_id": BOOK_ID,
                "song_no": song_no,
                "title": title,
                "url": url,
                "first_lyric_line": first_line,
                "lyrics": "\n\n".join(stanzas),
                "source_site": "ccel.org",
                "lyrics_source": "ccel_harmony_html3",
            }
        )

    first_line_rows, _alias_map = canonicalize_cache_rows(first_line_rows_raw)
    first_line_rows.sort(key=lambda row: song_no_sort_key(row["song_no"]))
    lyric_rows.sort(key=lambda row: song_no_sort_key(row["song_no"]))
    return first_line_rows, lyric_rows


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(prog="southern_harmony_scrape.py")
    parser.add_argument("--first-lines-out", default="southernharmony_first_lines.csv")
    parser.add_argument("--lyrics-out", default="southernharmony_lyrics.csv")
    args = parser.parse_args()

    html = fetch_html(SOURCE_URL)
    first_line_rows, lyric_rows = parse_rows(html)

    first_lines_out = Path(args.first_lines_out).expanduser().resolve()
    lyrics_out = Path(args.lyrics_out).expanduser().resolve()
    write_rows(first_lines_out, FIRST_LINE_FIELDS, first_line_rows)
    write_rows(lyrics_out, LYRICS_FIELDS, lyric_rows)

    print(f"[ok] source: {SOURCE_URL}")
    print(f"[ok] first-lines: {first_lines_out} ({len(first_line_rows)} rows)")
    print(f"[ok] lyrics: {lyrics_out} ({len(lyric_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
