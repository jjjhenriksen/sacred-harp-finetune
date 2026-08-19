from __future__ import annotations

import csv
import re
import sys
import time
from collections import defaultdict
from urllib.parse import urljoin

try:
    import cloudscraper
except Exception:  # pragma: no cover
    cloudscraper = None

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
    BeautifulSoup = None

from .common import (
    CACHE_CSV,
    ENTRY_RE_DASH,
    ENTRY_RE_DASH_TAIL,
    ENTRY_RE_PAREN,
    SONGS_INDEX_URLS,
    TUNEBOOKS,
    norm_first_line,
    resolve_first_line_prefix_alias,
)


SONG_INDEX_ENTRY_RE = re.compile(r"^(?P<num>\d+[a-z]?)\s+(?P<title>.+?)$", re.IGNORECASE)
PAGE_SONG_INDEX_LINE_RE = re.compile(r"^\d+[a-z]?\s+\S", re.IGNORECASE)


def fetch_soup(url: str, session) -> BeautifulSoup:
    if BeautifulSoup is None:
        raise RuntimeError("BeautifulSoup4 is required for Bremen scraping.")
    r = session.get(url, timeout=30)
    r.raise_for_status()
    text_lower = r.text.lower()
    if "cloudflare" in text_lower and ("attention required" in text_lower or "cf-browser-verification" in text_lower):
        raise RuntimeError(f"Blocked by Cloudflare at {url} (challenge page).")
    return BeautifulSoup(r.text, "html.parser")


def parse_bremen_first_lines(index_url: str, book_id: str, session):
    soup = fetch_soup(index_url, session)
    rows = []
    for a in soup.find_all("a", href=True):
        text = a.get_text(" ", strip=True)
        if not text:
            continue
        m = ENTRY_RE_PAREN.match(text) or ENTRY_RE_DASH.match(text)
        if not m:
            continue
        first_line = m.group("first").strip()
        rows.append(
            {
                "book_id": book_id,
                "song_no": m.group("num").strip(),
                "title": m.group("title").strip(),
                "first_line": first_line,
                "first_line_raw": first_line,
                "first_line_norm": norm_first_line(resolve_first_line_prefix_alias(first_line)),
                "url": urljoin(index_url, a["href"]),
            }
        )

    for a in soup.find_all("a", href=True):
        tail_text = a.get_text(" ", strip=True)
        m = ENTRY_RE_DASH_TAIL.match(tail_text)
        if not m:
            continue
        parent_lines = [ln.strip() for ln in a.parent.get_text("\n", strip=True).splitlines() if ln.strip()]
        first_line = None
        for i, ln in enumerate(parent_lines):
            if ln == tail_text and i > 0:
                first_line = parent_lines[i - 1]
                break
        if not first_line or first_line.lower().startswith("the sacred harp"):
            continue
        if re.fullmatch(r"\d+\s*(?:—|–|-)\s*\d+", first_line):
            continue
        if re.search(r"\b\d{1,4}[a-z]?\b", first_line):
            continue
        rows.append(
            {
                "book_id": book_id,
                "song_no": m.group("num").strip(),
                "title": m.group("title").strip(),
                "first_line": first_line,
                "first_line_raw": first_line,
                "first_line_norm": norm_first_line(resolve_first_line_prefix_alias(first_line)),
                "url": urljoin(index_url, a["href"]),
            }
        )

    by_song = defaultdict(list)
    for r in rows:
        by_song[(r["song_no"], r["title"])].append(r)

    cleaned_rows = []
    for group in by_song.values():
        slug_rows = [r for r in group if "-" in r["url"].rstrip("/").split("/")[-1]]
        cleaned_rows.extend(slug_rows if slug_rows else group)

    dedup = {}
    for r in cleaned_rows:
        dedup[(r["book_id"], r["song_no"], r["title"], r["first_line"])] = r
    return list(dedup.values())


def parse_bremen_songs_index(index_url: str, book_id: str, session):
    soup = fetch_soup(index_url, session)
    dedup: dict[tuple[str, str, str], dict] = {}

    for a in soup.find_all("a", href=True):
        text = a.get_text(" ", strip=True)
        if not text:
            continue
        m = SONG_INDEX_ENTRY_RE.match(text)
        if not m:
            continue

        song_no = m.group("num").strip().lower()
        title = m.group("title").strip()
        if re.fullmatch(r"[—–-]\s*\d+", title):
            continue
        href = a["href"]

        row = {
            "book_id": book_id,
            "song_no": song_no,
            "title": title,
            "first_line": "",
            "first_line_raw": "",
            "first_line_norm": "",
            "url": urljoin(index_url, href),
        }
        dedup[(row["book_id"], row["song_no"], row["title"])] = row

    return list(dedup.values())


def extract_trailing_lyrics_block_from_raw_lines(lines: list[str]) -> list[str]:
    if not lines:
        return []

    last_song_index = None
    for i, line in enumerate(lines):
        if PAGE_SONG_INDEX_LINE_RE.match((line or "").strip()):
            last_song_index = i
    if last_song_index is None:
        return []

    out: list[str] = []
    for raw_line in lines[last_song_index + 1 :]:
        line = (raw_line or "").strip()
        if not line:
            continue
        low = line.lower()
        if low == "recordings":
            break
        if low in {"songs", "songs (general index)", "index of first lines", "home", "our websites", "follow us", "protected email", "impressum"}:
            continue
        if PAGE_SONG_INDEX_LINE_RE.match(line):
            continue
        if re.fullmatch(r"[0-9,\s]+", line):
            continue
        if not re.search(r"[A-Za-z]", line):
            continue
        out.append(line)
    return out


def infer_first_line_from_song_page(song_url: str, session) -> str:
    soup = fetch_soup(song_url, session)
    root = soup.select_one("article .entry-content") or soup.select_one("article") or soup.select_one("main") or soup.select_one("div.entry-content") or soup.select_one("div#content") or soup
    for tag in root.find_all(["script", "style", "noscript"]):
        tag.decompose()
    raw_lines = [ln.strip() for ln in root.get_text("\n", strip=True).splitlines() if ln.strip()]
    trailing = extract_trailing_lyrics_block_from_raw_lines(raw_lines)
    return trailing[0] if trailing else ""


def build_prefix_alias_map(norm_lines: list[str], max_short_len: int = 30) -> dict[str, str]:
    unique = sorted({(s or "").strip() for s in norm_lines if (s or "").strip()}, key=lambda s: (len(s), s))
    alias_map: dict[str, str] = {}
    for short in unique:
        if len(short) > max_short_len:
            continue
        candidates = [cand for cand in unique if len(cand) > len(short) and cand.startswith(short + " ")]
        if len(candidates) == 1:
            alias_map[short] = candidates[0]
    return alias_map


def resolve_alias(alias_map: dict[str, str], key: str) -> str:
    out = (key or "").strip()
    seen: set[str] = set()
    while out in alias_map and out not in seen:
        seen.add(out)
        out = alias_map[out]
    return out


def canonicalize_cache_rows(rows: list[dict], max_short_len: int = 30) -> tuple[list[dict], dict[str, str]]:
    prepared = []
    norm_lines = []
    for r in rows:
        row = dict(r)
        first_line_raw = (row.get("first_line_raw") or row.get("first_line") or "").strip()
        first_line_norm = norm_first_line(resolve_first_line_prefix_alias(first_line_raw))
        row["first_line_raw"] = first_line_raw
        row["first_line_norm"] = first_line_norm
        row["first_line"] = first_line_raw
        prepared.append(row)
        norm_lines.append(first_line_norm)
    alias_map = build_prefix_alias_map(norm_lines, max_short_len=max_short_len)
    for row in prepared:
        row["text_key"] = resolve_alias(alias_map, row.get("first_line_norm", ""))
    return prepared, alias_map


def build_cache(path: str, sleep_s: float = 0.2, max_short_len: int = 30):
    session = cloudscraper.create_scraper()
    session.headers.update({"User-Agent": "bremen-crosswalk-cli/1.0 (+personal research)"})
    all_rows = []
    for url, book_id in TUNEBOOKS:
        rows = parse_bremen_first_lines(url, book_id, session)
        supplement_url = SONGS_INDEX_URLS.get(book_id)
        if supplement_url:
            song_rows = parse_bremen_songs_index(supplement_url, book_id, session)
            existing_song_nos = {(r.get("song_no") or "").strip().lower() for r in rows}
            supplemental_rows = [r for r in song_rows if (r.get("song_no") or "").strip().lower() not in existing_song_nos]
            if supplemental_rows:
                rows.extend(supplemental_rows)
                print(f"[ok] {book_id}: supplemented {len(supplemental_rows)} rows from songs index", file=sys.stderr)
        inferred_first_lines = 0
        for row in rows:
            if (row.get("first_line_raw") or "").strip():
                continue
            try:
                first_line = infer_first_line_from_song_page(row.get("url", ""), session)
            except Exception:
                first_line = ""
            if not first_line:
                continue
            row["first_line"] = first_line
            row["first_line_raw"] = first_line
            row["first_line_norm"] = norm_first_line(resolve_first_line_prefix_alias(first_line))
            inferred_first_lines += 1
        if inferred_first_lines:
            print(f"[ok] {book_id}: inferred {inferred_first_lines} first lines from song pages", file=sys.stderr)
        all_rows.extend(rows)
        print(f"[ok] {book_id}: {len(rows)} rows", file=sys.stderr)
        time.sleep(sleep_s)
    all_rows, alias_map = canonicalize_cache_rows(all_rows, max_short_len=max_short_len)
    fields = ["book_id", "song_no", "title", "first_line", "first_line_raw", "first_line_norm", "text_key", "url"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    if alias_map:
        print(f"[ok] prefix aliases: {len(alias_map)} (max_short_len={max_short_len})", file=sys.stderr)
    print(f"Wrote cache: {path} ({len(all_rows)} rows)", file=sys.stderr)


def load_cache(path: str = CACHE_CSV):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))
