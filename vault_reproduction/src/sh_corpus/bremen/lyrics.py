from __future__ import annotations

import csv
import re
import sys
import time
import unicodedata
from urllib.parse import urljoin

try:
    import cloudscraper
except Exception:  # pragma: no cover
    cloudscraper = None

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
    BeautifulSoup = None

from .common import CACHE_CSV, LYRICS_CSV, OVERRIDES_CSV, norm_first_line
from .scrape import fetch_soup, load_cache, extract_trailing_lyrics_block_from_raw_lines

JUNK_LINE_PATTERNS = [
    r"^\s*home\s*$", r"^\s*our websites\s*$", r"^\s*impressum\s*$", r"^\s*follow us\s*$", r"^\s*protected email\s*$",
    r"sacred harp bremen", r"shenandoah harmony", r"christian harmony", r"the sacred harp\s+1991", r"the sacred harp\s+-\s+1991 edition",
]
JUNK_MARKERS = ["home", "our websites", "shb", "main website", "shenandoah harmony", "christian harmony", "the sacred harp", "impressum", "wir treffen uns", "follow us", "protected email"]
ANCHOR_STOP_PATTERNS = [r"^\s*youtube\s*$", r"^\s*songs\s*$", r"^\s*songs\s*\(general index\)\s*$", r"^\s*index of first lines\s*$", r"^\s*our websites\s*$", r"^\s*home\s*$", r"^\s*impressum\s*$", r"^\s*follow us\s*$", r"^\s*protected email\s*$", r"^\s*\d+\s*[–—-]\s*\d+\s*$"]
LEADING_NUMBERED_NAV_RE = re.compile(r"^\d+[a-z]?\s")


def looks_like_nav_or_footer(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    low = s.lower()
    if any(re.search(pat, low) for pat in JUNK_LINE_PATTERNS):
        return True
    return s.startswith(("*", "•")) or re.fullmatch(r"\d+\s*[–—-]\s*\d+", s) is not None


def junk_score(lines: list[str]) -> int:
    if not lines:
        return 999
    score = sum(1 for ln in lines[:30] if looks_like_nav_or_footer(ln)) * 5
    joined = "\n".join(lines).lower()
    if "our websites" in joined:
        score += 25
    if joined.count("home") >= 3:
        score += 15
    if sum(1 for ln in lines if re.search(r"[,’';:!?]", ln)) == 0 and len(lines) >= 6:
        score += 10
    return score


def find_txt_lyrics_url(soup: BeautifulSoup, page_url: str) -> str | None:
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".txt") and "wp-content/uploads" in href.lower():
            return urljoin(page_url, href)
    return None


def pick_best_content_root(soup: BeautifulSoup):
    for sel in ["article .entry-content", "article", "main", "div.entry-content", "div#content", "body"]:
        node = soup.select_one(sel)
        if node:
            return node
    return soup


def extract_lines_from_node(node) -> list[str]:
    for tag in node.find_all(["script", "style", "noscript"]):
        tag.decompose()
    lines = [ln.strip() for ln in node.get_text("\n", strip=True).splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        if ln.lower().startswith("youtube"):
            lines = lines[:i]
            break
    lines = strip_leading_numbered_nav_block(lines)
    out = []
    for ln in lines:
        if looks_like_nav_or_footer(ln) or not re.search(r"[A-Za-z]", ln):
            continue
        if ln[0].isdigit() and len(ln.split()) <= 4:
            continue
        out.append(ln)
    return out


def load_overrides(path: str) -> dict[tuple[str, str], dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        return {}
    out = {}
    for r in rows:
        book = (r.get("book_id") or "").strip()
        song = (r.get("song_no") or "").strip()
        good = (r.get("good_url") or "").strip()
        if book and song and good:
            out[(book, song)] = r
    return out


def load_lyrics_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_lyrics_csv(path: str, rows: list[dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["book_id", "song_no", "title", "url", "first_lyric_line", "lyrics"])
        w.writeheader()
        w.writerows(rows)


def slugify_title(title: str) -> str:
    s = unicodedata.normalize("NFKD", title or "").lower().replace("’", "").replace("'", "")
    s = re.sub(r"[^a-z0-9\s-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().replace(" ", "-")
    return re.sub(r"-+", "-", s)


def split_song_no(song_no: str) -> tuple[str, str]:
    m = re.match(r"^(\d+)([a-z]?)$", (song_no or "").strip().lower())
    return (m.group(1), m.group(2)) if m else (song_no.strip(), "")


def looks_like_junk_text(s: str) -> bool:
    low = (s or "").strip().lower()
    if not low:
        return True
    if any(m in low for m in JUNK_MARKERS):
        return True
    lines = [ln.strip() for ln in (s or "").splitlines() if ln.strip()]
    return len(lines) >= 3 and sum(1 for ln in lines[:10] if len(ln.split()) <= 4) >= 6


def lyrics_block_is_valid(lines: list[str]) -> bool:
    text = "\n".join(lines).strip()
    return bool(lines) and not looks_like_junk_text(text) and len(lines) >= 4 and sum(1 for ch in text if ch.isalpha()) >= 40


def normalize_url_basic(url: str) -> str:
    return (url or "").strip().replace("=", "-")


def rewrite_last_segment(url: str, new_last: str) -> str:
    parts = url.rstrip("/").split("/")
    if len(parts) < 4:
        return url
    parts[-1] = new_last.strip("/")
    return "/".join(parts) + "/"


def candidate_urls(book_id: str, song_no: str, title: str, seed_url: str, override_url: str | None = None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    def add(u: str):
        u = normalize_url_basic(u)
        if not u:
            return
        for uu in (u, u.rstrip("/")):
            if uu and uu not in seen:
                seen.add(uu)
                out.append(uu)

    add(override_url or "")
    add(seed_url)
    base_num, _suffix = split_song_no(song_no)
    slug = slugify_title(title)
    if seed_url:
        last = seed_url.rstrip("/").split("/")[-1].lower()
        if re.fullmatch(r"\d+[a-z]?", last):
            if slug:
                add(rewrite_last_segment(seed_url, f"{song_no}-{slug}"))
                add(rewrite_last_segment(seed_url, f"{base_num}-{slug}"))
        last = seed_url.rstrip("/").split("/")[-1]
        m = re.match(r"^(?P<n>\d+)(?P<rest>[-].+)$", last)
        if m and m.group("n") != base_num:
            add(rewrite_last_segment(seed_url, f"{base_num}{m.group('rest')}"))
    return out


def _norm_line_for_anchor(s: str) -> str:
    return norm_first_line(s or "")


def extract_raw_lines_from_node(node) -> list[str]:
    for tag in node.find_all(["script", "style", "noscript"]):
        tag.decompose()
    lines = [ln.strip() for ln in node.get_text("\n", strip=True).splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        if ln.lower().startswith("youtube"):
            return lines[:i]
    return lines


def strip_leading_numbered_nav_block(lines: list[str], min_count: int = 5) -> list[str]:
    i = 0
    while i < len(lines) and (lines[i] or "").strip() and LEADING_NUMBERED_NAV_RE.match((lines[i] or "").strip()):
        i += 1
    return lines[i:] if i >= min_count else lines


def looks_like_anchor_stop(line: str) -> bool:
    low = (line or "").strip().lower()
    return any(re.search(pat, low) for pat in ANCHOR_STOP_PATTERNS) if low else False


def strip_navigation_blocks(lines: list[str]) -> list[str]:
    if not lines:
        return []

    def is_navish(s: str) -> bool:
        ss = (s or "").strip()
        return not ss or looks_like_nav_or_footer(ss) or looks_like_anchor_stop(ss) or (ss[0].isdigit() and len(ss.split()) <= 4)

    i = 0
    while i < len(lines) and is_navish(lines[i]):
        i += 1
    out = []
    nav_run = 0
    for ln in lines[i:]:
        s = (ln or "").strip()
        if not s:
            continue
        if is_navish(s):
            nav_run += 1
            if out and nav_run >= 3:
                break
            continue
        nav_run = 0
        out.append(s)
    while out and is_navish(out[-1]):
        out.pop()
    return out


def extract_lyrics_by_first_line_anchor(page_lines: list[str], first_line: str) -> list[str]:
    target = _norm_line_for_anchor(first_line)
    if not target:
        return []
    anchor_idx = next((i for i, ln in enumerate(page_lines) if _norm_line_for_anchor(ln) == target), None)
    if anchor_idx is None:
        return []
    out = []
    bad_streak = 0
    for ln in page_lines[anchor_idx + 1 :]:
        s = (ln or "").strip()
        if not s:
            continue
        if looks_like_anchor_stop(s):
            break
        if looks_like_nav_or_footer(s) or (s[0].isdigit() and len(s.split()) <= 4):
            bad_streak += 1
            if bad_streak >= 8:
                break
            continue
        out.append(s)
        bad_streak = 0
    while out and looks_like_nav_or_footer(out[-1]):
        out.pop()
    return out


def fetch_lyrics_for_url(song_url: str, session, first_line: str | None = None) -> list[str]:
    soup = fetch_soup(song_url, session)
    txt_url = find_txt_lyrics_url(soup, song_url)
    if txt_url:
        r = session.get(txt_url, timeout=30)
        r.raise_for_status()
        lines = [ln.strip() for ln in r.text.replace("\ufeff", "").strip().splitlines() if ln.strip()]
        if junk_score(lines) < 20:
            return lines
    root = pick_best_content_root(soup)
    raw_lines = strip_navigation_blocks(strip_leading_numbered_nav_block(extract_raw_lines_from_node(root)))
    if first_line:
        anchored = extract_lyrics_by_first_line_anchor(raw_lines, first_line=first_line)
        if anchored and junk_score(anchored) < 20:
            return anchored
    trailing = extract_trailing_lyrics_block_from_raw_lines(extract_raw_lines_from_node(root))
    if trailing and junk_score(trailing) < 20:
        return trailing
    filtered_lines = extract_lines_from_node(root)
    return [] if junk_score(filtered_lines) >= 20 else filtered_lines


def is_bad_lyrics_row(row: dict) -> bool:
    first = (row.get("first_lyric_line") or "").strip()
    lyr = (row.get("lyrics") or "").strip()
    if not first or not lyr or looks_like_nav_or_footer(first):
        return True
    return junk_score([ln for ln in lyr.splitlines() if ln.strip()]) >= 20


def build_lyrics_cache(cache_csv: str = CACHE_CSV, out_csv: str = LYRICS_CSV, overrides_csv: str = OVERRIDES_CSV, sleep_s: float = 0.2):
    rows = load_cache(cache_csv)
    overrides = load_overrides(overrides_csv)
    seen = set()
    targets = []
    for r in rows:
        key = (r.get("book_id", ""), r.get("song_no", ""), r.get("url", ""))
        if key in seen or not key[2]:
            continue
        seen.add(key)
        targets.append(r)
    session = cloudscraper.create_scraper()
    session.headers.update({"User-Agent": "bremen-crosswalk-cli/1.0 (+personal research)"})
    out_rows = []
    for i, r in enumerate(targets, 1):
        book_id = (r.get("book_id") or "").strip()
        song_no = (r.get("song_no") or "").strip()
        url = ((overrides.get((book_id, song_no)) or {}).get("good_url") or r.get("url") or "").strip()
        try:
            lyr_lines = fetch_lyrics_for_url(url, session, first_line=(r.get("first_line") or "").strip())
        except Exception as e:
            print(f"[warn] lyrics failed: {book_id} {song_no} {url}: {e}", file=sys.stderr)
            lyr_lines = []
        out_rows.append({"book_id": book_id, "song_no": song_no, "title": r.get("title", ""), "url": url, "first_lyric_line": lyr_lines[0] if lyr_lines else "", "lyrics": "\n".join(lyr_lines)})
        if i % 25 == 0:
            print(f"[ok] lyrics: {i}/{len(targets)}", file=sys.stderr)
        time.sleep(sleep_s)
    write_lyrics_csv(out_csv, out_rows)
    print(f"Wrote lyrics: {out_csv} ({len(out_rows)} rows)", file=sys.stderr)


def repair_lyrics_cache(in_csv: str = LYRICS_CSV, out_csv: str = "bremen_lyrics_fixed.csv", overrides_csv: str = OVERRIDES_CSV, sleep_s: float = 0.2, limit: int | None = None, cache_csv: str = CACHE_CSV):
    rows = load_lyrics_csv(in_csv)
    overrides = load_overrides(overrides_csv)
    cache = load_cache(cache_csv)
    anchor_map = {
        ((r.get("book_id") or "").strip(), (r.get("song_no") or "").strip()): (r.get("first_line") or "").strip()
        for r in cache
    }
    session = cloudscraper.create_scraper()
    session.headers.update({"User-Agent": "bremen-crosswalk-cli/1.0 (+personal research)"})
    repaired = 0
    checked = 0
    for r in rows:
        checked += 1
        if limit is not None and repaired >= limit:
            break
        if not is_bad_lyrics_row(r):
            continue
        book_id = (r.get("book_id") or "").strip()
        song_no = (r.get("song_no") or "").strip()
        title = (r.get("title") or "").strip()
        seed_url = (r.get("url") or "").strip()
        override_url = (overrides.get((book_id, song_no)) or {}).get("good_url")
        first_line = anchor_map.get((book_id, song_no), "")
        picked_url = None
        picked_lines: list[str] = []
        for url in candidate_urls(book_id, song_no, title, seed_url, override_url=override_url):
            try:
                lyr_lines = fetch_lyrics_for_url(url, session, first_line=first_line)
            except Exception:
                continue
            if lyrics_block_is_valid(lyr_lines):
                picked_url = url
                picked_lines = lyr_lines
                break
        if picked_url is None:
            print(f"[warn] repair failed: {book_id} {song_no} {seed_url}", file=sys.stderr)
            repaired += 1
            if repaired % 10 == 0:
                print(f"[ok] repaired: {repaired}", file=sys.stderr)
            time.sleep(sleep_s)
            continue
        r["url"] = picked_url
        r["lyrics"] = "\n".join(picked_lines)
        r["first_lyric_line"] = picked_lines[0] if picked_lines else ""
        repaired += 1
        if repaired % 10 == 0:
            print(f"[ok] repaired: {repaired}", file=sys.stderr)
        time.sleep(sleep_s)
    write_lyrics_csv(out_csv, rows)
    print(f"Wrote repaired lyrics: {out_csv} (repaired {repaired} rows; checked {checked})", file=sys.stderr)
