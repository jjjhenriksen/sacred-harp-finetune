#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sh_corpus.shape_note_normalization import norm_first_line


BOOK_ID = "shcooper2012"
BASE_INDEX_URL = "http://resources.texasfasola.org/index/"
USER_AGENT = "sh-corpus-ingest/1.0 (+personal research)"
SONG_NO_ALIASES = {
    "207h": "207b",
}


@dataclass
class IndexEntry:
    song_no: str
    title: str
    first_line_index_raw: str
    url: str


@dataclass
class OutputRow:
    book_id: str
    song_no: str
    title: str
    first_line: str
    first_line_raw: str
    first_line_norm: str
    text_key: str
    url: str
    first_line_source: str
    first_line_index_raw: str
    first_line_poetry_raw: str


class Fetcher:
    def __init__(self, min_interval_s: float = 1.0, retries: int = 5):
        self.min_interval_s = min_interval_s
        self.retries = retries
        self._last_request_at = 0.0

    def _rate_limit(self) -> None:
        now = time.time()
        elapsed = now - self._last_request_at
        if elapsed < self.min_interval_s:
            time.sleep(self.min_interval_s - elapsed)

    def get_html(self, url: str) -> str:
        last_err: Exception | None = None
        for attempt in range(1, self.retries + 1):
            self._rate_limit()
            req = Request(url, headers={"User-Agent": USER_AGENT})
            try:
                with urlopen(req, timeout=30) as resp:
                    self._last_request_at = time.time()
                    return resp.read().decode("utf-8", "ignore")
            except HTTPError as e:
                self._last_request_at = time.time()
                last_err = e
                if e.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    time.sleep(1.5 * attempt)
                    continue
                raise
            except URLError as e:
                self._last_request_at = time.time()
                last_err = e
                if attempt < self.retries:
                    time.sleep(1.5 * attempt)
                    continue
                raise
        assert last_err is not None
        raise last_err


def cache_path_for_url(cache_dir: Path, url: str) -> Path:
    p = urlparse(url)
    host = p.netloc.replace(":", "_")
    rel = (p.path or "/").lstrip("/")
    if not rel:
        rel = "root.html"
    if rel.endswith("/"):
        rel += "index.html"
    return cache_dir / host / rel


def load_html(url: str, fetcher: Fetcher, cache_dir: Path, refresh: bool) -> str:
    path = cache_path_for_url(cache_dir, url)
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")
    html = fetcher.get_html(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return html


def normalize_song_no(song_no: str) -> str:
    s = (song_no or "").strip().lower()
    if re.fullmatch(r"\d+[a-z]?", s):
        m = re.match(r"^0*(\d+)([a-z]?)$", s)
        if m:
            s = f"{int(m.group(1))}{m.group(2)}"
            return SONG_NO_ALIASES.get(s, s)
    if s:
        return SONG_NO_ALIASES.get(s, s)
    return s


def normalize_index_poetry_url(page_url: str, href: str) -> str:
    abs_url = urljoin(page_url, href)
    p = urlparse(abs_url)
    if p.netloc != "resources.texasfasola.org":
        return abs_url
    path = p.path or ""
    m = re.match(r"^/index/poetry/(.+\.html)$", path, flags=re.IGNORECASE)
    if m:
        path = f"/poetry/cooper/{m.group(1)}"
    else:
        m2 = re.match(r"^/poetry/(.+\.html)$", path, flags=re.IGNORECASE)
        if m2 and not path.lower().startswith("/poetry/cooper/"):
            path = f"/poetry/cooper/{m2.group(1)}"
    return f"http://resources.texasfasola.org{path}"


def discover_index_pages(fetcher: Fetcher, cache_dir: Path, refresh: bool) -> List[str]:
    root_html = load_html(BASE_INDEX_URL, fetcher, cache_dir, refresh)
    soup = BeautifulSoup(root_html, "html.parser")
    pages: set[str] = set()
    queue: List[str] = []

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if re.match(r"^1st_[a-z0-9]+\.html$", href, flags=re.IGNORECASE):
            queue.append(urljoin(BASE_INDEX_URL, href))

    if not queue:
        queue.append(urljoin(BASE_INDEX_URL, "1st_a.html"))

    # Crawl first-line index pages to collect all linked 1st_*.html pages.
    while queue:
        current = queue.pop(0)
        if current in pages:
            continue
        try:
            html = load_html(current, fetcher, cache_dir, refresh)
        except HTTPError as e:
            if e.code == 404:
                continue
            raise
        pages.add(current)
        csoup = BeautifulSoup(html, "html.parser")
        for a in csoup.find_all("a", href=True):
            href = a["href"].strip()
            if re.match(r"^1st_[a-z0-9]+\.html$", href, flags=re.IGNORECASE):
                nxt = urljoin(current, href)
                if nxt not in pages:
                    queue.append(nxt)

    if pages:
        return sorted(pages)

    candidates = [f"1st_{ch}.html" for ch in "abcdefghijklmnopqrstuvwxyz"] + [f"1st_{d}.html" for d in "0123456789"]
    out: List[str] = []
    for name in candidates:
        url = urljoin(BASE_INDEX_URL, name)
        try:
            html = load_html(url, fetcher, cache_dir, refresh)
        except HTTPError as e:
            if e.code == 404:
                continue
            raise
        if "first line index" in html.lower():
            out.append(url)
    return out


def parse_song_anchor_text(text: str) -> tuple[str, str]:
    s = re.sub(r"\s+", " ", (text or "").strip())
    m = re.match(r"^([0-9ivxlcdm]+[a-z]?)\s+(.+)$", s, flags=re.IGNORECASE)
    if m:
        return normalize_song_no(m.group(1)), m.group(2).strip()
    m2 = re.match(r"^([0-9ivxlcdm]+[a-z]?)$", s, flags=re.IGNORECASE)
    if m2:
        return normalize_song_no(m2.group(1)), ""
    return "", s


def parse_index_page(url: str, html: str) -> List[IndexEntry]:
    soup = BeautifulSoup(html, "html.parser")
    out: List[IndexEntry] = []

    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue
        first_line_index_raw = tds[0].get_text(" ", strip=True)
        if not first_line_index_raw:
            continue
        link_td = tds[-1]
        a = link_td.find("a", href=True)
        if not a:
            continue

        href = a["href"].strip()
        song_text = a.get_text(" ", strip=True)
        song_no, title = parse_song_anchor_text(song_text)
        if not song_no:
            name_attr = (a.get("name") or "").strip()
            if name_attr:
                song_no = normalize_song_no(name_attr)
        if not song_no:
            continue

        target_url = normalize_index_poetry_url(url, href)
        out.append(
            IndexEntry(
                song_no=song_no,
                title=title,
                first_line_index_raw=first_line_index_raw,
                url=target_url,
            )
        )

    return out


def extract_title_from_poetry_page(soup: BeautifulSoup, song_no: str) -> str:
    if soup.find("h2"):
        t = soup.find("h2").get_text(" ", strip=True)
    elif soup.title:
        t = soup.title.get_text(" ", strip=True)
    else:
        return ""

    t = re.sub(r"\s+", " ", t).strip()
    sn = re.escape(song_no)
    t = re.sub(rf"\b{sn}\b", "", t, flags=re.IGNORECASE).strip(" -–—")
    return t.strip()


def extract_poetry_first_line(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tr in soup.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 3:
            continue
        stanza_label = tds[0].get_text(" ", strip=True)
        if not re.fullmatch(r"1[.)]?", stanza_label):
            continue
        stanza_td = tds[2]
        lines = [re.sub(r"\s+", " ", x.strip()) for x in stanza_td.get_text("\n", strip=True).splitlines()]
        for ln in lines:
            if not ln:
                continue
            if re.fullmatch(r"\d+[.)]?", ln):
                continue
            if ln.lower() in {"chorus", "refrain"}:
                continue
            return ln

    # Fallback: first non-nav line in page body text.
    body = soup.get_text("\n", strip=True)
    for ln in body.splitlines():
        s = re.sub(r"\s+", " ", ln.strip())
        if not s:
            continue
        if s.startswith("[") and s.endswith("]"):
            continue
        if re.fullmatch(r"\d+[.)]?", s):
            continue
        return s
    return ""


def dedupe(entries: Iterable[IndexEntry]) -> List[IndexEntry]:
    seen: set[tuple[str, str, str, str]] = set()
    out: List[IndexEntry] = []
    for e in entries:
        key = (e.song_no, e.title, e.first_line_index_raw, e.url)
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def write_csv(path: Path, rows: List[OutputRow]) -> None:
    fields = [
        "book_id",
        "song_no",
        "title",
        "first_line",
        "first_line_raw",
        "first_line_norm",
        "text_key",
        "url",
        "first_line_source",
        "first_line_index_raw",
        "first_line_poetry_raw",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r.__dict__)


def read_song_no_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as f:
        return {normalize_song_no((r.get("song_no") or "").strip()) for r in csv.DictReader(f) if (r.get("song_no") or "").strip()}


def main() -> int:
    ap = argparse.ArgumentParser(prog="texasfasola_scrape_cooper_firstlines.py")
    ap.add_argument("--out", default="cooper2012_first_lines_raw_texasfasola.csv")
    ap.add_argument("--cache-dir", default=".cache/texasfasola_cooper")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--compare-hymnary", default="cooper2012_first_lines_raw_v3.csv")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    fetcher = Fetcher(min_interval_s=1.0, retries=5)

    pages = discover_index_pages(fetcher, cache_dir, args.refresh)
    if not pages:
        raise SystemExit("No 1st_*.html pages discovered.")
    print(f"[ok] index pages discovered: {len(pages)}")

    all_entries: List[IndexEntry] = []
    for p in pages:
        html = load_html(p, fetcher, cache_dir, args.refresh)
        rows = parse_index_page(p, html)
        print(f"[ok] {p.split('/')[-1]}: {len(rows)} rows")
        all_entries.extend(rows)

    entries = dedupe(all_entries)
    print(f"[ok] deduped rows: {len(entries)}")

    out_rows: List[OutputRow] = []
    validation_rows: dict[str, OutputRow] = {}

    for e in entries:
        poetry_html = ""
        poetry_first_line_raw = ""
        title = e.title.strip()
        try:
            poetry_html = load_html(e.url, fetcher, cache_dir, args.refresh)
        except HTTPError:
            poetry_html = ""
        except URLError:
            poetry_html = ""

        if poetry_html:
            poetry_first_line_raw = extract_poetry_first_line(poetry_html)
            if not title:
                soup = BeautifulSoup(poetry_html, "html.parser")
                title = extract_title_from_poetry_page(soup, e.song_no)

        if poetry_first_line_raw:
            first_line_source = "poetry_page"
            first_line_raw = poetry_first_line_raw
        else:
            first_line_source = "firstline_index"
            first_line_raw = e.first_line_index_raw

        first_line_norm = norm_first_line(first_line_raw)
        row = OutputRow(
            book_id=BOOK_ID,
            song_no=e.song_no,
            title=title,
            first_line=first_line_raw,
            first_line_raw=first_line_raw,
            first_line_norm=first_line_norm,
            text_key=first_line_norm,
            url=e.url,
            first_line_source=first_line_source,
            first_line_index_raw=e.first_line_index_raw,
            first_line_poetry_raw=poetry_first_line_raw,
        )
        out_rows.append(row)
        validation_rows[e.song_no] = row

    out_rows.sort(key=lambda r: (r.song_no, r.title.lower(), r.url))
    write_csv(Path(args.out), out_rows)

    unique_song_no = len({r.song_no for r in out_rows})
    print(f"[ok] wrote: {args.out}")
    print(f"[ok] total rows: {len(out_rows)}")
    print(f"[ok] unique song_no: {unique_song_no}")

    if "63" in validation_rows:
        r = validation_rows["63"]
        print(f"[check] 63\t{r.title}\t{r.first_line_raw}\t({r.first_line_source})")
    if "47b" in validation_rows:
        r = validation_rows["47b"]
        print(f"[check] 47b\t{r.title}\t{r.first_line_raw}\t({r.first_line_source})")

    compare_path = Path(args.compare_hymnary)
    hymnary_song_nos = read_song_no_set(compare_path)
    if hymnary_song_nos:
        new_song_nos = {r.song_no for r in out_rows}
        missing_from_new = sorted(hymnary_song_nos - new_song_nos)
        extra_in_new = sorted(new_song_nos - hymnary_song_nos)
        print(f"[compare] hymnary unique song_no: {len(hymnary_song_nos)} ({compare_path})")
        print(f"[compare] texasfasola unique song_no: {len(new_song_nos)}")
        if missing_from_new:
            print(f"[compare] missing from texasfasola ({len(missing_from_new)}): {', '.join(missing_from_new[:120])}")
        else:
            print("[compare] missing from texasfasola: none")
        if extra_in_new:
            print(f"[compare] extra in texasfasola ({len(extra_in_new)}): {', '.join(extra_in_new[:120])}")
        else:
            print("[compare] extra in texasfasola: none")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
