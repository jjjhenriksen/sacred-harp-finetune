#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import io
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from sh_corpus.shape_note_normalization import norm_first_line


USER_AGENT = "sh-corpus-ingest/1.0 (+personal research)"
REQUEST_TIMEOUT = 45

SACREDHARP_TUNES_HOME = "https://www.sacredharptunes.com/"
MNHARMONY_HOME = "https://mnharmony.com/index.html"
TRUMPET_COMPOSITE_PDF = "https://www.singthetrumpet.com/the-trumpet-composite-vol-1-5.pdf"
TRUMPET_SKIP_TITLE_SUBSTRINGS = {
    "dear sacred harp",
    "in this issue",
    "why write",
    "regional report",
    "singing and remembering",
    "bob meek has become",
}
MODERN_BOOK_IDS = {"mnharmony", "sacredharptunes", "trumpet"}
HISTORICAL_FIRST_LINE_CSVS = [
    ROOT / "bremen_first_lines_canonicalized.csv",
    ROOT / "cooper2012_first_lines_raw_texasfasola.csv",
    ROOT / "southernharmony_first_lines.csv",
    ROOT / "kentucky_first_lines.csv",
]

OUTPUT_FIELDS = [
    "book_id",
    "song_no",
    "title",
    "first_line",
    "first_line_raw",
    "first_line_norm",
    "text_key",
    "url",
    "source_site",
    "composer",
    "source_hint",
]

DUPLICATE_FIELDS = [
    "source_book_id",
    "source_song_no",
    "source_title",
    "source_text_key",
    "source_url",
    "matched_book_id",
    "matched_song_no",
    "matched_title",
    "matched_url",
    "match_kind",
]


@dataclass
class OverrideRow:
    url: str
    first_line_raw: str = ""
    title: str = ""
    composer: str = ""
    skip: bool = False
    note: str = ""


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
    source_site: str
    composer: str
    source_hint: str


@dataclass
class DuplicateRow:
    source_book_id: str
    source_song_no: str
    source_title: str
    source_text_key: str
    source_url: str
    matched_book_id: str
    matched_song_no: str
    matched_title: str
    matched_url: str
    match_kind: str


@dataclass
class TitleFallback:
    title_key: str
    first_line_raw: str
    text_key: str
    source_book_id: str
    source_song_no: str


def slugify(value: str) -> str:
    lowered = (value or "").strip().lower()
    lowered = lowered.replace("’", "'").replace("‘", "'")
    lowered = re.sub(r"[^a-z0-9]+", "-", lowered)
    lowered = re.sub(r"-+", "-", lowered).strip("-")
    return lowered or "untitled"


def normalize_title(value: str) -> str:
    lowered = unicodedata.normalize("NFKD", (value or "").strip().lower())
    lowered = lowered.replace("’", "'").replace("‘", "'")
    lowered = re.sub(r"\b(?:c\.?m\.?d?|c\.?m\.?|l\.?m\.?d?|l\.?m\.?|s\.?m\.?|p\.?m\.?|m\.?d\.?)\b", " ", lowered)
    lowered = re.sub(r"\b\d+s\b", " ", lowered)
    lowered = re.sub(r"\b\d+(?:[,&]\d+)*\b", " ", lowered)
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def row_sort_key(row: OutputRow) -> tuple[str, str, str]:
    return (row.book_id, row.song_no, row.title.lower())


def parse_bool(value: str) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y"}


def load_overrides(path: Path) -> dict[str, OverrideRow]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    overrides: dict[str, OverrideRow] = {}
    for row in rows:
        url = (row.get("url") or "").strip()
        if not url:
            continue
        overrides[url] = OverrideRow(
            url=url,
            first_line_raw=(row.get("first_line_raw") or "").strip(),
            title=(row.get("title") or "").strip(),
            composer=(row.get("composer") or "").strip(),
            skip=parse_bool(row.get("skip") or ""),
            note=(row.get("note") or "").strip(),
        )
    return overrides


def apply_override(
    *,
    override: OverrideRow | None,
    title: str,
    composer: str,
    first_line_raw: str,
) -> tuple[str, str, str, bool]:
    if not override:
        return title, composer, first_line_raw, False
    if override.skip:
        return title, composer, first_line_raw, True
    if override.title:
        title = override.title
    if override.composer:
        composer = override.composer
    if override.first_line_raw:
        first_line_raw = override.first_line_raw
    return title, composer, first_line_raw, False


class HttpClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
        )

    def get_text(self, url: str, encoding: str | None = None) -> str:
        response = self.session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        if encoding:
            response.encoding = encoding
        return response.text

    def get_bytes(self, url: str) -> bytes:
        response = self.session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.content


def extract_first_lyric_line_from_page_text(
    text: str,
    *,
    title_hint: str = "",
    require_numbered_stanza: bool = False,
) -> str:
    if not text:
        return ""

    normalized = text.replace("\xa0", " ").replace("\uf0b7", " ")
    title_key = normalize_title(title_hint)
    lines = []
    for raw in normalized.splitlines():
        cleaned = re.sub(r"\s+", " ", raw).strip()
        if cleaned:
            lines.append(cleaned)

    for line in lines:
        match = re.match(r"^1[.)]?\s*(.+)$", line)
        if match:
            candidate = re.sub(r"\s+", " ", match.group(1)).strip(" -")
            if title_key and normalize_title(candidate).startswith(title_key):
                continue
            if re.search(r"[A-Za-z]", candidate) and len(candidate) >= 8:
                return candidate

    flattened = re.sub(r"\s+", " ", normalized)
    match = re.search(r"\b1[.)]?\s*([A-Za-z][^0-9]{10,180}?)\s*(?:2[.)]|\Z)", flattened)
    if match:
        candidate = re.sub(r"\s+", " ", match.group(1)).strip(" -")
        if title_key and normalize_title(candidate).startswith(title_key):
            candidate = ""
        if re.search(r"[A-Za-z]", candidate):
            return candidate

    if require_numbered_stanza:
        return ""

    for line in lines:
        if re.search(r"[A-Za-z]", line) and not re.search(r"\b(?:major|minor|concluded|issue|volume|copyright)\b", line.lower()):
            if len(line) >= 12:
                return line
    return ""


def extract_pdf_page_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    if not reader.pages:
        return ""
    page = reader.pages[0]
    try:
        text = page.extract_text(extraction_mode="layout") or ""
    except Exception:
        text = ""
    if text:
        return text
    return page.extract_text() or ""


def looks_like_weak_first_line(value: str) -> bool:
    text = re.sub(r"\s+", " ", (value or "").strip())
    if not text:
        return True
    words = text.split()
    alpha_words = [word for word in words if re.search(r"[A-Za-z]", word)]
    long_words = [word for word in alpha_words if len(re.sub(r"[^A-Za-z]", "", word)) >= 3]
    short_words = [word for word in alpha_words if len(re.sub(r"[^A-Za-z]", "", word)) <= 2]
    weird = sum(1 for ch in text if not (ch.isalnum() or ch.isspace() or ch in ".,;:!?'-()[]/&"))
    upper = f" {text.upper()} "
    if len(alpha_words) < 4:
        return True
    if len(long_words) < 3:
        return True
    if any(ch.isdigit() for ch in text):
        return True
    if weird > 4:
        return True
    if " MAJOR " in upper or " MINOR " in upper:
        return True
    if short_words and len(short_words) > max(4, len(alpha_words) // 2):
        return True
    return False


def load_title_fallbacks(paths: Iterable[Path]) -> dict[str, TitleFallback]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                title_key = normalize_title(row.get("title") or "")
                if title_key:
                    grouped.setdefault(title_key, []).append(row)

    fallbacks: dict[str, TitleFallback] = {}
    for title_key, rows in grouped.items():
        counts: dict[str, int] = {}
        for row in rows:
            text_key = (row.get("text_key") or "").strip()
            if text_key:
                counts[text_key] = counts.get(text_key, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if not ranked:
            continue
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            continue
        winning_text_key = ranked[0][0]
        preferred = sorted(
            [row for row in rows if (row.get("text_key") or "").strip() == winning_text_key],
            key=lambda row: ((row.get("book_id") or "").strip(), (row.get("song_no") or "").strip()),
        )[0]
        first_line_raw = (preferred.get("first_line_raw") or preferred.get("first_line") or "").strip()
        if not first_line_raw:
            continue
        fallbacks[title_key] = TitleFallback(
            title_key=title_key,
            first_line_raw=first_line_raw,
            text_key=winning_text_key,
            source_book_id=(preferred.get("book_id") or "").strip(),
            source_song_no=(preferred.get("song_no") or "").strip(),
        )
    return fallbacks


def maybe_apply_title_fallback(
    *,
    title: str,
    first_line_raw: str,
    title_fallbacks: dict[str, TitleFallback],
) -> tuple[str, bool]:
    if not looks_like_weak_first_line(first_line_raw):
        return first_line_raw, False
    fallback = title_fallbacks.get(normalize_title(title))
    if not fallback:
        return first_line_raw, False
    return fallback.first_line_raw, True


def make_output_row(
    *,
    book_id: str,
    song_no: str,
    title: str,
    first_line_raw: str,
    url: str,
    source_site: str,
    composer: str = "",
    source_hint: str = "",
) -> OutputRow | None:
    title = re.sub(r"\s+", " ", (title or "").strip())
    first_line_raw = re.sub(r"\s+", " ", (first_line_raw or "").strip())
    if not title or not first_line_raw:
        return None
    first_line_norm = norm_first_line(first_line_raw)
    return OutputRow(
        book_id=book_id,
        song_no=song_no.strip().lower(),
        title=title,
        first_line=first_line_raw,
        first_line_raw=first_line_raw,
        first_line_norm=first_line_norm,
        text_key=first_line_norm,
        url=url,
        source_site=source_site,
        composer=re.sub(r"\s+", " ", composer.strip()),
        source_hint=source_hint,
    )


def scrape_sacredharptunes(client: HttpClient, overrides: dict[str, OverrideRow]) -> list[OutputRow]:
    rows: list[OutputRow] = []
    home = BeautifulSoup(client.get_text(SACREDHARP_TUNES_HOME), "html.parser")
    author_urls = [urljoin(SACREDHARP_TUNES_HOME, a["href"]) for a in home.select("main table a[href]")]

    for author_url in author_urls:
        try:
            author_html = client.get_text(author_url)
        except Exception:
            continue
        author_page = BeautifulSoup(author_html, "html.parser")
        tune_urls: set[str] = set()
        for anchor in author_page.select("main a[href]"):
            href = urljoin(author_url, anchor["href"])
            if href.startswith(author_url) and href.rstrip("/") != author_url.rstrip("/"):
                tune_urls.add(href)

        for tune_url in sorted(tune_urls):
            override = overrides.get(tune_url)
            if override and override.skip:
                continue
            try:
                page_html = client.get_text(tune_url)
            except Exception:
                continue
            page = BeautifulSoup(page_html, "html.parser")
            title_node = page.select_one("main h1")
            title = title_node.get_text(" ", strip=True) if title_node else ""
            composer_node = page.select_one("main p.heading-4")
            composer = ""
            if composer_node:
                composer = re.sub(r"^\s*By\s+", "", composer_node.get_text(" ", strip=True), flags=re.IGNORECASE)

            first_line_raw = ""
            lyric_name_match = re.search(r'"lyrics"\s*:\s*\{.*?"name"\s*:\s*"(.*?)"', page_html, re.S)
            if lyric_name_match:
                first_line_raw = html.unescape(lyric_name_match.group(1)).strip()

            if not first_line_raw:
                pdf_link = page.select_one("ul.c-files a[href$='.pdf']")
                if pdf_link:
                    pdf_url = urljoin(tune_url, pdf_link["href"])
                    try:
                        pdf_text = extract_pdf_page_text(client.get_bytes(pdf_url))
                    except Exception:
                        pdf_text = ""
                    first_line_raw = extract_first_lyric_line_from_page_text(pdf_text, title_hint=title)

            title, composer, first_line_raw, should_skip = apply_override(
                override=override,
                title=title,
                composer=composer,
                first_line_raw=first_line_raw,
            )
            if should_skip:
                continue

            row = make_output_row(
                book_id="sacredharptunes",
                song_no=slugify(urlparse(tune_url).path.rstrip("/").split("/")[-1]),
                title=title,
                first_line_raw=first_line_raw,
                url=tune_url,
                source_site="sacredharptunes.com",
                composer=composer,
                source_hint="html_jsonld",
            )
            if row:
                rows.append(row)
    return rows


def scrape_mnharmony(
    client: HttpClient,
    overrides: dict[str, OverrideRow],
    title_fallbacks: dict[str, TitleFallback],
) -> list[OutputRow]:
    rows: list[OutputRow] = []
    home = BeautifulSoup(client.get_text(MNHARMONY_HOME, encoding="latin-1"), "html.parser")
    page_urls: set[str] = set()
    for anchor in home.select("a[href]"):
        href = urljoin(MNHARMONY_HOME, anchor["href"])
        if href.endswith(".html") and href != MNHARMONY_HOME and urlparse(href).netloc == "mnharmony.com":
            page_urls.add(href)

    for page_url in sorted(page_urls):
        page_html = client.get_text(page_url, encoding="latin-1")
        page = BeautifulSoup(page_html, "html.parser")
        h1 = page.find("h1")
        composer = ""
        if h1:
            composer = re.sub(r"^The Minnesota Harmony Project\s*-\s*", "", h1.get_text(" ", strip=True))

        for anchor in page.select("a[href$='.pdf']"):
            pdf_url = urljoin(page_url, anchor["href"])
            override = overrides.get(pdf_url)
            if override and override.skip:
                continue
            title = anchor.get_text(" ", strip=True).replace(".pdf", "")
            try:
                pdf_text = extract_pdf_page_text(client.get_bytes(pdf_url))
            except Exception:
                pdf_text = ""
            first_line_raw = extract_first_lyric_line_from_page_text(pdf_text, title_hint=title)
            title, composer, first_line_raw, should_skip = apply_override(
                override=override,
                title=title,
                composer=composer,
                first_line_raw=first_line_raw,
            )
            if should_skip:
                continue
            first_line_raw, used_fallback = maybe_apply_title_fallback(
                title=title,
                first_line_raw=first_line_raw,
                title_fallbacks=title_fallbacks,
            )
            row = make_output_row(
                book_id="mnharmony",
                song_no=slugify(title),
                title=title,
                first_line_raw=first_line_raw,
                url=pdf_url,
                source_site="mnharmony.com",
                composer=composer,
                source_hint="pdf_first_page+title_fallback" if used_fallback else "pdf_first_page",
            )
            if row:
                rows.append(row)
    return rows


def trumpet_title_from_lines(lines: list[str]) -> str:
    for line in lines[:18]:
        if re.search(r"\bconcluded\b", line, flags=re.IGNORECASE):
            return ""
        match = re.search(r"([A-Z][A-Za-z'’\- ]+)\.\s+[0-9A-Za-z,&. '\-]{1,20}", line)
        if match:
            candidate = match.group(1).strip()
            if len(candidate) >= 4:
                return candidate
        match = re.match(r"^([A-Z][A-Za-z'’\- ]+)\.\s+[A-Z0-9,&. '\-]{1,20}$", line)
        if match:
            return match.group(1).strip()
        match = re.match(r"^([A-Z][A-Za-z'’\- ]{3,60})\s+[0-9A-Za-z,&. '\-]{1,20}$", line)
        if match and line == line.title():
            return match.group(1).strip()
    return ""


def scrape_trumpet(
    client: HttpClient,
    overrides: dict[str, OverrideRow],
    title_fallbacks: dict[str, TitleFallback],
) -> list[OutputRow]:
    rows: list[OutputRow] = []
    reader = PdfReader(io.BytesIO(client.get_bytes(TRUMPET_COMPOSITE_PDF)))
    seen: set[tuple[str, str]] = set()

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text(extraction_mode="layout") or page.extract_text() or ""
        if not text:
            continue
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
        lines = [line for line in lines if line]
        title = trumpet_title_from_lines(lines)
        if not title:
            continue
        title_key = normalize_title(title)
        if any(bad in title_key for bad in TRUMPET_SKIP_TITLE_SUBSTRINGS):
            continue
        first_line_raw = extract_first_lyric_line_from_page_text(
            text,
            title_hint=title,
            require_numbered_stanza=True,
        )
        if not first_line_raw:
            continue
        if " page " in first_line_raw.lower():
            continue
        key = (title_key, norm_first_line(first_line_raw))
        if key in seen:
            continue
        seen.add(key)
        override_key = f"{TRUMPET_COMPOSITE_PDF}#page={page_number}"
        title, composer, first_line_raw, should_skip = apply_override(
            override=overrides.get(override_key),
            title=title,
            composer="",
            first_line_raw=first_line_raw,
        )
        if should_skip:
            continue
        first_line_raw, used_fallback = maybe_apply_title_fallback(
            title=title,
            first_line_raw=first_line_raw,
            title_fallbacks=title_fallbacks,
        )
        row = make_output_row(
            book_id="trumpet",
            song_no=f"p{page_number}",
            title=title,
            first_line_raw=first_line_raw,
            url=TRUMPET_COMPOSITE_PDF,
            source_site="singthetrumpet.com",
            composer=composer,
            source_hint=f"composite_pdf_page_{page_number}" + ("+title_fallback" if used_fallback else ""),
        )
        if row:
            rows.append(row)
    return rows


def dedupe_rows(rows: Iterable[OutputRow]) -> list[OutputRow]:
    deduped: dict[tuple[str, str, str, str], OutputRow] = {}
    for row in rows:
        key = (row.book_id, row.song_no, normalize_title(row.title), row.text_key)
        deduped[key] = row
    return sorted(deduped.values(), key=row_sort_key)


def build_duplicate_report(rows: list[OutputRow], existing_rows: list[dict]) -> list[DuplicateRow]:
    by_text_key: dict[str, list[dict]] = {}
    by_title_and_text: dict[tuple[str, str], list[dict]] = {}
    for row in existing_rows:
        if (row.get("book_id") or "").strip() in MODERN_BOOK_IDS:
            continue
        text_key = (row.get("text_key") or "").strip()
        title_key = normalize_title(row.get("title") or "")
        if text_key:
            by_text_key.setdefault(text_key, []).append(row)
            by_title_and_text.setdefault((title_key, text_key), []).append(row)

    report: list[DuplicateRow] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for row in rows:
        title_key = normalize_title(row.title)
        matched = by_title_and_text.get((title_key, row.text_key), [])
        match_kind = "same_title_and_text_key"
        if not matched:
            matched = by_text_key.get(row.text_key, [])
            match_kind = "same_text_key"
        for existing in matched:
            key = (row.book_id, row.song_no, existing.get("book_id", ""), existing.get("song_no", ""), match_kind)
            if key in seen:
                continue
            seen.add(key)
            report.append(
                DuplicateRow(
                    source_book_id=row.book_id,
                    source_song_no=row.song_no,
                    source_title=row.title,
                    source_text_key=row.text_key,
                    source_url=row.url,
                    matched_book_id=existing.get("book_id", ""),
                    matched_song_no=existing.get("song_no", ""),
                    matched_title=existing.get("title", ""),
                    matched_url=existing.get("url", ""),
                    match_kind=match_kind,
                )
            )
    report.sort(key=lambda r: (r.source_book_id, r.source_song_no, r.matched_book_id, r.matched_song_no))
    return report


def suppress_same_title_and_text_key_duplicates(rows: list[OutputRow], existing_rows: list[dict]) -> tuple[list[OutputRow], int]:
    existing_keys = {
        (normalize_title(row.get("title") or ""), (row.get("text_key") or "").strip())
        for row in existing_rows
        if (row.get("book_id") or "").strip() not in MODERN_BOOK_IDS
        if (row.get("text_key") or "").strip()
    }
    kept: list[OutputRow] = []
    suppressed = 0
    for row in rows:
        key = (normalize_title(row.title), row.text_key)
        if key in existing_keys:
            suppressed += 1
            continue
        kept.append(row)
    return kept, suppressed


def read_existing_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_output_rows(path: Path, rows: list[OutputRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_duplicate_rows(path: Path, rows: list[DuplicateRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DUPLICATE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def main() -> int:
    parser = argparse.ArgumentParser(prog="modern_shape_note_scrape_firstlines.py")
    parser.add_argument("--out", default="modern_shape_note_first_lines.csv")
    parser.add_argument("--duplicates-out", default="modern_shape_note_duplicate_candidates.csv")
    parser.add_argument("--compare-existing", default="combined_first_lines_canonicalized.csv")
    parser.add_argument("--overrides", default="modern_shape_note_overrides.csv")
    args = parser.parse_args()

    client = HttpClient()
    overrides = load_overrides(Path(args.overrides))
    title_fallbacks = load_title_fallbacks(HISTORICAL_FIRST_LINE_CSVS)
    scraped_rows = dedupe_rows(
        [
            *scrape_sacredharptunes(client, overrides),
            *scrape_mnharmony(client, overrides, title_fallbacks),
            *scrape_trumpet(client, overrides, title_fallbacks),
        ]
    )
    existing_rows = read_existing_rows(Path(args.compare_existing))
    duplicate_rows = build_duplicate_report(scraped_rows, existing_rows)
    all_rows = scraped_rows
    suppressed = 0

    write_output_rows(Path(args.out), all_rows)
    write_duplicate_rows(Path(args.duplicates_out), duplicate_rows)

    counts: dict[str, int] = {}
    for row in all_rows:
        counts[row.book_id] = counts.get(row.book_id, 0) + 1

    print(f"[ok] wrote: {args.out}")
    print(f"[ok] total rows: {len(all_rows)}")
    print(f"[ok] suppressed exact title+text duplicates: {suppressed}")
    for book_id in sorted(counts):
        print(f"- {book_id}: {counts[book_id]}")
    print(f"[ok] duplicate candidates: {len(duplicate_rows)}")
    print(f"[ok] duplicate report: {args.duplicates_out}")
    print(f"[ok] overrides loaded: {len(overrides)}")
    print(f"[ok] title fallbacks loaded: {len(title_fallbacks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
