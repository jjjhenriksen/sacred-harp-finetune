#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from modern_shape_note_scrape_firstlines import (
    HttpClient,
    OverrideRow,
    SACREDHARP_TUNES_HOME,
    apply_override,
    load_overrides,
    slugify,
)


OUTPUT_FIELDS = [
    "book_id",
    "song_no",
    "title",
    "url",
    "first_lyric_line",
    "lyrics",
    "source_site",
    "composer",
    "lyrics_source",
]


@dataclass
class LyricsOverrideRow:
    url: str
    first_lyric_line: str = ""
    lyrics: str = ""
    title: str = ""
    composer: str = ""
    skip: bool = False
    note: str = ""


def parse_bool(value: str) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "y"}


def load_lyrics_overrides(path: Path) -> dict[str, LyricsOverrideRow]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    overrides: dict[str, LyricsOverrideRow] = {}
    for row in rows:
        url = (row.get("url") or "").strip()
        if not url:
            continue
        overrides[url] = LyricsOverrideRow(
            url=url,
            first_lyric_line=(row.get("first_lyric_line") or "").strip(),
            lyrics=(row.get("lyrics") or "").strip(),
            title=(row.get("title") or "").strip(),
            composer=(row.get("composer") or "").strip(),
            skip=parse_bool(row.get("skip") or ""),
            note=(row.get("note") or "").strip(),
        )
    return overrides


def apply_lyrics_override(
    *,
    override: LyricsOverrideRow | None,
    title: str,
    composer: str,
    first_line: str,
    lyrics: str,
) -> tuple[str, str, str, str, bool, str]:
    if not override:
        source = "html_jsonld" if lyrics else ""
        return title, composer, first_line, lyrics, False, source
    if override.skip:
        return title, composer, first_line, lyrics, True, ""
    if override.title:
        title = override.title
    if override.composer:
        composer = override.composer
    if override.lyrics:
        lyrics = override.lyrics
    if override.first_lyric_line:
        first_line = override.first_lyric_line
    elif lyrics and not first_line:
        first_line = lyrics.splitlines()[0].strip()
    source = "manual_override" if override.lyrics else ("html_jsonld" if lyrics else "")
    return title, composer, first_line, lyrics, False, source


def extract_json_object_after_key(field_name: str, text: str) -> dict[str, object]:
    match = re.search(rf'"{re.escape(field_name)}"\s*:\s*\{{', text)
    if not match:
        return {}
    brace_start = text.find("{", match.start())
    if brace_start < 0:
        return {}

    depth = 0
    in_string = False
    escaped = False
    for idx in range(brace_start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[brace_start : idx + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


def normalize_lyrics_html(lyrics_html: str) -> str:
    if not lyrics_html:
        return ""
    cleaned = lyrics_html.strip()
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1]
    cleaned = cleaned.replace("\\/", "/")
    cleaned = re.sub(r"<br\s*/?>", "\n", cleaned, flags=re.IGNORECASE)
    soup = BeautifulSoup(cleaned, "html.parser")
    paragraphs: list[str] = []
    for p in soup.find_all("p"):
        text = p.get_text("\n", strip=True)
        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r"[ \t]+", " ", text).strip()
        if not text or text == "\xa0":
            continue
        paragraphs.append(text)
    if paragraphs:
        return "\n\n".join(paragraphs)
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


def scrape_sacredharptunes_lyrics(
    client: HttpClient,
    overrides: dict[str, OverrideRow],
    lyrics_overrides: dict[str, LyricsOverrideRow],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
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
            lyrics_override = lyrics_overrides.get(tune_url)
            if override and override.skip:
                continue
            if lyrics_override and lyrics_override.skip:
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

            lyrics_obj = extract_json_object_after_key("lyrics", page_html)
            first_line = html.unescape(str(lyrics_obj.get("name") or ""))
            lyrics_html = str(lyrics_obj.get("text") or "")
            lyrics = normalize_lyrics_html(lyrics_html)
            title, composer, first_line, should_skip = apply_override(
                override=override,
                title=title,
                composer=composer,
                first_line_raw=first_line,
            )
            if should_skip:
                continue
            title, composer, first_line, lyrics, should_skip, lyrics_source = apply_lyrics_override(
                override=lyrics_override,
                title=title,
                composer=composer,
                first_line=first_line,
                lyrics=lyrics,
            )
            if should_skip or not lyrics:
                continue
            if not first_line:
                first_line = lyrics.splitlines()[0].strip()

            rows.append(
                {
                    "book_id": "sacredharptunes",
                    "song_no": slugify(urlparse(tune_url).path.rstrip("/").split("/")[-1]),
                    "title": title,
                    "url": tune_url,
                    "first_lyric_line": first_line,
                    "lyrics": lyrics,
                    "source_site": "sacredharptunes.com",
                    "composer": composer,
                    "lyrics_source": lyrics_source,
                }
            )
    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(prog="modern_shape_note_scrape_lyrics.py")
    parser.add_argument("--out", default="modern_shape_note_lyrics.csv")
    parser.add_argument("--overrides", default="modern_shape_note_overrides.csv")
    parser.add_argument("--lyrics-overrides", default="modern_shape_note_lyrics_overrides.csv")
    args = parser.parse_args()

    client = HttpClient()
    overrides = load_overrides(Path(args.overrides))
    lyrics_overrides = load_lyrics_overrides(Path(args.lyrics_overrides))
    rows = scrape_sacredharptunes_lyrics(client, overrides, lyrics_overrides)
    rows.sort(key=lambda row: (row["book_id"], row["song_no"], row["title"].lower()))
    write_rows(Path(args.out), rows)

    print(f"[ok] wrote: {args.out}")
    print(f"[ok] rows: {len(rows)}")
    print(f"[ok] overrides loaded: {len(overrides)}")
    print(f"[ok] lyrics overrides loaded: {len(lyrics_overrides)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
