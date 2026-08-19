#!/usr/bin/env python3
"""Fetch public, source-linked metadata for Sacred Harp RAG records.

The generated CSV is an enrichment layer only: the Obsidian notes remain the
source of truth.  Fasola supplies tune/words/music/meter records, while the
shapenote.net MusicXML catalog supplies key and time signature from scores.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

from bs4 import BeautifulSoup


MUSIC_URL = "https://shapenote.net/music.htm"
ED_DATA_URL = "https://raw.githubusercontent.com/edjw/Sacred-Harp-datasets/master/sacred_harp_songs_data.csv"
COOPER_BASE = "http://resources.texasfasola.org/index/"
COOPER_TIME_URL = "https://fasola.org/indexes/timesigs/cooper/"
SH2025_KEY_CHANGES_URL = "https://sacredharp.com/2025/08/13/verse-key-and-part-changes-in-the-sacred-harp-2025-edition/"
FASOLA_INDEXES = {
    "sh1991": "https://fasola.org/indexes/1991/?v=pagenum",
    "sh2025": "https://www.fasola.org/indexes/2025/?v=pagenum",
}
SECTION_TO_BOOK = {
    "Sacred Harp (1991 Denson Revision)": "sh1991",
    "Sacred Harp (2025 Revision)": "sh2025",
    # shapenote.net labels this simply “Cooper Revision”; the local corpus
    # uses the more specific shcooper2012 identity for the same song family.
    "Sacred Harp (Cooper Revision)": "shcooper2012",
    "The Christian Harmony 2010": "ch7",
    "Southern Harmony": "southernharmony",
}
FIELDS = (
    "book_id", "song_no", "tune", "meter", "time_signature",
    "key_signature", "composer", "lyricist", "book_edition", "source_url",
    "confidence", "notes",
)


def fetch(url: str, cache_dir: Path, timeout: float = 30.0) -> bytes:
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(urlparse(url).path).suffix or ".bin"
    cache_path = cache_dir / f"{hashlib.sha256(url.encode()).hexdigest()[:24]}{suffix}"
    if cache_path.exists():
        return cache_path.read_bytes()
    request = Request(url, headers={"User-Agent": "SacredHarpRAG/1.0 (metadata research)"})
    with urlopen(request, timeout=timeout) as response:
        content = response.read()
    cache_path.write_bytes(content)
    return content


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def first_text(root: ET.Element, name: str) -> str:
    for element in root.iter():
        if local_name(element.tag) == name and (element.text or "").strip():
            return (element.text or "").strip()
    return ""


def musicxml_metadata(url: str, cache_dir: Path) -> dict[str, str]:
    with zipfile.ZipFile(BytesIO(fetch(url, cache_dir))) as archive:
        xml_names = [name for name in archive.namelist() if name.endswith(".xml") and "container" not in name]
        if not xml_names:
            raise ValueError("MusicXML archive has no score XML")
        root = ET.fromstring(archive.read(xml_names[0]))

    title = first_text(root, "work-title") or first_text(root, "movement-title")
    composer = ""
    for element in root.iter():
        if local_name(element.tag) == "creator" and element.attrib.get("type") == "composer":
            composer = (element.text or "").strip()
            if composer:
                break
    if composer.lower().startswith("alto by"):
        composer = ""
    key = next((element for element in root.iter() if local_name(element.tag) == "key"), None)
    time_element = next((element for element in root.iter() if local_name(element.tag) == "time"), None)
    key_signature = ""
    if key is not None:
        fifths = first_text(key, "fifths")
        mode = first_text(key, "mode") or "major"
        try:
            fifths_value = int(fifths)
        except ValueError:
            fifths_value = None
        major = {-7: "Cb", -6: "Gb", -5: "Db", -4: "Ab", -3: "Eb", -2: "Bb", -1: "F", 0: "C", 1: "G", 2: "D", 3: "A", 4: "E", 5: "B", 6: "F#", 7: "C#"}
        minor = {-7: "Ab", -6: "Eb", -5: "Bb", -4: "F", -3: "C", -2: "G", -1: "D", 0: "A", 1: "E", 2: "B", 3: "F#", 4: "C#", 5: "G#", 6: "D#", 7: "A#"}
        if fifths_value in major:
            key_signature = f"{(minor if mode.lower() == 'minor' else major)[fifths_value]} {mode.lower()}"
        elif fifths:
            key_signature = f"fifths={fifths} {mode}"
    time_signature = ""
    if time_element is not None:
        beats = first_text(time_element, "beats")
        beat_type = first_text(time_element, "beat-type")
        if beats and beat_type:
            time_signature = f"{beats}/{beat_type}"
    return {"tune": title, "composer": composer, "key_signature": key_signature, "time_signature": time_signature}


def song_identity(text: str) -> tuple[str, str] | None:
    match = re.match(r"\s*(\d+[a-z]?)\s+(.+?)(?:\s+\[|$)", text, re.IGNORECASE)
    if not match:
        title = text.split("[", 1)[0].strip()
        return ("", title) if title else None
    return match.group(1), match.group(2).strip()


def parse_music_catalog(html: bytes) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    section = ""
    records: list[dict[str, str]] = []
    for element in soup.find_all(["h3", "li"]):
        if element.name == "h3":
            section = element.get_text(" ", strip=True)
            continue
        book_id = next((book for heading, book in SECTION_TO_BOOK.items() if section.startswith(heading)), None)
        if not book_id:
            continue
        identity = song_identity(element.get_text(" ", strip=True))
        if not identity:
            continue
        song_no, title = identity
        links = [urljoin(MUSIC_URL, anchor.get("href", "")) for anchor in element.find_all("a") if "musicxml/" in anchor.get("href", "")]
        if not links:
            continue
        records.append({"book_id": book_id, "song_no": song_no, "title": title, "musicxml_url": links[0], "variant_urls": "; ".join(links)})
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for record in records:
        unique.setdefault((record["book_id"], (record["song_no"] or record["title"]).lower()), record)
    return list(unique.values())


def parse_fasola_index(html: bytes, edition: str) -> dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    pages: dict[str, str] = {}
    for anchor in soup.find_all("a", href=True):
        match = re.search(r"[?&]p=(\d+[a-z]?)", anchor["href"], re.IGNORECASE)
        if match:
            song_no = match.group(1)
            pages.setdefault(song_no.lower(), f"https://fasola.org/indexes/{edition}/?p={song_no}")
    return pages


def detail(soup: BeautifulSoup, class_name: str) -> tuple[str, str]:
    paragraph = soup.find("p", class_=lambda classes: classes and class_name in classes)
    if paragraph is None:
        return "", ""
    anchor = paragraph.find("a")
    value = anchor.get_text(" ", strip=True) if anchor else paragraph.get_text(" ", strip=True)
    value = re.sub(r"^\s*(?:Words|Music|Meter):\s*", "", value, flags=re.IGNORECASE).strip(" ,")
    year_match = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", paragraph.get_text(" ", strip=True))
    return value, year_match.group(1) if year_match else ""


def parse_fasola_song(html: bytes, url: str) -> dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    # Fasola uses h1 for the index banner and h2 for the individual song.
    heading = soup.find("h2") or soup.find("h1")
    tune = ""
    if heading:
        match = re.match(r"\s*\d+[a-z]?\s+(.+)", heading.get_text(" ", strip=True), re.IGNORECASE)
        tune = (match.group(1) if match else heading.get_text(" ", strip=True)).strip()
    words, words_year = detail(soup, "words")
    music, music_year = detail(soup, "music")
    meter, _ = detail(soup, "meter")
    return {
        "tune": tune, "lyricist": words, "composer": music, "meter": meter,
        "source_url": url, "notes": f"Fasola online index; words/music dates {words_year or '?'} / {music_year or '?'}",
    }


def parse_cooper_index(html: bytes) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    pages: dict[str, str] = {}
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "")
        if not href.startswith("poetry/") or not href.endswith(".html"):
            continue
        match = re.match(r"\s*(\d+[tb]?)\s+", anchor.get_text(" ", strip=True), re.IGNORECASE)
        if match:
            pages.setdefault(match.group(1).lower(), urljoin(COOPER_BASE, href))
    return list(pages.items())


def parse_cooper_song(html: bytes, url: str) -> dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    heading = soup.find("h2")
    tune = heading.get_text(" ", strip=True) if heading else ""
    tune = re.sub(r"\s+\d+[tb]?\s*$", "", tune, flags=re.IGNORECASE).strip()

    def labeled(label: str) -> str:
        marker = soup.find(string=lambda value: value and value.strip().lower().startswith(f"{label.lower()}:"))
        if marker is None:
            return ""
        anchor = marker.find_next("a")
        return anchor.get_text(" ", strip=True) if anchor else ""

    composer = labeled("Tune") or labeled("Music")
    lyricist = labeled("Lyrics") or labeled("Words") or labeled("Poetry")
    meter = labeled("Meter")
    return {
        "tune": tune, "composer": composer, "lyricist": lyricist, "meter": meter,
        "source_url": url, "confidence": "source", "notes": "TexasFasola Cooper online index",
    }


def parse_time_signature_index(html: bytes) -> dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    result: dict[str, str] = {}
    for row in soup.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
        for start in (0, 3):
            if len(cells) <= start + 2:
                continue
            signature, _name, page = cells[start:start + 3]
            if re.fullmatch(r"(?:\d+|Multiple)(?:/\d+)?", signature) and re.fullmatch(r"\d+[ab]?", page, re.IGNORECASE):
                song_no = page[:-1] + ("t" if page[-1].lower() == "a" else "b") if page[-1].isalpha() else page
                result.setdefault(song_no.lower(), signature)
    return result


def merge(rows: dict[tuple[str, str], dict[str, str]], incoming: dict[str, str]) -> None:
    key = (incoming["book_id"], incoming["song_no"].lower())
    current = rows.setdefault(key, {field: "" for field in FIELDS})
    for field in FIELDS:
        value = incoming.get(field, "").strip()
        if not value:
            continue
        if field in {"source_url", "notes"} and current[field] and value not in current[field]:
            current[field] = f"{current[field]}; {value}"
        elif not current[field]:
            current[field] = value
    current["book_id"] = incoming["book_id"]
    current["song_no"] = incoming["song_no"]
    current["confidence"] = "source"


def add_musicxml(rows: dict[tuple[str, str], dict[str, str]], record: dict[str, str], cache_dir: Path) -> str:
    song_no = record["song_no"]
    if not song_no:
        wanted = re.sub(r"[^a-z0-9]+", "", re.sub(r"\([^)]*\)", "", record["title"].lower()))
        matches = [
            row for (book_id, _), row in rows.items()
            if book_id == record["book_id"] and re.sub(r"[^a-z0-9]+", "", re.sub(r"\([^)]*\)", "", row.get("tune", "").lower())) == wanted
        ]
        if len(matches) != 1:
            return "error"
        song_no = matches[0]["song_no"]
    try:
        values = musicxml_metadata(record["musicxml_url"], cache_dir)
    except Exception as error:  # retain the error in the output instead of losing the song
        merge(rows, {"book_id": record["book_id"], "song_no": song_no, "tune": record["title"], "source_url": record["musicxml_url"], "confidence": "review", "notes": f"MusicXML fetch/parse failed: {error}"})
        return "error"
    merge(rows, {
        "book_id": record["book_id"], "song_no": song_no, "tune": values["tune"] or record["title"],
        "composer": values["composer"], "key_signature": values["key_signature"], "time_signature": values["time_signature"],
        "source_url": record["musicxml_url"], "confidence": "source",
        "notes": f"MusicXML from shapenote.net; primary catalog variant; variants: {record['variant_urls']}",
    })
    return "ok"


def add_fasola(rows: dict[tuple[str, str], dict[str, str]], item: tuple[str, str, str], cache_dir: Path) -> str:
    book_id, song_no, url = item
    try:
        values = parse_fasola_song(fetch(url, cache_dir), url)
        merge(rows, {"book_id": book_id, "song_no": song_no, **values})
        return "ok"
    except Exception as error:
        merge(rows, {"book_id": book_id, "song_no": song_no, "source_url": url, "confidence": "review", "notes": f"Fasola fetch/parse failed: {error}"})
        return "error"


def add_cooper(rows: dict[tuple[str, str], dict[str, str]], item: tuple[str, str], cache_dir: Path) -> str:
    song_no, url = item
    try:
        values = parse_cooper_song(fetch(url, cache_dir), url)
        merge(rows, {"book_id": "shcooper2012", "song_no": song_no, **values})
        return "ok"
    except Exception as error:
        merge(rows, {"book_id": "shcooper2012", "song_no": song_no, "source_url": url, "confidence": "review", "notes": f"Cooper fetch/parse failed: {error}"})
        return "error"


def add_ed_dataset(rows: dict[tuple[str, str], dict[str, str]], html: bytes) -> int:
    added = 0
    for item in csv.DictReader(html.decode("utf-8-sig").splitlines()):
        song_no = (item.get("song_number") or item.get("song_no") or "").strip()
        if not song_no:
            continue
        before = ("sh1991", song_no.lower()) in rows
        merge(rows, {
            "book_id": "sh1991", "song_no": song_no,
            "tune": item.get("title", ""), "meter": item.get("poetic_meter", "") or item.get("meter", ""),
            "time_signature": item.get("time_signature", ""), "composer": item.get("composer_source", ""),
            "lyricist": item.get("poet_source", ""), "source_url": ED_DATA_URL,
            "confidence": "secondary source", "notes": "Ed Johnson-Williams Sacred Harp dataset fallback",
        })
        if not before:
            added += 1
    return added


def add_cross_edition_fallbacks(rows: dict[tuple[str, str], dict[str, str]]) -> int:
    """Fill 2025 gaps only where one unambiguous 1991 tune matches.

    This is explicitly secondary evidence because the 2025 revision changed
    the keys of some songs.  The official change notice is retained in the
    row provenance so those records remain easy to audit.
    """
    def normalize(value: str) -> str:
        value = re.sub(r"\([^)]*\)", "", value.lower())
        return re.sub(r"[^a-z0-9]+", "", value)

    by_tune: dict[str, list[dict[str, str]]] = {}
    for (book_id, _), row in rows.items():
        if book_id == "sh1991" and row.get("tune"):
            by_tune.setdefault(normalize(row["tune"]), []).append(row)
    filled = 0
    for (book_id, _), row in rows.items():
        if book_id != "sh2025":
            continue
        candidates = by_tune.get(normalize(row.get("tune", "")), [])
        if len(candidates) != 1:
            continue
        source = candidates[0]
        changed = False
        for field in ("key_signature", "time_signature"):
            if not row.get(field) and source.get(field):
                row[field] = source[field]
                changed = True
        if changed:
            row["confidence"] = "secondary"
            row["source_url"] = "; ".join(part for part in (row["source_url"], source.get("source_url", ""), SH2025_KEY_CHANGES_URL) if part and part not in row["source_url"])
            row["notes"] = f"{row['notes']}; cross-edition fallback from 1991; verify against 2025 score and revision key changes"
            filled += 1
    return filled


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("rag_web_metadata.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/rag_metadata"))
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    rows: dict[tuple[str, str], dict[str, str]] = {}
    errors = 0

    fasola_items: list[tuple[str, str, str]] = []
    for book_id, index_url in FASOLA_INDEXES.items():
        edition = "1991" if book_id == "sh1991" else "2025"
        pages = parse_fasola_index(fetch(index_url, args.cache_dir), edition)
        fasola_items.extend((book_id, song_no, url) for song_no, url in pages.items())
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(add_fasola, rows, item, args.cache_dir) for item in fasola_items]
        for future in as_completed(futures):
            if future.result() == "error":
                errors += 1

    cooper_items = parse_cooper_index(fetch(urljoin(COOPER_BASE, "page.html"), args.cache_dir))
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(add_cooper, rows, item, args.cache_dir) for item in cooper_items]
        for future in as_completed(futures):
            if future.result() == "error":
                errors += 1
    try:
        cooper_times = parse_time_signature_index(fetch(COOPER_TIME_URL, args.cache_dir))
        for song_no, signature in cooper_times.items():
            row = rows.get(("shcooper2012", song_no))
            if row:
                row["time_signature"] = signature
                row["source_url"] = f"{row['source_url']}; {COOPER_TIME_URL}" if COOPER_TIME_URL not in row["source_url"] else row["source_url"]
                row["notes"] = f"{row['notes']}; Cooper time-signature index"
    except Exception as error:
        errors += 1
        print(f"Cooper time-signature index failed: {error}")

    catalog = parse_music_catalog(fetch(MUSIC_URL, args.cache_dir))
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(add_musicxml, rows, record, args.cache_dir) for record in catalog]
        for future in as_completed(futures):
            if future.result() == "error":
                errors += 1

    try:
        added_from_ed = add_ed_dataset(rows, fetch(ED_DATA_URL, args.cache_dir))
    except Exception as error:
        added_from_ed = 0
        errors += 1
        print(f"Ed dataset fallback failed: {error}")
    cross_edition_filled = add_cross_edition_fallbacks(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for key in sorted(rows):
            row = rows[key]
            row["confidence"] = row.get("confidence") or "source"
            writer.writerow({field: row.get(field, "") for field in FIELDS})
    print(f"catalog records: {len(catalog)}")
    print(f"Fasola pages: {len(fasola_items)}")
    print(f"Cooper pages: {len(cooper_items)}")
    print(f"metadata rows: {len(rows)}")
    print(f"Ed fallback rows added: {added_from_ed}")
    print(f"2025 cross-edition fallbacks filled: {cross_edition_filled}")
    print(f"errors: {errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
