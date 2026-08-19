from __future__ import annotations

import re
import unicodedata

from ..shape_note_normalization import norm_first_line

TUNEBOOKS = [
    ("https://sh1991.sacredharpbremen.org/index-of-first-lines/", "sh1991"),
    ("https://sacredharpbremen.org/index-of-first-lines/", "sh2025"),
    ("https://shenandoah.harmony.sacredharpbremen.org/index-of-first-lines/", "shenandoah"),
    ("https://sevenshapes.sacredharpbremen.org/index-of-first-lines/", "ch7"),
]

SONGS_INDEX_URLS = {
    "shenandoah": "https://shenandoah.harmony.sacredharpbremen.org/songs/",
}

CACHE_CSV = "bremen_first_lines.csv"
LYRICS_CSV = "bremen_lyrics.csv"
OVERRIDES_CSV = "lyrics_overrides.csv"

LETTER_RE = re.compile(r"^[A-Z]$")
ENTRY_RE_PAREN = re.compile(r"^(?P<first>.+?)\s*\((?P<num>\d+[a-z]?)\s+(?P<title>.+?)\)\s*$", re.IGNORECASE)
ENTRY_RE_DASH = re.compile(r"^(?P<first>.+?)\s*[—-]\s*(?P<num>\d+[a-z]?)\s+(?P<title>.+?)\s*$", re.IGNORECASE)
ENTRY_RE_DASH_TAIL = re.compile(r"^(?:—|–|-)\s*(?P<num>\d+[a-z]?)\s+(?P<title>.+?)\s*$", re.IGNORECASE)

FIRST_LINE_PREFIX_ALIASES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^\s*(?:verse|v|stanza)\s*\d+\s*[:.\-]\s*", re.IGNORECASE), ""),
    (re.compile(r"^\s*\d+\s*[:.\-]\s*", re.IGNORECASE), ""),
    (re.compile(r"^\s*(?:chorus|refrain|repeat)\s*[:.\-]\s*", re.IGNORECASE), ""),
]

BOOK_SORT = {
    "sh": 0,
    "sh1991": 0,
    "sh2025": 1,
    "shenandoah": 2,
    "ch7": 3,
    "southernharmony": 4,
    "kentucky": 5,
}

BOOKS_DEFAULT = ["sh1991", "sh2025", "shcooper2012", "ch7", "shenandoah", "southernharmony", "kentucky"]
BACKBONE_BOOKS = ["sh", "shcooper2012", "ch7", "shenandoah"]


def resolve_first_line_prefix_alias(first_line: str) -> str:
    out = (first_line or "").strip()
    if not out:
        return ""
    changed = True
    while changed and out:
        changed = False
        for pattern, repl in FIRST_LINE_PREFIX_ALIASES:
            new_out = pattern.sub(repl, out, count=1).strip()
            if new_out != out:
                out = new_out
                changed = True
                break
    return out


def song_no_sort_key(song_no: str):
    m = re.match(r"^(?P<n>\d+)(?P<s>[a-z]?)$", song_no.strip().lower())
    if not m:
        return (10**9, song_no)
    base = int(m.group("n"))
    suffix = m.group("s")
    if suffix == "":
        suffix_rank = 0
    elif suffix == "t":
        suffix_rank = 1
    else:
        suffix_rank = 2 + (ord(suffix) - ord("a"))
    return (base, suffix_rank)


def norm_title(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def merge_sh_editions(entries: list[dict]) -> list[dict]:
    sh91 = {}
    sh25 = {}
    other = []
    for r in entries:
        bid = r.get("book_id", "")
        key = (r.get("song_no", ""), norm_title(r.get("title", "")))
        if bid == "sh1991":
            sh91[key] = r
        elif bid == "sh2025":
            sh25[key] = r
        else:
            other.append(r)

    merged = []
    for key, r91 in sh91.items():
        out = dict(r91)
        out["book_id"] = "sh"
        r25 = sh25.get(key)
        if r25:
            out["url_2025"] = r25.get("url", "")
        merged.append(out)

    for key, r25 in sh25.items():
        if key not in sh91:
            merged.append(r25)

    return merged + other
