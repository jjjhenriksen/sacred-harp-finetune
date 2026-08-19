#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlparse

try:
    import cloudscraper  # type: ignore
except Exception:  # pragma: no cover
    cloudscraper = None

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sh_corpus.shape_note_normalization import norm_first_line


TAG_ROOT = "music/shape-note"
TAG_TEXT = f"{TAG_ROOT}/text"
TAG_SONG = f"{TAG_ROOT}/song"
SH_BOOK_IDS = {"sh1991", "sh2025"}
DISPLAY_BOOK_IDS = {
    "sh1991",
    "sh2025",
    "ch7",
    "shenandoah",
    "shcooper2012",
    "southernharmony",
    "kentucky",
    "cooper",
    "mnharmony",
    "sacredharptunes",
    "trumpet",
}
TEXT_HUB_QUERY_BLOCK = """<!--TEXT_HUB_QUERY_V1-->
## Appearances (by book)

```dataviewjs
const pages = dv.pages(`[[${dv.current().file.name}]]`)
  .where(p => !p.file.tags.includes("music/shape-note/text"));

function bookFromTags(tags) {
  const books = ["sh1991","sh2025","sh","ch7","shenandoah","southernharmony","kentucky","cooper","mnharmony","sacredharptunes","trumpet"];
  for (const b of books) {
    if (tags.includes(`music/shape-note/${b}`)) return b;
  }
  return "—";
}

function parseSongNo(song_no) {
  const m = String(song_no ?? "").match(/^(\\d+)([a-z]?)$/i);
  if (!m) return {num: 9999, suffix: 9};
  const num = parseInt(m[1], 10);
  const suf = (m[2] || "").toLowerCase();
  const order = {"": 0, "b": 1, "t": 2};
  return {num, suffix: order[suf] ?? 9};
}

const rows = pages.map(p => {
  const s = parseSongNo(p.song_no);
  return {
    book: bookFromTags(p.file.tags),
    no: p.song_no ?? "",
    title: p.title ?? p.file.name,
    link: p.file.link,
    sortA: s.num,
    sortB: s.suffix
  };
});

const grouped = rows.groupBy(r => r.book);

for (const g of grouped) {
  dv.header(3, g.key);
  const sorted = g.rows.sort((a,b) =>
    a.sortA === b.sortA ? a.sortB - b.sortB : a.sortA - b.sortA
  );
  dv.table(
    ["No", "Title", "Song"],
    sorted.map(r => [r.no, r.title, r.link])
  );
}
```"""

BOOK_ORDER = {
    "sh": 0,
    "sh1991": 1,
    "sh2025": 2,
    "ch7": 3,
    "shenandoah": 4,
    "shcooper2012": 5,
    "southernharmony": 6,
    "cooper": 7,
    "kentucky": 8,
    "mnharmony": 9,
    "sacredharptunes": 10,
    "trumpet": 11,
}

BOOK_FAMILY_MAP = {
    "sh1991": "sh",
    "sh2025": "sh",
    "ch7": "ch7",
    "shenandoah": "shenandoah",
    "shcooper2012": "cooper",
    "southernharmony": "southernharmony",
    "cooper": "cooper",
    "kentucky": "kentucky",
    "mnharmony": "mnharmony",
    "sacredharptunes": "sacredharptunes",
    "trumpet": "trumpet",
}

# Sacred Harp union merges only these display-source families.
SH_UNION_SOURCE_FAMILIES = {"sh", "cooper"}
SH_UNION_MERGE_FAMILY = "sh-union"
DISPLAY_TAG_FAMILY_MAP = {
    "sh-union": "sh",
}
TUNE_TITLE_ALIAS_MAP = {
    # Cooper/SH spelling drift that should not block SH Union display collapse.
    "ecstacy": "ecstasy",
}


@dataclass
class Row:
    book_id: str
    book_family: str
    song_no: str
    title: str
    text_key: str
    url: str
    first_line_raw: str


@dataclass
class SongNoteSpec:
    source_family: str
    song_no: str
    title: str
    text_key: str
    books: List[str]
    urls: List[str]
    alt_titles: List[str]
    source_title_index: str
    titles_by_book: Dict[str, str]
    first_line_raw: str
    tune_merge_title: str = ""
    text_keys_by_book: Dict[str, str] | None = None
    source_note_ids: List[str] | None = None
    display_family: str = ""
    merge_family: str = ""
    source_families: List[str] | None = None

    def __post_init__(self) -> None:
        if not self.display_family:
            self.display_family = self.source_family
        if not self.merge_family:
            self.merge_family = self.source_family
        if self.source_families is None:
            self.source_families = [self.source_family]
        if self.source_note_ids is None:
            self.source_note_ids = [make_source_note_id(self.source_family, self.song_no, self.title, self.text_key)]


def strip_outer_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == "'") or (s[0] == s[-1] == '"')):
        return s[1:-1].strip()
    return s


def read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_many_csv_rows(paths: List[Path]) -> List[dict]:
    rows: List[dict] = []
    for path in paths:
        if not path.exists():
            continue
        rows.extend(read_csv_rows(path))
    return rows


def split_lyrics_into_stanzas(lyrics_text: str) -> List[str]:
    text = (lyrics_text or "").strip()
    if not text:
        return []
    blocks = re.split(r"\n\s*\n+", text)
    stanzas: List[str] = []
    for block in blocks:
        lines = [re.sub(r"\s+", " ", line).strip() for line in block.splitlines()]
        lines = [line for line in lines if line]
        if not lines:
            continue
        stanzas.append("\n".join(lines))
    if stanzas:
        return stanzas
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return ["\n".join(lines)] if lines else []


def read_yaml_rows(path: Path) -> List[dict]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        rows = yaml.safe_load(f) or []
    if not isinstance(rows, list):
        raise ValueError(f"Expected list in YAML file: {path}")
    return rows


def stanza_order_sort_key(lyrics_text: str, stanza_text: str, ordinal: int) -> tuple[int, int]:
    if not lyrics_text:
        return (10**9, ordinal)
    idx = lyrics_text.find(stanza_text)
    if idx >= 0:
        return (idx, ordinal)

    # Fall back to whitespace-normalized matching if punctuation/spacing drift
    # keeps the raw stanza string from appearing verbatim in the witness text.
    norm_lyrics = re.sub(r"\s+", " ", lyrics_text.strip())
    norm_stanza = re.sub(r"\s+", " ", stanza_text.strip())
    idx = norm_lyrics.find(norm_stanza)
    if idx >= 0:
        return (idx, ordinal)
    return (10**9, ordinal)


def slugify_text_key(text_key: str) -> str:
    s = unicodedata.normalize("NFKD", text_key or "")
    s = s.lower().strip()
    s = s.replace("’", "").replace("'", "")
    s = re.sub(r"[^a-z0-9\s-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace(" ", "-")
    s = re.sub(r"-+", "-", s)
    return (s[:120] if len(s) > 120 else s) or "untitled"


def song_no_sort_key(song_no: str) -> Tuple[int, int, str]:
    sn = (song_no or "").strip().lower()
    m = re.match(r"^(\d+)([a-z]?)$", sn)
    if not m:
        return (10**9, 99, sn)
    n = int(m.group(1))
    suf = m.group(2) or ""
    if suf == "":
        rank = 0
    elif suf == "t":
        rank = 1
    else:
        rank = 2 + (ord(suf) - ord("a"))
    return (n, rank, sn)


def safe_title_for_filename(title: str) -> str:
    s = unicodedata.normalize("NFKC", title or "").strip()
    s = s.replace("[", "(").replace("]", ")")
    s = re.sub(r"[\\/:*?\"<>|]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or "Untitled"


def yaml_scalar(s: str) -> str:
    t = (s or "").replace("\r", " ").replace("\n", " ").strip().replace('"', '\\"')
    return f'"{t}"'


def to_wikilink(vault: Path, md_path: Path, label: str | None = None, use_stem: bool = False) -> str:
    target = md_path.stem if use_stem else md_path.relative_to(vault).with_suffix("").as_posix()
    if label:
        return f"[[{target}|{label}]]"
    return f"[[{target}]]"


def write_if_changed(path: Path, content: str, dry_run: bool) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return True


def prune_stale_generated_files(expected_paths: List[Path], roots: List[Path], dry_run: bool) -> int:
    expected = {p.resolve() for p in expected_paths}
    stale: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.md"):
            if path.resolve() not in expected:
                stale.append(path)
    if not dry_run:
        for path in stale:
            path.unlink()
    return len(stale)


def normalize_title_for_compare(s: str) -> str:
    t = unicodedata.normalize("NFKD", s or "").lower().strip()
    t = t.replace("’", "'")
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_tune_title_for_merge(s: str) -> str:
    """
    Conservative tune normalization for SH Union display merges.

    This handles case/spacing/punctuation drift such as "North Port" vs
    "Northport" without collapsing distinct tune names that differ by words.
    """
    t = unicodedata.normalize("NFKD", s or "").lower()
    t = t.replace("’", "").replace("'", "")
    t = re.sub(r"[^a-z0-9]", "", t)
    t = t.strip()
    return TUNE_TITLE_ALIAS_MAP.get(t, t)


def normalize_text_id_for_merge(text_key: str) -> str:
    return norm_first_line(text_key or "")


def make_source_note_id(source_family: str, song_no: str, title: str, text_key: str) -> str:
    return "|".join([
        source_family,
        song_no,
        normalize_tune_title_for_merge(title),
        normalize_text_id_for_merge(text_key),
    ])


def compute_display_merge_key(spec: SongNoteSpec) -> Tuple[str, str, str, str]:
    tradition_family = spec.source_family
    if spec.source_family in SH_UNION_SOURCE_FAMILIES:
        tradition_family = SH_UNION_MERGE_FAMILY
    tune_key = normalize_tune_title_for_merge(spec.tune_merge_title or spec.title)
    return (
        tradition_family,
        spec.song_no,
        tune_key,
        normalize_text_id_for_merge(spec.text_key),
    )


def should_collapse_to_sh_union(specs: List[SongNoteSpec]) -> bool:
    return len({s.source_family for s in specs} & SH_UNION_SOURCE_FAMILIES) >= 2


def choose_display_title(specs: List[SongNoteSpec]) -> str:
    sh_specs = [s for s in specs if s.source_family == "sh" and s.title]
    if sh_specs:
        return sorted(sh_specs, key=lambda s: (-len(s.title), s.title.lower()))[0].title
    title_counts = Counter([s.title for s in specs if s.title])
    if title_counts:
        max_count = max(title_counts.values())
        candidates = [t for t, n in title_counts.items() if n == max_count]
        return sorted(candidates, key=lambda t: (-len(t), t.lower()))[0]
    return specs[0].song_no


def choose_display_text_key(specs: List[SongNoteSpec]) -> str:
    sh_specs = [s for s in specs if s.source_family == "sh" and s.text_key]
    if sh_specs:
        return sorted(sh_specs, key=lambda s: (-len(s.text_key), s.text_key))[0].text_key
    text_counts = Counter([s.text_key for s in specs if s.text_key])
    if text_counts:
        max_count = max(text_counts.values())
        candidates = [t for t, n in text_counts.items() if n == max_count]
        return sorted(candidates, key=lambda t: (-len(t), t))[0]
    return ""


def collapse_display_song_specs(source_specs: List[SongNoteSpec]) -> List[SongNoteSpec]:
    by_merge_key: Dict[Tuple[str, str, str, str], List[SongNoteSpec]] = defaultdict(list)
    for spec in source_specs:
        by_merge_key[compute_display_merge_key(spec)].append(spec)

    collapsed: List[SongNoteSpec] = []
    for merge_key in sorted(by_merge_key.keys()):
        specs = by_merge_key[merge_key]
        if merge_key[0] != SH_UNION_MERGE_FAMILY or not should_collapse_to_sh_union(specs):
            collapsed.extend(specs)
            continue

        # This is a display-layer collapse only. Raw book/edition provenance
        # stays attached through books, titles_by_book, text_keys_by_book, and
        # source_note_ids.
        title = choose_display_title(specs)
        text_key = choose_display_text_key(specs)
        books = sort_books([b for spec in specs for b in spec.books])
        urls = sorted({u for spec in specs for u in spec.urls})
        alt_titles = sorted(
            {t for spec in specs for t in ([spec.title] + spec.alt_titles) if t and t != title},
            key=lambda t: t.lower(),
        )
        titles_by_book: Dict[str, str] = {}
        for spec in specs:
            for bid, raw_title in spec.titles_by_book.items():
                titles_by_book.setdefault(bid, raw_title)
        text_keys_by_book: Dict[str, str] = {}
        for spec in specs:
            if spec.text_keys_by_book:
                for bid, tk in spec.text_keys_by_book.items():
                    text_keys_by_book.setdefault(bid, tk)
            else:
                for bid in spec.books:
                    text_keys_by_book.setdefault(bid, spec.text_key)
        first_line_candidates = [spec.first_line_raw for spec in specs if spec.first_line_raw]
        first_line_raw = sorted(first_line_candidates, key=lambda s: (-len(s), s.lower()))[0] if first_line_candidates else ""
        source_families = sorted({spec.source_family for spec in specs}, key=lambda fam: (BOOK_ORDER.get(fam, 99), fam))
        source_note_ids = sorted({nid for spec in specs for nid in (spec.source_note_ids or [])})

        collapsed.append(SongNoteSpec(
            source_family="sh",
            display_family="sh",
            merge_family=SH_UNION_MERGE_FAMILY,
            source_families=source_families,
            song_no=specs[0].song_no,
            title=title,
            text_key=text_key,
            books=books,
            urls=urls,
            alt_titles=alt_titles,
            source_title_index=title,
            titles_by_book=titles_by_book,
            first_line_raw=first_line_raw,
            text_keys_by_book=text_keys_by_book,
            source_note_ids=source_note_ids,
        ))

    return sorted(collapsed, key=lambda s: (BOOK_ORDER.get(s.display_family, 99), s.display_family, s.song_no, s.title.lower()))


def token_prefix_match(a: str, b: str) -> bool:
    a_tokens = (a or "").split()
    b_tokens = (b or "").split()
    if len(a_tokens) < 4:
        return False
    if len(b_tokens) < len(a_tokens):
        return False
    if (len(b_tokens) - len(a_tokens)) > 6:
        return False
    return b_tokens[: len(a_tokens)] == a_tokens


def detect_truncation_difference(keys: List[str]) -> tuple[bool, str, str]:
    """
    Returns (matched, shorter_key, longer_key) using normalized keys.
    """
    normed = sorted({norm_first_line(k) for k in keys if (k or "").strip()}, key=lambda x: (len(x.split()), len(x), x))
    for i in range(len(normed)):
        for j in range(i + 1, len(normed)):
            a, b = normed[i], normed[j]
            if token_prefix_match(a, b):
                return True, a, b
    return False, "", ""


def is_non_song_row(raw: dict) -> bool:
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

    m = re.search(r"(\d{4})", book_id)
    book_year = m.group(1) if m else ""

    if is_root and book_year and song_no == book_year:
        return True

    return False


def choose_group_title(group: List[Row]) -> str:
    sh2025 = [g.title for g in group if g.book_id == "sh2025" and g.title]
    if sh2025:
        return sorted(sh2025, key=lambda t: (-len(t), t.lower()))[0]
    sh1991 = [g.title for g in group if g.book_id == "sh1991" and g.title]
    if sh1991:
        return sorted(sh1991, key=lambda t: (-len(t), t.lower()))[0]
    title_counts = Counter([g.title for g in group if g.title])
    if title_counts:
        max_count = max(title_counts.values())
        candidates = [t for t, n in title_counts.items() if n == max_count]
        return sorted(candidates, key=lambda t: (-len(t), t.lower()))[0]
    return group[0].song_no


def title_mismatch_is_extreme(titles: List[str]) -> bool:
    vals = [normalize_title_for_compare(t) for t in titles if t.strip()]
    vals = sorted(set(v for v in vals if v))
    if len(vals) <= 1:
        return False
    if len(vals) >= 3:
        return True

    a, b = vals[0], vals[1]
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return True
    jaccard = len(sa & sb) / len(sa | sb)
    if jaccard < 0.2 and a not in b and b not in a:
        return True
    return False


def clean_page_title(raw_title: str, song_no: str) -> str:
    t = (raw_title or "").strip()
    t = re.sub(r"\s+", " ", t)
    # Remove leading song number if present in headings.
    t = re.sub(rf"^{re.escape(song_no)}\s*[-–—:]?\s+", "", t, flags=re.IGNORECASE)
    for sep in [" - ", " | "]:
        parts = t.split(sep)
        if len(parts) >= 2:
            tail = sep.join(parts[1:]).lower()
            if "sacred harp" in tail or "bremen" in tail or "shenandoah" in tail or "christian harmony" in tail:
                t = parts[0].strip()
                break
    return t.strip()


class TitleResolver:
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.cache: Dict[str, str] = {}
        self.changed = False
        self.disabled = False
        if cache_path.exists():
            try:
                loaded = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.cache = {str(k): str(v) for k, v in loaded.items()}
            except Exception:
                self.cache = {}

        self.session = None
        if cloudscraper is not None:
            try:
                self.session = cloudscraper.create_scraper()
                self.session.headers.update({"User-Agent": "shape-note-title-resolver/1.0"})
            except Exception:
                self.session = None

    def _is_network_fatal(self, msg: str) -> bool:
        m = msg.lower()
        return (
            "name or service not known" in m
            or "temporary failure in name resolution" in m
            or "failed to resolve" in m
            or "nodename nor servname" in m
            or "network is unreachable" in m
        )

    def _fetch_html(self, url: str) -> str:
        if self.session is not None:
            r = self.session.get(url, timeout=10)
            r.raise_for_status()
            return r.text
        raise RuntimeError("cloudscraper unavailable")

    def _extract_title(self, html: str, song_no: str) -> str | None:
        if BeautifulSoup is None:
            return None
        soup = BeautifulSoup(html, "html.parser")
        selectors = [
            "article h1.entry-title",
            "h1.entry-title",
            "article h1",
            "main h1",
            "h1",
        ]
        for sel in selectors:
            node = soup.select_one(sel)
            if node:
                t = clean_page_title(node.get_text(" ", strip=True), song_no)
                if t:
                    return t

        if soup.title:
            t = clean_page_title(soup.title.get_text(" ", strip=True), song_no)
            if t:
                return t
        return None

    def resolve(self, url: str, song_no: str) -> str | None:
        key = (url or "").strip()
        if not key:
            return None
        if key in self.cache:
            return self.cache[key] or None
        if self.disabled:
            self.cache[key] = ""
            self.changed = True
            return None

        try:
            html = self._fetch_html(key)
            title = self._extract_title(html, song_no) or ""
            self.cache[key] = title
            self.changed = True
            return title or None
        except Exception as e:
            msg = str(e)
            if self._is_network_fatal(msg):
                self.disabled = True
            self.cache[key] = ""
            self.changed = True
            return None

    def save(self) -> None:
        if not self.changed:
            return
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def preferred_group_url(group: List[Row]) -> str:
    for bid in ("sh2025", "sh1991"):
        for r in group:
            if r.book_id == bid and r.url:
                return r.url
    urls = sorted({r.url for r in group if r.url})
    return urls[0] if urls else ""


def load_same_number_tunecomparison_map(path: Path) -> Dict[Tuple[str, str], str]:
    """
    Load only same-number Cooper/Denson equivalences from Texas Fasola.

    We deliberately ignore cross-number mappings. The comparison index is used
    only to stabilize tune identity when SH and Cooper already share the same
    song number.
    """
    if not path.exists():
        return {}
    out: Dict[Tuple[str, str], str] = {}
    rows = read_csv_rows(path)
    for row in rows:
        title = (row.get("title") or "").strip()
        cooper_no = (row.get("cooper_no") or "").strip().lower()
        denson_no = (row.get("denson_no") or "").strip().lower()
        if not title or not cooper_no or not denson_no:
            continue
        if cooper_no != denson_no:
            continue
        out[("cooper", cooper_no)] = title
        out[("sh", denson_no)] = title
    return out


def build_song_frontmatter(
    book_family: str,
    song_no: str,
    title_canonical: str,
    text_key: str,
    books: List[str],
    urls: List[str],
    alt_titles: List[str],
    source_title_index: str,
    titles_by_book: Dict[str, str],
    display_family: str | None = None,
    merge_family: str | None = None,
    source_families: List[str] | None = None,
    source_note_ids: List[str] | None = None,
    text_keys_by_book: Dict[str, str] | None = None,
) -> str:
    # Keep per-book edition tags only for split SH notes; all other notes are family-tagged.
    include_book_tags = book_family == "sh" and len(books) == 1
    display_family = display_family or book_family
    merge_family = merge_family or display_family
    source_families = source_families or [book_family]
    tag_family = DISPLAY_TAG_FAMILY_MAP.get(merge_family, DISPLAY_TAG_FAMILY_MAP.get(display_family, display_family))
    lines = [
        "---",
        "type: song",
        f"book_family: {yaml_scalar(book_family)}",
        f"display_family: {yaml_scalar(display_family)}",
        f"merge_family: {yaml_scalar(merge_family)}",
        f"song_no: {yaml_scalar(song_no)}",
        f"title: {yaml_scalar(title_canonical)}",
        f"text_key: {yaml_scalar(text_key)}",
        f"source_title_index: {yaml_scalar(source_title_index)}",
        "source_families:",
    ]
    for fam in source_families:
        lines.append(f"  - {fam}")
    lines.extend([
        "books:",
    ])
    for b in books:
        lines.append(f"  - {b}")
    if source_note_ids:
        lines.append("source_note_ids:")
        for note_id in source_note_ids:
            lines.append(f"  - {yaml_scalar(note_id)}")
    if titles_by_book:
        lines.append("titles_by_book:")
        for b in sorted(titles_by_book.keys(), key=lambda x: (BOOK_ORDER.get(x, 99), x)):
            lines.append(f"  {b}: {yaml_scalar(titles_by_book[b])}")
    if text_keys_by_book:
        lines.append("text_keys_by_book:")
        for b in sorted(text_keys_by_book.keys(), key=lambda x: (BOOK_ORDER.get(x, 99), x)):
            lines.append(f"  {b}: {yaml_scalar(text_keys_by_book[b])}")
    if alt_titles:
        lines.append("alt_titles:")
        for t in alt_titles:
            lines.append(f"  - {yaml_scalar(t)}")
    if urls:
        lines.append("urls:")
        for u in urls:
            lines.append(f"  - {yaml_scalar(u)}")
    lines.append("tags:")
    lines.append(f"  - {TAG_SONG}")
    lines.append(f"  - {TAG_ROOT}/{tag_family}")
    if include_book_tags:
        for b in books:
            lines.append(f"  - {TAG_ROOT}/{b}")
    lines.append("---")
    return "\n".join(lines)


def build_song_body(song_no: str, title: str, text_hub_link: str | None, first_line_raw: str, urls: List[str]) -> str:
    out = [
        f"# {song_no} — {title}",
        "",
    ]
    if text_hub_link:
        out.append(f"- Text Hub: {text_hub_link}")
    else:
        out.append("- Text Hub: not yet available")
    if first_line_raw:
        out.append(f"- Raw First Line: {first_line_raw}")
    if urls:
        out.append("- Sources:")
        for u in urls:
            out.append(f"  - {u}")
    out.append("")
    return "\n".join(out)


def build_text_frontmatter(text_key: str) -> str:
    return "\n".join([
        "---",
        "type: text",
        f"text_key: {yaml_scalar(text_key)}",
        "tags:",
        f"  - {TAG_TEXT}",
        "---",
    ])


def format_song_ref(song_ref: str) -> str:
    if ":" not in song_ref:
        return song_ref
    book_id, song_no = song_ref.split(":", 1)
    if book_id not in DISPLAY_BOOK_IDS:
        return song_ref
    display_book = BOOK_FAMILY_MAP.get(book_id, book_id)
    if book_id in SH_BOOK_IDS:
        display_book = book_id
    return f"{display_book} {song_no}"


def collect_full_texts(
    members: List[Row],
    stanza_rows: List[dict],
    lyrics_by_song_ref: Dict[str, str] | None = None,
) -> List[tuple[List[str], str]]:
    stanza_rows_by_song: Dict[str, List[dict]] = defaultdict(list)
    for ordinal, row in enumerate(stanza_rows):
        row = dict(row)
        row["_ordinal"] = ordinal
        for song_ref in row.get("associated_songs", []) or []:
            stanza_rows_by_song[str(song_ref)].append(row)

    song_refs = sorted(
        {
            f"{member.book_id}:{member.song_no}"
            for member in members
            if member.book_id in DISPLAY_BOOK_IDS and member.song_no
        },
        key=lambda ref: (BOOK_ORDER.get(ref.split(":", 1)[0], 99), song_no_sort_key(ref.split(":", 1)[1])),
    )

    witness_stanzas: List[tuple[str, List[str]]] = []
    stanza_to_witnesses: Dict[str, List[str]] = defaultdict(list)

    for song_ref in song_refs:
        song_stanza_rows = stanza_rows_by_song.get(song_ref, [])
        lyrics_text = ""
        if lyrics_by_song_ref:
            lyrics_text = lyrics_by_song_ref.get(song_ref, "")
        song_stanza_rows = sorted(
            song_stanza_rows,
            key=lambda row: stanza_order_sort_key(
                lyrics_text,
                (row.get("text") or "").strip(),
                int(row.get("_ordinal") or 0),
            ),
        )
        stanza_texts = [
            (row.get("text") or "").strip()
            for row in song_stanza_rows
            if (row.get("text") or "").strip()
        ]
        if not stanza_texts and lyrics_text:
            stanza_texts = split_lyrics_into_stanzas(lyrics_text)
        if not stanza_texts:
            continue
        witness_stanzas.append((song_ref, stanza_texts))
        for stanza in stanza_texts:
            if song_ref not in stanza_to_witnesses[stanza]:
                stanza_to_witnesses[stanza].append(song_ref)

    if not witness_stanzas:
        return []

    ordered_pairs: List[tuple[List[str], str]] = []
    emitted_stanzas: set[str] = set()

    # The first witness defines the reading backbone for the text hub.
    _, base_stanzas = witness_stanzas[0]
    for stanza in base_stanzas:
        if stanza in emitted_stanzas:
            continue
        ordered_pairs.append((stanza_to_witnesses[stanza], stanza))
        emitted_stanzas.add(stanza)

    # Later witnesses contribute only genuinely new stanzas, in their own order.
    for _, stanzas in witness_stanzas[1:]:
        for stanza in stanzas:
            if stanza in emitted_stanzas:
                continue
            ordered_pairs.append((stanza_to_witnesses[stanza], stanza))
            emitted_stanzas.add(stanza)

    merged_runs: List[tuple[List[str], List[str]]] = []
    current_song_refs: List[str] | None = None
    current_stanzas: List[str] = []
    for song_refs, stanza in ordered_pairs:
        if current_song_refs == song_refs:
            current_stanzas.append(stanza)
            continue
        if current_song_refs is not None:
            merged_runs.append((current_song_refs, current_stanzas))
        current_song_refs = song_refs
        current_stanzas = [stanza]

    if current_song_refs is not None:
        merged_runs.append((current_song_refs, current_stanzas))

    return [(song_refs, "\n\n".join(stanzas)) for song_refs, stanzas in merged_runs]


def build_text_body(
    display_first_line: str,
    text_key: str,
    song_links: List[str],
    full_texts: List[tuple[List[str], str]] | None = None,
) -> str:
    out = [
        f"# {display_first_line}",
        "",
        "## Canonical Key",
        text_key,
        "",
        "## Songs",
    ]
    for lnk in song_links:
        out.append(f"- {lnk}")
    if full_texts:
        out.extend(["", "## Full Texts"])
        if len(full_texts) > 1:
            out.extend([
                "",
                "_Exact duplicate verses are shown once, preserving the first witness as the base order._",
            ])
        for song_refs, full_text in full_texts:
            out.append("")
            out.append(f"### {', '.join(format_song_ref(song_ref) for song_ref in song_refs)}")
            out.append("")
            out.append(full_text)
    out.append("")
    out.append(TEXT_HUB_QUERY_BLOCK)
    out.append("")
    return "\n".join(out)


def sort_books(books: List[str]) -> List[str]:
    return sorted(set(books), key=lambda b: (BOOK_ORDER.get(b, 99), b))


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(prog="sh_text_hubs.py")
    ap.add_argument("--vault", required=True, help="Path to Obsidian vault")
    ap.add_argument("--first-lines", required=True, help="Canonical first-lines CSV path")
    ap.add_argument("--generated-root", default="04 Music/shape-note", help="Generated root folder inside vault")
    ap.add_argument("--report", default="sh_text_hub_report.csv", help="Output report CSV")
    ap.add_argument("--review", default="needs_review_groups.csv", help="Groups needing review CSV")
    ap.add_argument("--changed-across-editions", default="changed_across_editions.csv", help="SH split-cases CSV")
    ap.add_argument("--title-cache", default="title_cache.json", help="Title cache JSON path")
    ap.add_argument("--tune-comparison", default="texasfasola_tunecomparison.csv", help="Texas Fasola tune comparison CSV")
    ap.add_argument("--stanzas", default="data/stanzas.yaml", help="Canonical stanza YAML for full-text sections")
    ap.add_argument("--lyrics-csv", default="bremen_lyrics_fixed4.csv", help="Lyrics CSV used to recover stanza order within witnesses")
    ap.add_argument("--extra-lyrics-csv", action="append", default=[], help="Additional lyrics CSV(s) to merge for full-text witness rendering")
    ap.add_argument("--dry-run", action="store_true", help="Do not write files")
    args = ap.parse_args()

    vault = Path(strip_outer_quotes(args.vault)).expanduser().resolve()
    first_lines_csv = Path(strip_outer_quotes(args.first_lines)).expanduser().resolve()
    generated_root = vault / Path(args.generated_root)
    songs_dir = generated_root / "songs"
    texts_dir = generated_root / "texts"
    report_path = Path(strip_outer_quotes(args.report)).expanduser().resolve()
    review_path = Path(strip_outer_quotes(args.review)).expanduser().resolve()
    changed_path = Path(strip_outer_quotes(args.changed_across_editions)).expanduser().resolve()
    title_cache_path = Path(strip_outer_quotes(args.title_cache)).expanduser().resolve()
    tune_comparison_path = Path(strip_outer_quotes(args.tune_comparison)).expanduser().resolve()
    stanzas_path = Path(strip_outer_quotes(args.stanzas)).expanduser().resolve()
    lyrics_csv_path = Path(strip_outer_quotes(args.lyrics_csv)).expanduser().resolve()
    extra_lyrics_csv_paths = [
        Path(strip_outer_quotes(path)).expanduser().resolve()
        for path in (args.extra_lyrics_csv or [])
    ]

    if not vault.exists():
        raise SystemExit(f"Vault not found: {vault}")
    if not first_lines_csv.exists():
        raise SystemExit(f"CSV not found: {first_lines_csv}")
    if not stanzas_path.exists():
        raise SystemExit(f"Stanza YAML not found: {stanzas_path}")
    if not lyrics_csv_path.exists():
        raise SystemExit(f"Lyrics CSV not found: {lyrics_csv_path}")

    rows_raw = read_csv_rows(first_lines_csv)
    stanza_rows = read_yaml_rows(stanzas_path)
    lyrics_rows_raw = read_many_csv_rows([lyrics_csv_path, *extra_lyrics_csv_paths])
    lyrics_by_song_ref: Dict[str, str] = {}
    for raw in lyrics_rows_raw:
        book_id = (raw.get("book_id") or "").strip()
        song_no = (raw.get("song_no") or "").strip().lower()
        lyrics_text = (raw.get("lyrics") or "").strip()
        if not book_id or not song_no or not lyrics_text:
            continue
        lyrics_by_song_ref[f"{book_id}:{song_no}"] = lyrics_text

    filtered_non_song_rows = 0
    rows: List[Row] = []
    for raw in rows_raw:
        if is_non_song_row(raw):
            filtered_non_song_rows += 1
            continue
        song_no = (raw.get("song_no") or "").strip().lower()
        if not song_no:
            continue
        text_key = (raw.get("text_key") or "").strip()
        book_id = (raw.get("book_id") or "").strip()
        rows.append(Row(
            book_id=book_id,
            book_family=BOOK_FAMILY_MAP.get(book_id, book_id),
            song_no=song_no,
            title=(raw.get("title") or "").strip(),
            text_key=text_key,
            url=(raw.get("url") or "").strip(),
            first_line_raw=(raw.get("first_line_raw") or raw.get("first_line") or "").strip(),
        ))

    by_song_key: Dict[tuple[str, str], List[Row]] = defaultdict(list)
    by_text_key: Dict[str, List[Row]] = defaultdict(list)
    for r in rows:
        by_song_key[(r.book_family, r.song_no)].append(r)
        if r.text_key:
            by_text_key[r.text_key].append(r)

    text_keys_sorted = sorted(by_text_key.keys())
    slug_to_keys: Dict[str, List[str]] = defaultdict(list)
    for tk in text_keys_sorted:
        slug_to_keys[slugify_text_key(tk)].append(tk)

    text_path_by_key: Dict[str, Path] = {}
    for slug, keys in sorted(slug_to_keys.items(), key=lambda t: t[0]):
        ks = sorted(keys)
        if len(ks) == 1:
            text_path_by_key[ks[0]] = texts_dir / f"text--{slug}.md"
        else:
            for i, tk in enumerate(ks, 1):
                text_path_by_key[tk] = texts_dir / f"text--{slug}-{i}.md"

    resolver = TitleResolver(title_cache_path)
    same_number_tune_map = load_same_number_tunecomparison_map(tune_comparison_path)

    report_rows: List[dict] = []
    review_rows: List[dict] = []
    changed_rows: List[dict] = []
    song_note_path_by_key: Dict[tuple[str, str], Path] = {}
    song_title_by_key: Dict[tuple[str, str], str] = {}
    song_notes_by_text_key: Dict[str, List[tuple[Path, str, tuple[int, str, str]]]] = defaultdict(list)
    source_song_specs: List[SongNoteSpec] = []
    family_counts: Counter[str] = Counter()
    title_diff_examples: List[dict] = []
    suspect_groups: List[dict] = []
    skipped_groups = 0
    merged_notes_count = 0
    split_notes_count = 0
    total_song_notes_count = 0
    number_of_song_no_with_multiple_text_key = 0
    normalization_difference_count = 0
    truncation_difference_count = 0
    true_text_change_count = 0
    truncation_merge_examples: List[dict] = []
    written_paths: List[Path] = []
    expected_generated_paths: List[Path] = []

    for song_key in sorted(by_song_key.keys(), key=lambda t: (BOOK_ORDER.get(t[0], 99), t[0], t[1])):
        book_family, song_no = song_key
        group = by_song_key[song_key]

        text_keys = sorted({g.text_key for g in group if g.text_key})
        titles = sorted({g.title for g in group if g.title})
        force_merge_change_type = ""
        forced_text_key = ""
        forced_text_keys_by_book: Dict[str, str] = {}

        is_normalization_difference = False
        if len(text_keys) > 1 and book_family == "sh":
            # SH editions diverged by raw text_key; determine if this is only normalization drift.
            truncation_match, _short_key, long_key = detect_truncation_difference(text_keys)
            if truncation_match:
                force_merge_change_type = "truncation_difference"
                truncation_difference_count += 1
                forced_text_keys_by_book = {}
                for bid in sorted({g.book_id for g in group if g.book_id}, key=lambda b: (BOOK_ORDER.get(b, 99), b)):
                    bid_keys = sorted({g.text_key for g in group if g.book_id == bid and g.text_key}, key=lambda s: (-len(s), s))
                    if bid_keys:
                        forced_text_keys_by_book[bid] = bid_keys[0]
                if "sh2025" in forced_text_keys_by_book and forced_text_keys_by_book["sh2025"]:
                    forced_text_key = forced_text_keys_by_book["sh2025"]
                elif long_key:
                    # choose the raw key corresponding to the longer normalized key
                    candidates = sorted({g.text_key for g in group if norm_first_line(g.text_key) == long_key}, key=lambda s: (-len(s), s))
                    forced_text_key = candidates[0] if candidates else long_key
                else:
                    forced_text_key = sorted(text_keys, key=lambda s: (-len(s), s))[0]

                if len(truncation_merge_examples) < 5:
                    truncation_merge_examples.append({
                        "song_no": song_no,
                        "sh1991": forced_text_keys_by_book.get("sh1991", ""),
                        "sh2025": forced_text_keys_by_book.get("sh2025", ""),
                    })

            normalized_keys = sorted({norm_first_line(k) for k in text_keys if k})
            min_ratio = 1.0
            if len(normalized_keys) > 1:
                for i in range(len(normalized_keys)):
                    for j in range(i + 1, len(normalized_keys)):
                        ratio = difflib.SequenceMatcher(None, normalized_keys[i], normalized_keys[j]).ratio()
                        if ratio < min_ratio:
                            min_ratio = ratio
            is_normalization_difference = len(normalized_keys) > 1 and min_ratio >= 0.90

            number_of_song_no_with_multiple_text_key += 1
            if force_merge_change_type:
                change_type = force_merge_change_type
            elif is_normalization_difference:
                change_type = "normalization_difference"
                normalization_difference_count += 1
            else:
                change_type = "text_changed"
                true_text_change_count += 1
            for g in sorted(group, key=lambda r: (r.book_id, r.title, r.url, r.text_key)):
                changed_rows.append({
                    "song_no": song_no,
                    "book_id": g.book_id,
                    "title": g.title,
                    "text_key": g.text_key,
                    "url": g.url,
                    "change_type": change_type,
                })

            if is_normalization_difference or force_merge_change_type:
                # Treat as merge: continue using canonical merged-note flow below.
                pass
            else:
                # SH editions diverged by text: split into per-(book_id, song_no) notes.
                by_book: Dict[str, List[Row]] = defaultdict(list)
                for g in group:
                    by_book[g.book_id].append(g)

                for book_id in sorted(by_book.keys(), key=lambda b: (BOOK_ORDER.get(b, 99), b)):
                    sub = by_book[book_id]
                    sub_text_counts = Counter([x.text_key for x in sub if x.text_key])
                    text_key = sorted(sub_text_counts.items(), key=lambda t: (-t[1], t[0]))[0][0] if sub_text_counts else ""
                    source_title_index = choose_group_title(sub)
                    title_canonical = source_title_index
                    alt_titles = sorted({x.title for x in sub if x.title and x.title != title_canonical}, key=lambda t: t.lower())
                    urls = sorted({x.url for x in sub if x.url})
                    first_line_raws = [x.first_line_raw for x in sub if x.first_line_raw]
                    first_line_raw = sorted(first_line_raws, key=lambda s: (-len(s), s.lower()))[0] if first_line_raws else ""
                    books = [book_id]
                    titles_by_book = {book_id: source_title_index} if source_title_index else {}
                    source_song_specs.append(SongNoteSpec(
                        source_family=book_family,
                        song_no=song_no,
                        title=title_canonical,
                        text_key=text_key,
                        books=books,
                        urls=urls,
                        alt_titles=alt_titles,
                        source_title_index=source_title_index,
                        titles_by_book=titles_by_book,
                        first_line_raw=first_line_raw,
                        tune_merge_title=same_number_tune_map.get((book_family, song_no), ""),
                    ))
                    split_notes_count += 1
                continue


        if len(text_keys) > 1 and not is_normalization_difference and not force_merge_change_type:
            skipped_groups += 1
            for g in sorted(group, key=lambda r: (r.book_id, r.title, r.url)):
                review_rows.append({
                    "group_key": f"{book_family}:{song_no}",
                    "reason": "text_key_mismatch",
                    "book_id": g.book_id,
                    "title": g.title,
                    "url": g.url,
                    "text_key": g.text_key,
                })
            continue

        if len(titles) > 1:
            suspect_groups.append({
                "group_key": f"{book_family}:{song_no}",
                "rows": sorted(group, key=lambda r: (r.book_id, r.title, r.url)),
            })
            if title_mismatch_is_extreme(titles):
                for g in sorted(group, key=lambda r: (r.book_id, r.title, r.url)):
                    review_rows.append({
                        "group_key": f"{book_family}:{song_no}",
                        "reason": "title_mismatch_extreme",
                        "book_id": g.book_id,
                        "title": g.title,
                        "url": g.url,
                        "text_key": g.text_key,
                    })

        text_key = forced_text_key or (text_keys[0] if text_keys else "")
        books = sort_books([g.book_id for g in group if g.book_id])
        urls = sorted({g.url for g in group if g.url})
        first_line_raws = [g.first_line_raw for g in group if g.first_line_raw]
        first_line_raw = sorted(first_line_raws, key=lambda s: (-len(s), s.lower()))[0] if first_line_raws else ""

        source_title_index = choose_group_title(group)
        title_canonical = source_title_index
        titles_by_book: Dict[str, str] = {}
        for bid in sorted({g.book_id for g in group if g.book_id}, key=lambda b: (BOOK_ORDER.get(b, 99), b)):
            bid_titles = [g.title for g in group if g.book_id == bid and g.title]
            if bid_titles:
                titles_by_book[bid] = sorted(bid_titles, key=lambda t: (-len(t), t.lower()))[0]
        alt_titles = sorted({g.title for g in group if g.title and g.title != title_canonical}, key=lambda t: t.lower())

        if title_canonical != source_title_index and len(title_diff_examples) < 5:
            title_diff_examples.append({
                "group_key": f"{book_family}:{song_no}",
                "title_canonical": title_canonical,
                "source_title_index": source_title_index,
                "url": "",
            })

        source_song_specs.append(SongNoteSpec(
            source_family=book_family,
            song_no=song_no,
            title=title_canonical,
            text_key=text_key,
            books=books,
            urls=urls,
            alt_titles=alt_titles,
            source_title_index=source_title_index,
            titles_by_book=titles_by_book,
            first_line_raw=first_line_raw,
            tune_merge_title=same_number_tune_map.get((book_family, song_no), ""),
            text_keys_by_book=forced_text_keys_by_book if force_merge_change_type == "truncation_difference" else None,
        ))
        merged_notes_count += 1

    display_song_specs = collapse_display_song_specs(source_song_specs)
    total_song_notes_count = len(display_song_specs)

    for spec in display_song_specs:
        note_prefix = spec.display_family
        if spec.source_family == "sh" and len(spec.books) == 1 and spec.books[0] in {"sh1991", "sh2025"} and spec.merge_family != SH_UNION_MERGE_FAMILY:
            fname = f"{note_prefix} {spec.song_no} ({spec.books[0]}) — {safe_title_for_filename(spec.title)}.md"
            label = f"{note_prefix} {spec.song_no} ({spec.books[0]}) — {spec.title}"
            action = "song_note_generate_split"
            sort_song_no = f"{spec.song_no} ({spec.books[0]})"
        else:
            fname = f"{note_prefix} {spec.song_no} — {safe_title_for_filename(spec.title)}.md"
            label = f"{note_prefix} {spec.song_no} — {spec.title}"
            action = "song_note_generate_merged"
            sort_song_no = spec.song_no

        song_path = songs_dir / spec.display_family / fname
        expected_generated_paths.append(song_path)
        family_counts[spec.display_family] += 1

        text_hub_path = text_path_by_key.get(spec.text_key) if spec.text_key else None
        text_hub_link = to_wikilink(vault, text_hub_path, use_stem=True) if text_hub_path else None

        frontmatter = build_song_frontmatter(
            book_family=spec.source_family,
            display_family=spec.display_family,
            merge_family=spec.merge_family,
            source_families=spec.source_families,
            source_note_ids=spec.source_note_ids,
            song_no=spec.song_no,
            title_canonical=spec.title,
            text_key=spec.text_key,
            books=spec.books,
            urls=spec.urls,
            alt_titles=spec.alt_titles,
            source_title_index=spec.source_title_index,
            titles_by_book=spec.titles_by_book,
            text_keys_by_book=spec.text_keys_by_book,
        )
        body = build_song_body(spec.song_no, spec.title, text_hub_link, spec.first_line_raw, spec.urls)
        content = frontmatter + "\n" + body

        changed = write_if_changed(song_path, content, dry_run=args.dry_run)
        if changed and not args.dry_run:
            written_paths.append(song_path)

        report_rows.append({
            "action": action,
            "book_family": spec.source_family,
            "display_family": spec.display_family,
            "merge_family": spec.merge_family,
            "song_no": spec.song_no,
            "title": spec.title,
            "source_title_index": spec.source_title_index,
            "books": ",".join(spec.books),
            "source_families": ",".join(spec.source_families or []),
            "text_key": spec.text_key,
            "song_path": str(song_path),
            "changed": str(bool(changed)),
        })
        if spec.text_key:
            song_notes_by_text_key[spec.text_key].append(
                (song_path, label, (BOOK_ORDER.get(spec.display_family, 99), spec.display_family, sort_song_no))
            )

    for text_key in text_keys_sorted:
        members = by_text_key[text_key]
        text_path = text_path_by_key[text_key]

        raw_candidates = [m.first_line_raw for m in members if m.first_line_raw]
        display_first = sorted(raw_candidates, key=lambda s: (-len(s), s.lower()))[0] if raw_candidates else text_key

        song_links: List[str] = []
        for p, label, _sortkey in sorted(song_notes_by_text_key.get(text_key, []), key=lambda t: t[2]):
            # Song links in text hubs should remain stable after syncing out of the
            # generated workspace, so link by note stem instead of the staged path.
            song_links.append(to_wikilink(vault, p, label=label, use_stem=True))

        # Skip detached text hubs that have no associated song notes in the
        # generated corpus. These create zero-inbound orphan notes in the live
        # vault and do not help navigation.
        if not song_links:
            report_rows.append({
                "action": "text_hub_skip_orphan",
                "text_key": text_key,
                "text_path": str(text_path),
                "song_count": "0",
                "changed": "False",
            })
            continue

        full_texts = collect_full_texts(members, stanza_rows, lyrics_by_song_ref)
        content = build_text_frontmatter(text_key) + "\n" + build_text_body(
            display_first,
            text_key,
            song_links,
            full_texts=full_texts,
        )
        changed = write_if_changed(text_path, content, dry_run=args.dry_run)
        expected_generated_paths.append(text_path)
        if changed and not args.dry_run:
            written_paths.append(text_path)

        report_rows.append({
            "action": "text_hub_generate",
            "text_key": text_key,
            "text_path": str(text_path),
            "song_count": str(len(song_links)),
            "changed": str(bool(changed)),
        })

    if not args.dry_run:
        stale_deleted = prune_stale_generated_files(expected_generated_paths, [songs_dir, texts_dir], dry_run=False)
        resolver.save()
    else:
        stale_deleted = 0

    write_csv(report_path, report_rows)
    write_csv(review_path, review_rows)
    write_csv(changed_path, changed_rows)

    wrote_outside_generated = any(generated_root not in p.parents and p != generated_root for p in written_paths)

    print(f"Vault: {vault}")
    print(f"Generated root: {generated_root}")
    print(f"Song notes: {total_song_notes_count}")
    print(f"Text hubs: {len(text_keys_sorted)}")
    print("Song notes by family:")
    for fam, n in sorted(family_counts.items(), key=lambda t: (BOOK_ORDER.get(t[0], 99), t[0])):
        print(f"  {fam}: {n}")
    print(f"Filtered non-song rows: {filtered_non_song_rows}")
    print(f"Skipped groups (written to review): {skipped_groups}")
    print(f"merged_notes_count: {merged_notes_count}")
    print(f"split_notes_count: {split_notes_count}")
    print(f"normalization_difference_count: {normalization_difference_count}")
    print(f"truncation_difference_count: {truncation_difference_count}")
    print(f"true_text_change_count: {true_text_change_count}")
    print(f"number_of_song_no_with_multiple_text_key: {number_of_song_no_with_multiple_text_key}")
    print(f"Review CSV: {review_path}")
    print(f"Changed-across-editions CSV: {changed_path}")
    print(f"Report CSV: {report_path}")
    print(f"Stale generated files deleted: {stale_deleted}")
    print(f"Writes outside generated root: {'YES' if wrote_outside_generated else 'NO'}")

    print("\nCanonical title differs from index title (up to 5):")
    if title_diff_examples:
        for ex in title_diff_examples:
            print(f"- {ex['group_key']}: index='{ex['source_title_index']}' canonical='{ex['title_canonical']}' ({ex['url']})")
    else:
        print("- none")

    print("\nFirst 20 suspect groups (title variants):")
    if suspect_groups:
        for s in suspect_groups[:20]:
            print(f"- {s['group_key']}")
            for r in s["rows"]:
                print(f"  ({r.book_id}, {r.title}, {r.url}, {r.text_key})")
    else:
        print("- none")

    print("\nTruncation merges (up to 5):")
    if truncation_merge_examples:
        for ex in truncation_merge_examples:
            print(f"- {ex['song_no']}: sh1991='{ex['sh1991']}' | sh2025='{ex['sh2025']}'")
    else:
        print("- none")


if __name__ == "__main__":
    main()
