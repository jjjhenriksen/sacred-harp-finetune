#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# ----------------------------
# Helpers
# ----------------------------
def read_csv(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def slugify(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = s.lower().strip()
    s = s.replace("’", "").replace("'", "")
    s = re.sub(r"[^a-z0-9\s-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace(" ", "-")
    s = re.sub(r"-+", "-", s)
    return s or "untitled"


def song_no_sort_key(song_no: str) -> Tuple[int, int, str]:
    sn = (song_no or "").strip().lower()
    m = re.match(r"^(\d+)([a-z]?)$", sn)
    if not m:
        return (10**9, 99, sn)
    n = int(m.group(1))
    suf = m.group(2) or ""
    if suf == "":
        r = 0
    elif suf == "t":
        r = 1
    else:
        r = 2 + (ord(suf) - ord("a"))
    return (n, r, sn)


def wikilink(rel_no_ext: str, label: str) -> str:
    rel_no_ext = rel_no_ext.replace("\\", "/")
    return f"[[{rel_no_ext}|{label}]]"


def dump_frontmatter(meta: Dict[str, object]) -> str:
    preferred = [
        "book_id",
        "song_no",
        "title",
        "url",
        "first_line",
        "first_lyric_line",
        "text_key",
        "sh1991_song_no",
        "sh1991_url",
        "sh2025_song_no",
        "sh2025_url",
        "lineage_links",
    ]
    keys: List[str] = []
    for k in preferred:
        if k in meta:
            keys.append(k)
    for k in sorted(meta.keys()):
        if k not in keys:
            keys.append(k)

    lines = ["---"]
    for k in keys:
        v = meta[k]
        if isinstance(v, list):
            lines.append(f"{k}:")
            for item in v:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{k}: {v}")
    lines.append("---")
    return "\n".join(lines) + "\n"


BOOK_ORDER = {"sh": 0, "shenandoah": 1, "ch7": 2}


def book_sort(bid: str) -> int:
    return BOOK_ORDER.get((bid or "").strip(), 99)


def canon_book_id(bid: str) -> str:
    """Merge sh1991/sh2025 into sh namespace."""
    b = (bid or "").strip()
    if b in ("sh1991", "sh2025"):
        return "sh"
    return b


# ----------------------------
# Data model
# ----------------------------
@dataclass(frozen=True)
class SongKey:
    book_id: str
    song_no: str  # normalized lowercase


@dataclass
class CacheRow:
    book_id: str
    song_no: str
    title: str
    first_line: str
    text_key: str
    url: str


@dataclass
class LyricsRow:
    book_id: str
    song_no: str
    title: str
    url: str
    first_lyric_line: str
    lyrics: str


@dataclass(frozen=True)
class NoteKey:
    """One Obsidian note."""
    book_id: str        # 'sh', 'shenandoah', 'ch7'
    text_key: str       # primary grouping key


# ----------------------------
# Main pipeline
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(prog="export_to_obsidian.py")
    ap.add_argument("--vault", default="obsidian_export", help="Obsidian vault export folder")
    ap.add_argument("--cache", default="bremen_first_lines.csv", help="Cache CSV (first lines)")
    ap.add_argument("--lyrics", required=True, help="Lyrics CSV (e.g., bremen_lyrics_fixed4.csv)")
    ap.add_argument("--tag", default="#music/the-sacred-harp", help="Tag line to include in every note body")
    ap.add_argument("--index-note", default="Text Lineage Index.md", help="Index note filename in vault root")
    ap.add_argument("--dry-run", action="store_true", help="Do not write files; just report counts")
    args = ap.parse_args()

    vault_dir = args.vault
    cache_csv = args.cache
    lyrics_csv = args.lyrics
    tag_line = args.tag.strip()
    index_note_path = os.path.join(vault_dir, args.index_note)

    # ---- Load cache
    cache_rows_raw = read_csv(cache_csv)
    cache_by_song: Dict[SongKey, CacheRow] = {}
    groups_by_text: Dict[str, List[SongKey]] = defaultdict(list)  # text_key -> song keys

    for r in cache_rows_raw:
        bid = (r.get("book_id") or "").strip()
        sn = (r.get("song_no") or "").strip().lower()
        tk = (r.get("text_key") or "").strip()
        if not bid or not sn or not tk:
            continue
        cr = CacheRow(
            book_id=bid,
            song_no=sn,
            title=(r.get("title") or "").strip(),
            first_line=(r.get("first_line") or "").strip(),
            text_key=tk,
            url=(r.get("url") or "").strip(),
        )
        sk = SongKey(bid, sn)
        cache_by_song[sk] = cr
        groups_by_text[tk].append(sk)

    # ---- Load lyrics
    lyrics_rows_raw = read_csv(lyrics_csv)
    lyrics_by_song: Dict[SongKey, LyricsRow] = {}

    for r in lyrics_rows_raw:
        bid = (r.get("book_id") or "").strip()
        sn = (r.get("song_no") or "").strip().lower()
        if not bid or not sn:
            continue
        lyrics_by_song[SongKey(bid, sn)] = LyricsRow(
            book_id=bid,
            song_no=sn,
            title=(r.get("title") or "").strip(),
            url=(r.get("url") or "").strip(),
            first_lyric_line=(r.get("first_lyric_line") or "").strip(),
            lyrics=(r.get("lyrics") or "").rstrip(),
        )

    # ---- Build note inventory from cache (preferred) + lyrics-only fallback
    # Primary: every cache text_key becomes a note (per canon book namespace).
    note_members: Dict[NoteKey, List[SongKey]] = defaultdict(list)

    for tk, song_keys in groups_by_text.items():
        # Split membership by canonical book namespace (sh merged)
        by_canon: Dict[str, List[SongKey]] = defaultdict(list)
        for sk in song_keys:
            cr = cache_by_song.get(sk)
            if not cr:
                continue
            by_canon[canon_book_id(cr.book_id)].append(sk)

        for canon_bid, members in by_canon.items():
            if canon_bid in ("sh", "shenandoah", "ch7"):
                note_members[NoteKey(canon_bid, tk)].extend(members)

    # Fallback: lyrics rows not in cache (rare) => create synthetic note per (canon book, song_no)
    # Only if they lack a text_key (since we can't lineage them reliably).
    for sk, lr in lyrics_by_song.items():
        if sk in cache_by_song:
            continue
        canon_bid = canon_book_id(lr.book_id)
        if canon_bid not in ("sh", "shenandoah", "ch7"):
            continue
        synthetic_tk = f"__lyrics_only__::{lr.book_id}::{lr.song_no}"
        nk = NoteKey(canon_bid, synthetic_tk)
        note_members[nk].append(sk)

    # ---- Determine stable filenames per note
    note_rel_by_note: Dict[NoteKey, str] = {}
    for nk, members in note_members.items():
        # pick a "best" representative for naming (prefer sh2025 over sh1991; else smallest song_no)
        def rep_rank(sk: SongKey) -> Tuple[int, Tuple[int, int, str]]:
            bid = sk.book_id
            pref = 0
            if bid == "sh2025":
                pref = 0
            elif bid == "sh1991":
                pref = 1
            else:
                pref = 2
            return (pref, song_no_sort_key(sk.song_no))

        rep = sorted(members, key=rep_rank)[0]
        rep_title = ""
        if rep in cache_by_song:
            rep_title = cache_by_song[rep].title
        if not rep_title and rep in lyrics_by_song:
            rep_title = lyrics_by_song[rep].title

        # "song_no" in filename for SH is from representative (usually sh2025 if present)
        fname = f"{rep.song_no}-{slugify(rep_title)}.md"
        rel = os.path.join(nk.book_id, fname)
        note_rel_by_note[nk] = rel

    # ---- Build lineage links across notes using cache groups
    # For each text_key, collect corresponding NoteKeys across books (sh merged)
    notes_by_textkey: Dict[str, List[NoteKey]] = defaultdict(list)
    for nk in note_members.keys():
        if nk.text_key.startswith("__lyrics_only__::"):
            continue
        notes_by_textkey[nk.text_key].append(nk)

    lineage_links_by_note: Dict[NoteKey, List[str]] = {}
    for tk, nks in notes_by_textkey.items():
        nks_sorted = sorted(
            nks,
            key=lambda nk: (book_sort(nk.book_id), os.path.splitext(note_rel_by_note[nk])[0].lower()),
        )
        links: List[str] = []
        for nk in nks_sorted:
            rel_no_ext = os.path.splitext(note_rel_by_note[nk])[0].replace("\\", "/")
            # label: book + a representative song_no + title
            members = note_members[nk]
            rep = sorted(members, key=lambda sk: song_no_sort_key(sk.song_no))[0]
            title = cache_by_song.get(rep).title if rep in cache_by_song else (lyrics_by_song.get(rep).title if rep in lyrics_by_song else "")
            label = f"{nk.book_id} {rep.song_no} {title}".strip()
            links.append(wikilink(rel_no_ext, label))
        for nk in nks_sorted:
            lineage_links_by_note[nk] = links

    # ---- Write index note
    def group_display_first_line(tk: str, members: List[SongKey]) -> str:
        lines = []
        for sk in members:
            cr = cache_by_song.get(sk)
            if cr and cr.first_line:
                lines.append(cr.first_line)
        if not lines:
            return "(unknown first line)"
        return sorted(lines, key=lambda s: (len(s), s.lower()))[0]

    sorted_index = sorted(
        notes_by_textkey.items(),
        key=lambda kv: (-len(kv[1]), kv[0].lower()),
    )

    index_lines: List[str] = [
        "# Text Lineage Index",
        "",
        f"- Groups (text_key): {len(notes_by_textkey)}",
        f"- Notes exported: {len(note_members)}",
        "",
    ]

    for tk, nks in sorted_index:
        # choose a title line from any cache member
        # prefer sh member if available
        all_song_members: List[SongKey] = []
        for nk in nks:
            all_song_members.extend(note_members[nk])
        title_line = group_display_first_line(tk, all_song_members)
        index_lines.append(f"## {title_line}")
        index_lines.append(f"- text_key: `{tk}`")
        for nk in sorted(nks, key=lambda nk: book_sort(nk.book_id)):
            rel_no_ext = os.path.splitext(note_rel_by_note[nk])[0].replace("\\", "/")
            # label: book + rep song_no + title
            members = note_members[nk]
            rep = sorted(members, key=lambda sk: song_no_sort_key(sk.song_no))[0]
            title = cache_by_song.get(rep).title if rep in cache_by_song else (lyrics_by_song.get(rep).title if rep in lyrics_by_song else "")
            label = f"{nk.book_id} {rep.song_no} {title}".strip()
            index_lines.append(f"- {wikilink(rel_no_ext, label)}")
        index_lines.append("")

    index_text = "\n".join(index_lines).rstrip() + "\n"

    # ---- Write notes
    writes = 0
    for nk, members in sorted(note_members.items(), key=lambda kv: (book_sort(kv[0].book_id), kv[0].text_key.lower())):
        # Choose representative for display header (prefer sh2025)
        def rep_rank(sk: SongKey) -> Tuple[int, Tuple[int, int, str]]:
            bid = sk.book_id
            pref = 2
            if bid == "sh2025":
                pref = 0
            elif bid == "sh1991":
                pref = 1
            return (pref, song_no_sort_key(sk.song_no))

        rep = sorted(members, key=rep_rank)[0]

        # Collect cache + lyrics for all members
        cache_members = [cache_by_song.get(sk) for sk in members if sk in cache_by_song]
        lyric_members = [lyrics_by_song.get(sk) for sk in members if sk in lyrics_by_song]

        # Title preference: rep cache title > rep lyrics title > any cache title > any lyrics title
        title = ""
        if rep in cache_by_song and cache_by_song[rep].title:
            title = cache_by_song[rep].title
        elif rep in lyrics_by_song and lyrics_by_song[rep].title:
            title = lyrics_by_song[rep].title
        if not title:
            for cr in cache_members:
                if cr and cr.title:
                    title = cr.title
                    break
        if not title:
            for lr in lyric_members:
                if lr and lr.title:
                    title = lr.title
                    break

        # First line: prefer cache from rep
        first_line = cache_by_song.get(rep).first_line if rep in cache_by_song else ""
        text_key = cache_by_song.get(rep).text_key if rep in cache_by_song else (nk.text_key if not nk.text_key.startswith("__lyrics_only__::") else "")

        # SH merge fields
        sh1991_song_no = ""
        sh1991_url = ""
        sh2025_song_no = ""
        sh2025_url = ""

        # Prefer SH URLs from lyrics rows if present (they tend to be the post-repair truth)
        for sk in members:
            if sk.book_id == "sh1991":
                sh1991_song_no = sk.song_no
                if sk in lyrics_by_song and lyrics_by_song[sk].url:
                    sh1991_url = lyrics_by_song[sk].url
                elif sk in cache_by_song and cache_by_song[sk].url:
                    sh1991_url = cache_by_song[sk].url
            elif sk.book_id == "sh2025":
                sh2025_song_no = sk.song_no
                if sk in lyrics_by_song and lyrics_by_song[sk].url:
                    sh2025_url = lyrics_by_song[sk].url
                elif sk in cache_by_song and cache_by_song[sk].url:
                    sh2025_url = cache_by_song[sk].url

        # Primary URL in frontmatter/body:
        # - For sh merged, prefer sh2025_url then sh1991_url
        # - Otherwise, rep url
        url = ""
        if nk.book_id == "sh":
            url = sh2025_url or sh1991_url
        else:
            if rep in lyrics_by_song and lyrics_by_song[rep].url:
                url = lyrics_by_song[rep].url
            elif rep in cache_by_song and cache_by_song[rep].url:
                url = cache_by_song[rep].url

        # Primary lyrics block:
        # - For sh merged, prefer sh2025 lyrics if non-empty, else sh1991
        first_lyric_line = ""
        lyrics_text = ""
        if nk.book_id == "sh":
            lr25 = None
            lr91 = None
            for sk in members:
                if sk.book_id == "sh2025":
                    lr25 = lyrics_by_song.get(sk)
                elif sk.book_id == "sh1991":
                    lr91 = lyrics_by_song.get(sk)

            if lr25 and (lr25.lyrics or "").strip():
                lyrics_text = lr25.lyrics.strip()
                first_lyric_line = lr25.first_lyric_line
            elif lr91 and (lr91.lyrics or "").strip():
                lyrics_text = lr91.lyrics.strip()
                first_lyric_line = lr91.first_lyric_line
        else:
            lr = lyrics_by_song.get(rep)
            if lr and (lr.lyrics or "").strip():
                lyrics_text = lr.lyrics.strip()
                first_lyric_line = lr.first_lyric_line

        lineage_links = lineage_links_by_note.get(nk, [])

        # song_no displayed in header/frontmatter:
        # - For sh merged: show preferred (sh2025 if present else sh1991)
        display_song_no = ""
        if nk.book_id == "sh":
            display_song_no = sh2025_song_no or sh1991_song_no or rep.song_no
        else:
            display_song_no = rep.song_no

        meta: Dict[str, object] = {
            "book_id": nk.book_id,
            "song_no": display_song_no,
            "title": title,
            "url": url,
            "first_line": (first_line or "").strip(),
            "first_lyric_line": (first_lyric_line or "").strip(),
            "text_key": (text_key or "").strip(),
            "lineage_links": lineage_links,
        }
        if nk.book_id == "sh":
            meta["sh1991_song_no"] = sh1991_song_no
            meta["sh1991_url"] = sh1991_url
            meta["sh2025_song_no"] = sh2025_song_no
            meta["sh2025_url"] = sh2025_url

        body: List[str] = []
        if tag_line:
            body.append(tag_line)
            body.append("")

        body.append(f"# {display_song_no} — {title}".strip())
        body.append("")
        if meta.get("first_line"):
            body.append(f"**Index first line:** {meta['first_line']}")
            body.append("")
        if url:
            body.append(f"**Source:** {url}")
            body.append("")

        if nk.book_id == "sh":
            body.append("## Editions")
            if sh1991_song_no or sh1991_url:
                body.append(f"- **sh1991:** {sh1991_song_no or '(no number)'}  {sh1991_url or ''}".rstrip())
            if sh2025_song_no or sh2025_url:
                body.append(f"- **sh2025:** {sh2025_song_no or '(no number)'}  {sh2025_url or ''}".rstrip())
            body.append("")

        if lineage_links:
            body.append("## Lineage")
            for lnk in lineage_links:
                body.append(f"- {lnk}")
            body.append("")

        body.append("## Lyrics")
        if lyrics_text.strip():
            body.append("```text")
            body.append(lyrics_text.strip())
            body.append("```")
        else:
            body.append("_Missing / not scraped._")

        note_text = dump_frontmatter(meta) + "\n".join(body).rstrip() + "\n"

        out_path = os.path.join(vault_dir, note_rel_by_note[nk])
        if not args.dry_run:
            write_text(out_path, note_text)
        writes += 1

    if not args.dry_run:
        write_text(index_note_path, index_text)

    print(f"[ok] vault: {vault_dir}")
    print(f"[ok] wrote notes: {writes}")
    print(f"[ok] wrote index: {index_note_path}")


if __name__ == "__main__":
    main()
