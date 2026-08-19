#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from collections import Counter
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sh_corpus.bremen.scrape import canonicalize_cache_rows


OUTPUT_FIELDS = [
    "book_id",
    "song_no",
    "title",
    "first_line",
    "first_line_raw",
    "first_line_norm",
    "text_key",
    "url",
]


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def normalize_title(s: str) -> str:
    t = unicodedata.normalize("NFKD", s or "").lower().strip()
    t = t.replace("’", "'")
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def reconcile_sh1991_cooper_same_song_title(rows: list[dict]) -> int:
    """
    If SH1991 and Cooper 2012 share the same song number and normalized title,
    treat SH1991's text_key as canonical for that matched piece.
    """
    by_song_title: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        song_no = (row.get("song_no") or "").strip().lower()
        title_norm = normalize_title(row.get("title") or "")
        if song_no and title_norm:
            by_song_title[(song_no, title_norm)].append(row)

    updated = 0
    for _key, group in by_song_title.items():
        sh1991_keys = sorted({r.get("text_key", "").strip() for r in group if r.get("book_id") == "sh1991" and r.get("text_key", "").strip()})
        cooper_rows = [r for r in group if r.get("book_id") == "shcooper2012"]
        if len(sh1991_keys) != 1 or not cooper_rows:
            continue
        canonical_key = sh1991_keys[0]
        for row in cooper_rows:
            if row.get("text_key") != canonical_key:
                row["text_key"] = canonical_key
                updated += 1
    return updated


def main() -> int:
    ap = argparse.ArgumentParser(prog="build_combined_first_lines.py")
    ap.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "bremen_first_lines_canonicalized.csv",
            "cooper2012_first_lines_raw_texasfasola.csv",
            "modern_shape_note_first_lines.csv",
            "southernharmony_first_lines.csv",
            "kentucky_first_lines.csv",
        ],
        help="Input first-lines CSVs to combine and recanonicalize",
    )
    ap.add_argument(
        "--out",
        default="combined_first_lines_canonicalized.csv",
        help="Output combined canonical CSV",
    )
    ap.add_argument(
        "--max-short-len",
        type=int,
        default=30,
        help="Max short first-line length for prefix aliasing",
    )
    args = ap.parse_args()

    input_paths = [Path(p).expanduser().resolve() for p in args.inputs]
    out_path = Path(args.out).expanduser().resolve()

    rows: list[dict] = []
    source_counts: Counter[str] = Counter()
    for path in input_paths:
        if not path.exists():
            raise SystemExit(f"Input CSV not found: {path}")
        source_rows = read_csv_rows(path)
        rows.extend(source_rows)
        source_counts[str(path)] = len(source_rows)

    canonical_rows, alias_map = canonicalize_cache_rows(rows, max_short_len=args.max_short_len)

    reconciled = reconcile_sh1991_cooper_same_song_title(canonical_rows)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in canonical_rows:
            writer.writerow({k: row.get(k, "") for k in OUTPUT_FIELDS})

    print(f"[ok] wrote: {out_path}")
    print(f"[ok] rows: {len(canonical_rows)}")
    print(f"[ok] prefix_aliases: {len(alias_map)}")
    print(f"[ok] sh1991/cooper reconciled rows: {reconciled}")
    print("[ok] source rows:")
    for path in input_paths:
        print(f"- {path}: {source_counts[str(path)]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
