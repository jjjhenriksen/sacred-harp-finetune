from __future__ import annotations

import argparse

from ..paths import project_default
from .analysis import cmd_backbone, cmd_books, cmd_dups, cmd_missing, cmd_overlap, cmd_search, cmd_stats, cmd_unique
from .lyrics import build_lyrics_cache, repair_lyrics_cache
from .scrape import build_cache, load_cache


def register_subcommands(sub, *, defaults: dict[str, str] | None = None, handler=None):
    defaults = defaults or {}
    analysis_cache_default = defaults.get("analysis_cache", project_default("combined_cache"))
    build_cache_default = defaults.get("build_cache", project_default("bremen_cache"))
    lyrics_default = defaults.get("lyrics", project_default("bremen_lyrics"))
    lyrics_fixed_default = defaults.get("lyrics_fixed", project_default("bremen_lyrics_fixed"))
    overrides_default = defaults.get("overrides", project_default("bremen_overrides"))

    def add(name: str, help_text: str):
        parser = sub.add_parser(name, help=help_text)
        parser.set_defaults(cmd=name)
        if handler is not None:
            parser.set_defaults(func=handler)
        return parser

    b = add("build", "Scrape Bremen indexes and write cache CSV")
    b.add_argument("--out", default=build_cache_default, help="Cache CSV path")
    b.add_argument("--sleep", type=float, default=0.2, help="Delay between tunebooks (seconds)")
    b.add_argument("--max-short-len", type=int, default=30, help="Max short first-line length for prefix aliasing")

    s = add("search", "Search by first line across the combined corpus, including Cooper")
    s.add_argument("query")
    s.add_argument("--cache", default=analysis_cache_default, help="Cache CSV path")
    s.add_argument("--key", action="store_true", help="Exact normalized-key match")
    s.add_argument("--fuzzy", action="store_true", help="Fuzzy match instead of substring")
    s.add_argument("--threshold", type=int, default=88, help="Fuzzy threshold (0-100)")
    s.add_argument("--limit", type=int, default=25, help="Fuzzy result limit")

    d = add("dups", "List texts appearing in >= N places across the combined corpus")
    d.add_argument("--cache", default=analysis_cache_default, help="Cache CSV path")
    d.add_argument("--min", type=int, default=2, help="Minimum occurrences")

    bk = add("books", "List book_id counts in the combined corpus")
    bk.add_argument("--cache", default=analysis_cache_default, help="Cache CSV path")

    st = add("stats", "Show summary stats for the combined corpus")
    st.add_argument("--cache", default=analysis_cache_default, help="Cache CSV path")

    bb = add("backbone", "Texts present in sh + shcooper2012 + ch7 + shenandoah")
    bb.add_argument("--cache", default=analysis_cache_default, help="Cache CSV path")
    bb.add_argument("--limit", type=int, default=100)

    ov = add("overlap", "Texts present in all listed books within the combined corpus")
    ov.add_argument("books", nargs="+", help="book_ids, e.g. sh shcooper2012 ch7 shenandoah")
    ov.add_argument("--cache", default=analysis_cache_default, help="Cache CSV path")
    ov.add_argument("--limit", type=int, default=100)

    uq = add("unique", "Texts unique to one book_id within the default combined-book universe")
    uq.add_argument("book", help="book_id, e.g. sh2025 or shcooper2012")
    uq.add_argument("--cache", default=analysis_cache_default, help="Cache CSV path")
    uq.add_argument("--limit", type=int, default=100)

    ms = add("missing", "Present in some books, missing from others within the combined corpus")
    ms.add_argument("--in", dest="present_in", nargs="*", default=[], help="Require present in these book_ids")
    ms.add_argument("--not-in", dest="missing_from", nargs="*", default=[], help="Require absent from these book_ids")
    ms.add_argument("--cache", default=analysis_cache_default, help="Cache CSV path")
    ms.add_argument("--limit", type=int, default=100)

    ly = add("lyrics", "Scrape full lyric text for each song URL in cache")
    ly.add_argument("--cache", default=build_cache_default, help="Cache CSV path (bremen_first_lines.csv)")
    ly.add_argument("--out", default=lyrics_default, help="Output CSV path (bremen_lyrics.csv)")
    ly.add_argument("--sleep", type=float, default=0.2, help="Delay between song pages (seconds)")
    ly.add_argument("--overrides", default=overrides_default, help="Overrides CSV path")

    rp = add("repair-lyrics", "Re-scrape ONLY junk/failed lyrics rows from an existing lyrics CSV")
    rp.add_argument("--in", dest="in_csv", default=lyrics_default, help="Input lyrics CSV")
    rp.add_argument("--out", dest="out_csv", default=lyrics_fixed_default, help="Output lyrics CSV")
    rp.add_argument("--sleep", type=float, default=0.2, help="Delay between song pages (seconds)")
    rp.add_argument("--overrides", default=overrides_default, help="Overrides CSV path")
    rp.add_argument("--limit", type=int, default=None, help="Max number of repairs (debug)")
    rp.add_argument("--cache", default=build_cache_default, help="Cache CSV path (for first_line anchors)")


def dispatch(args) -> None:
    if args.cmd == "build":
        build_cache(args.out, sleep_s=args.sleep, max_short_len=args.max_short_len)
        return
    if args.cmd == "lyrics":
        build_lyrics_cache(cache_csv=args.cache, out_csv=args.out, overrides_csv=args.overrides, sleep_s=args.sleep)
        return
    if args.cmd == "repair-lyrics":
        repair_lyrics_cache(in_csv=args.in_csv, out_csv=args.out_csv, overrides_csv=args.overrides, sleep_s=args.sleep, limit=args.limit, cache_csv=args.cache)
        return
    rows = load_cache(getattr(args, "cache"))
    if args.cmd == "search":
        cmd_search(rows, args.query, by_key=args.key, fuzzy=args.fuzzy, threshold=args.threshold, limit=args.limit)
    elif args.cmd == "dups":
        cmd_dups(rows, args.min)
    elif args.cmd == "books":
        cmd_books(rows)
    elif args.cmd == "stats":
        cmd_stats(rows)
    elif args.cmd == "backbone":
        cmd_backbone(rows, limit=args.limit)
    elif args.cmd == "overlap":
        cmd_overlap(rows, books=args.books, limit=args.limit)
    elif args.cmd == "unique":
        cmd_unique(rows, book=args.book, limit=args.limit)
    elif args.cmd == "missing":
        cmd_missing(rows, present_in=args.present_in, missing_from=args.missing_from, limit=args.limit)


def build_parser(prog: str = "bremen_crosswalk.py", *, defaults: dict[str, str] | None = None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog)
    sub = ap.add_subparsers(dest="cmd", required=True)
    register_subcommands(sub, defaults=defaults)
    return ap


def main(argv: list[str] | None = None, *, prog: str = "bremen_crosswalk.py", defaults: dict[str, str] | None = None):
    ap = build_parser(prog=prog, defaults=defaults)
    dispatch(ap.parse_args(argv))
