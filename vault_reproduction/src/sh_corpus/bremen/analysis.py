from __future__ import annotations

from collections import Counter, defaultdict

from .common import BACKBONE_BOOKS, BOOKS_DEFAULT, BOOK_SORT, merge_sh_editions, norm_first_line, song_no_sort_key


def group_by_text_key(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["text_key"]].append(r)
    return grouped


def print_group(entries: list[dict], score: int | None = None):
    if not entries:
        return
    entries = merge_sh_editions(entries)
    first = entries[0].get("first_line", "").strip()
    print(f"\n== {first} =={f'  ({score}%)' if score is not None else ''}")
    entries.sort(key=lambda r: (BOOK_SORT.get(r.get("book_id", ""), 99), song_no_sort_key(r.get("song_no", "")), r.get("title", "").lower()))
    for r in entries:
        bid = r.get("book_id", "")
        label = "sh" if bid == "sh" else bid
        print(f"- {label:<11} {r.get('song_no', '')} {r.get('title', '')}  ({r.get('url', '')})")


def fuzzy_candidates(groups, query: str, limit: int = 25, threshold: int = 88):
    q = norm_first_line(query)
    try:
        from rapidfuzz import fuzz  # type: ignore

        scorer = fuzz.token_set_ratio
        scored = [(scorer(q, k), k, entries) for k, entries in groups.items() if scorer(q, k) >= threshold]
    except Exception:
        import difflib

        scored = []
        for k, entries in groups.items():
            score = int(100 * difflib.SequenceMatcher(None, q, k).ratio())
            if score >= threshold:
                scored.append((score, k, entries))
    scored.sort(key=lambda t: (-t[0], -len(t[2]), t[2][0]["first_line"].lower()))
    return scored[:limit]


def cmd_search(rows, query: str, by_key: bool, fuzzy: bool, threshold: int, limit: int):
    groups = group_by_text_key(rows)
    if by_key:
        entries = groups.get(norm_first_line(query), [])
        if not entries:
            print("No exact normalized-key match.")
            return
        print_group(entries)
        return
    if fuzzy:
        hits = fuzzy_candidates(groups, query, limit=limit, threshold=threshold)
        if not hits:
            print("No fuzzy matches.")
            return
        for score, _k, entries in hits:
            print_group(entries, score=score)
        return
    qn = norm_first_line(query)
    hits = [(k, entries) for k, entries in groups.items() if qn in k]
    if not hits:
        print("No matches.")
        return
    hits.sort(key=lambda t: (-len(t[1]), t[1][0]["first_line"].lower()))
    for _k, entries in hits[:50]:
        print_group(entries)
    if len(hits) > 50:
        print(f"\n(+ {len(hits) - 50} more matches; refine your query.)")


def cmd_dups(rows, min_occ: int):
    groups = group_by_text_key(rows)
    dups = [(k, v) for k, v in groups.items() if len(v) >= min_occ]
    dups.sort(key=lambda t: (-len(t[1]), t[1][0]["first_line"].lower()))
    for _k, entries in dups:
        print_group(entries)


def cmd_books(rows: list[dict]):
    counts = Counter(r.get("book_id", "") for r in rows)
    for bid, n in sorted(counts.items(), key=lambda t: (-t[1], t[0])):
        if bid:
            print(f"{bid:<10} {n}")


def _bookset(entries: list[dict]) -> set[str]:
    bset = {e.get("book_id", "") for e in entries if e.get("book_id")}
    if "sh1991" in bset or "sh2025" in bset:
        bset.add("sh")
    return bset


def cmd_stats(rows: list[dict]):
    book_counts = Counter(r.get("book_id", "") for r in rows if r.get("book_id"))
    unique_song_pairs = {(r.get("book_id", ""), r.get("song_no", "")) for r in rows if r.get("book_id") and r.get("song_no")}
    groups = group_by_text_key(rows)
    by_span = Counter(len(_bookset(entries)) for entries in groups.values() if entries)
    print(f"rows: {len(rows)}")
    print(f"unique songs: {len(unique_song_pairs)}")
    print(f"unique texts: {len(groups)}")
    print(f"books: {len(book_counts)}")
    print()
    print("book counts:")
    for bid, n in sorted(book_counts.items(), key=lambda t: (-t[1], t[0])):
        print(f"  {bid:<12} {n}")
    print()
    print("text span across books:")
    for span, n in sorted(by_span.items()):
        print(f"  {span} {'book' if span == 1 else 'books'}: {n}")


def _filter_groups(rows: list[dict], require: set[str] | None = None, forbid: set[str] | None = None) -> list[list[dict]]:
    require = require or set()
    forbid = forbid or set()
    out = []
    for entries in group_by_text_key(rows).values():
        if not entries:
            continue
        bset = _bookset(entries)
        if require and not require.issubset(bset):
            continue
        if forbid and (bset & forbid):
            continue
        out.append(entries)
    return out


def _print_groups(groups: list[list[dict]], limit: int = 100):
    groups = sorted(groups, key=lambda g: (-len(g), (g[0].get("first_line") or "").strip().lower()))
    for entries in groups[:limit]:
        print_group(entries)
    if len(groups) > limit:
        print(f"\n(+ {len(groups) - limit} more; use --limit to show more.)")


def cmd_overlap(rows: list[dict], books: list[str], limit: int):
    groups = _filter_groups(rows, require=set(books))
    print(f"Found {len(groups)} texts appearing in {', '.join(books)}.\n")
    _print_groups(groups, limit=limit)


def cmd_unique(rows: list[dict], book: str, limit: int):
    groups = _filter_groups(rows, require={book}, forbid=set(BOOKS_DEFAULT) - {book})
    print(f"Found {len(groups)} texts unique to {book} (within {BOOKS_DEFAULT}).\n")
    _print_groups(groups, limit=limit)


def cmd_missing(rows: list[dict], present_in: list[str], missing_from: list[str], limit: int):
    groups = _filter_groups(rows, require=set(present_in), forbid=set(missing_from))
    label_a = ",".join(present_in) if present_in else "(any)"
    label_b = ",".join(missing_from) if missing_from else "(none)"
    print(f"Found {len(groups)} texts present in {label_a} and missing from {label_b}.\n")
    _print_groups(groups, limit=limit)


def cmd_backbone(rows: list[dict], limit: int):
    print(f"Backbone books: {', '.join(BACKBONE_BOOKS)}.\n")
    cmd_overlap(rows, books=BACKBONE_BOOKS, limit=limit)
