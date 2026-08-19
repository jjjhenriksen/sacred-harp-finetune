from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .paths import project_default, resolve_project_path
from .review import (
    DEFAULT_EXTRACTION_REVIEW_RESOLVED_PATH,
    DEFAULT_REVIEW_TRIAGE_PATH,
    apply_review_resolutions,
    generate_review_triage,
    load_review_items,
    load_review_resolutions,
    load_review_triage,
    review_item_id,
    upsert_review_resolution,
)
from .selector import select_stanza
from .storage import append_usage_entry, load_data_file, save_data_file

USER_AGENT = "sh-corpus-ingest/1.0 (+personal research)"


def _route_tag(tag: str, allowed: dict[str, list[str]]) -> tuple[str, str]:
    if tag in allowed["operational_fit"]:
        return "operational_fit", tag
    if tag in allowed["theme_tags"]:
        return "theme_tags", tag
    if tag in allowed["tone_tags"]:
        return "tone_tags", tag
    raise ValueError(f"Unknown tag: {tag}")


def _prompt_multiline() -> list[str]:
    print("Enter stanza blocks. Separate blocks with a blank line. End input with a single '.' on its own line.")
    lines: list[str] = []
    while True:
        line = input()
        if line == ".":
            break
        lines.append(line)
    text = "\n".join(lines).strip()
    if not text:
        return []
    return [block.strip() for block in text.split("\n\n") if block.strip()]


def _cache_path_for_basename(cache_dir: Path, basename: str) -> Path:
    return cache_dir / basename


def _fetch_url(url: str) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=8) as resp:
        return resp.read().decode("utf-8", "ignore")


def _cmd_extract(args: argparse.Namespace) -> int:
    from .extractor import extract_stanzas

    stanzas, review = extract_stanzas(
        args.lyrics_csv,
        args.stanzas_out,
        args.review_out,
        extra_lyrics_csv_paths=args.extra_lyrics_csv,
        cooper_first_lines_csv=args.cooper_first_lines_csv,
        cooper_cache_dir=args.cooper_cache_dir,
        text_equivalence_overrides_path=args.text_equivalence_overrides,
        progress=args.progress,
        progress_every=args.progress_every,
    )
    print(f"Extracted {len(stanzas)} stanzas.")
    print(f"Wrote {len(review)} extraction review entries.")
    return 0


def _cmd_classify(args: argparse.Namespace) -> int:
    from .classifier import classify_stanzas_file

    paths = classify_stanzas_file(
        stanzas_path=args.stanzas,
        rubric_path=args.rubric,
        batch_size=args.batch_size,
        apply=args.apply,
        review_dir=args.review_dir,
    )
    print(f"Wrote {len(paths)} classification review batches.")
    if args.apply:
        print(f"Applied classifications to {args.stanzas}.")
    return 0


def _cmd_select(args: argparse.Namespace) -> int:
    from .selector import select_stanza

    rubric = load_data_file(args.rubric, default={})
    allowed = rubric.get("allowed_tags", {})

    operational_fit = list(args.operational_fit)
    theme_tags = list(args.theme)
    tone_tags: list[str] = []

    for value in args.tone:
        bucket, remapped = _route_tag(value, allowed)
        if bucket == "operational_fit":
            operational_fit.append(remapped)
        elif bucket == "theme_tags":
            theme_tags.append(remapped)
        else:
            tone_tags.append(remapped)

    for value in args.tag:
        bucket, remapped = _route_tag(value, allowed)
        if bucket == "operational_fit":
            operational_fit.append(remapped)
        elif bucket == "theme_tags":
            theme_tags.append(remapped)
        else:
            tone_tags.append(remapped)

    selected = select_stanza(
        operational_fit=sorted(set(operational_fit)),
        theme_tags=sorted(set(theme_tags)),
        tone_tags=sorted(set(tone_tags)),
        stanzas_path=args.stanzas,
        usage_history_path=args.usage_history,
        rubric_path=args.rubric,
    )

    if args.record_usage:
        append_usage_entry(selected["id"], path=args.usage_history)

    print(json.dumps(selected, ensure_ascii=False, indent=2))
    return 0


def _cmd_triage_review(args: argparse.Namespace) -> int:
    _rows, summary = generate_review_triage(args.review, args.out)
    print(f"probable_junk={summary['probable_junk']}")
    print(f"probable_safe_irregular_lyric={summary['probable_safe_irregular_lyric']}")
    print(f"truly_ambiguous={summary['truly_ambiguous']}")
    print(f"out={args.out}")
    return 0


def _cmd_review_extraction(args: argparse.Namespace) -> int:
    items = load_review_items(args.review)
    resolved = {row["review_id"] for row in load_review_resolutions(args.resolved)}
    pending = [item for item in items if review_item_id(item) not in resolved]
    if args.bucket:
        triage_rows = load_review_triage(args.triage)
        bucket_ids = {row["review_id"] for row in triage_rows if row.get("bucket") == args.bucket}
        pending = [item for item in pending if review_item_id(item) in bucket_ids]

    if not pending:
        print("No unresolved extraction review items.")
        return 0

    for index, item in enumerate(pending, start=1):
        item_id = review_item_id(item)
        print("=" * 72)
        print(f"[{index}/{len(pending)}] {item['book_id']} {item['song_no']} - {item['title']}")
        print(f"Reason: {item.get('review_reason', '') or '(none)'}")
        print(f"Confidence: {item.get('segmentation_confidence', '')}")
        print(f"Review ID: {item_id}")
        if args.bucket:
            print(f"Bucket: {args.bucket}")
        print("\nRaw lyrics:\n")
        print(item.get("raw_lyrics", ""))
        print("\nProposed blocks:\n")
        for block_no, block in enumerate(item.get("proposed_blocks", []), start=1):
            print(f"[Block {block_no}]")
            print(block)
            print()

        while True:
            try:
                choice = input("[a] accept  [e] edit  [r] reject  [s] skip: ").strip().lower()
            except EOFError:
                print("\nInput closed. Exiting review session without recording a decision for this item.")
                return 0
            if choice not in {"a", "e", "r", "s"}:
                print("Please choose one of: a, e, r, s.")
                continue
            break

        resolution = {
            "review_id": item_id,
            "book_id": item["book_id"],
            "song_no": item["song_no"],
            "title": item["title"],
            "first_line": item.get("first_line", ""),
        }
        if choice == "a":
            resolution["decision"] = "accept"
        elif choice == "e":
            resolution["decision"] = "edit"
            resolution["resolved_blocks"] = _prompt_multiline()
        elif choice == "r":
            resolution["decision"] = "reject"
        else:
            resolution["decision"] = "skip"

        upsert_review_resolution(resolution, args.resolved)

    print(f"Saved review decisions to {args.resolved}.")
    return 0


def _cmd_apply_review(args: argparse.Namespace) -> int:
    stanzas, remaining = apply_review_resolutions(
        stanzas_path=args.stanzas,
        review_path=args.review,
        resolutions_path=args.resolved,
        text_equivalence_overrides_path=args.text_equivalence_overrides,
    )
    print(f"Updated stanzas: {len(stanzas)}")
    print(f"Remaining review items: {len(remaining)}")
    return 0


def _cmd_near_duplicate_report(args: argparse.Namespace) -> int:
    from .extractor import build_near_duplicate_report

    stanza_rows = load_data_file(args.stanzas, default=[])
    report = build_near_duplicate_report(stanza_rows, report_threshold=args.threshold)
    save_data_file(args.out, report)
    print(f"rows={len(stanza_rows)}")
    print(f"near_duplicates={len(report)}")
    print(f"out={args.out}")
    return 0


def _cmd_fetch_missing_cooper(args: argparse.Namespace) -> int:
    from .extractor import cooper_poetry_url_candidates

    review_rows = load_data_file(args.review, default=[])
    missing = [row for row in review_rows if row.get("review_reason") == "missing-cooper-poetry-page"]
    with Path(args.cooper_first_lines).open(encoding="utf-8") as handle:
        by_song_no = {row["song_no"]: row for row in csv.DictReader(handle)}
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    report_rows = []
    filled = 0
    unresolved = 0
    for review_row in missing:
        song_no = review_row["song_no"]
        source = by_song_no.get(song_no)
        if source is None:
            report_rows.append(
                {
                    "book_id": review_row["book_id"],
                    "song_no": song_no,
                    "title": review_row["title"],
                    "first_line": review_row.get("first_line", ""),
                    "status": "unresolved",
                    "failure_reason": "missing-from-cooper-first-lines-csv",
                    "resolved_url": "",
                }
            )
            unresolved += 1
            continue

        basename = source["url"].rstrip("/").split("/")[-1]
        cache_path = _cache_path_for_basename(cache_dir, basename)
        if cache_path.exists():
            report_rows.append(
                {
                    "book_id": source["book_id"],
                    "song_no": song_no,
                    "title": source["title"],
                    "first_line": source.get("first_line_raw", ""),
                    "status": "already_cached",
                    "failure_reason": "",
                    "resolved_url": source["url"],
                }
            )
            continue

        resolved_url = ""
        failure_reason = "not-fetched"
        for candidate in cooper_poetry_url_candidates(source["url"]):
            try:
                html = _fetch_url(candidate)
                cache_path.write_text(html, encoding="utf-8")
                resolved_url = candidate
                failure_reason = ""
                filled += 1
                print(f"filled {song_no} via {candidate}", flush=True)
                break
            except HTTPError as exc:
                failure_reason = f"http-{exc.code}"
            except URLError as exc:
                failure_reason = f"url-error:{exc.reason}"
            except Exception as exc:  # pragma: no cover
                failure_reason = type(exc).__name__

        if resolved_url:
            report_rows.append(
                {
                    "book_id": source["book_id"],
                    "song_no": song_no,
                    "title": source["title"],
                    "first_line": source.get("first_line_raw", ""),
                    "status": "filled",
                    "failure_reason": "",
                    "resolved_url": resolved_url,
                }
            )
        else:
            unresolved += 1
            print(f"unresolved {song_no} {source['title']} ({failure_reason})", flush=True)
            report_rows.append(
                {
                    "book_id": source["book_id"],
                    "song_no": song_no,
                    "title": source["title"],
                    "first_line": source.get("first_line_raw", ""),
                    "status": "unresolved",
                    "failure_reason": failure_reason,
                    "resolved_url": "",
                }
            )

    save_data_file(args.report_out, report_rows)
    print(f"missing_input={len(missing)}")
    print(f"filled={filled}")
    print(f"unresolved={unresolved}")
    print(f"report={args.report_out}")
    return 0


def _cmd_bremen(args: argparse.Namespace) -> int:
    from .bremen import cli as bremen_cli

    bremen_cli.dispatch(args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sh-corpus",
        description="Sacred Harp stanza extraction, classification, and selection tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract", help="Extract stanza records from lyrics sources.")
    extract.add_argument("--lyrics-csv", default=project_default("lyrics_csv"))
    extract.add_argument("--extra-lyrics-csv", action="append", default=[], help="Additional lyrics CSV(s) to merge before extraction.")
    extract.add_argument("--cooper-first-lines-csv", default=project_default("cooper_first_lines_csv"))
    extract.add_argument("--cooper-cache-dir", default=project_default("cooper_cache_dir"))
    extract.add_argument("--text-equivalence-overrides", default=project_default("text_equivalence_overrides"))
    extract.add_argument("--progress", action="store_true", help="Print phase timing and row progress to stderr.")
    extract.add_argument("--progress-every", type=int, default=250, help="Row interval for progress logs when --progress is enabled.")
    extract.add_argument("--stanzas-out", default=project_default("stanzas"))
    extract.add_argument("--review-out", default=project_default("review"))
    extract.set_defaults(func=_cmd_extract)

    classify = subparsers.add_parser("classify", help="Classify stanza records with rubric tags.")
    classify.add_argument("--stanzas", default=project_default("stanzas"))
    classify.add_argument("--rubric", default=project_default("rubric"))
    classify.add_argument("--batch-size", type=int, default=75)
    classify.add_argument("--review-dir", default=project_default("classification_review_dir"))
    classify.add_argument("--apply", action="store_true")
    classify.set_defaults(func=_cmd_classify)

    select = subparsers.add_parser("select", help="Select a stanza for the day.")
    select.add_argument("--operational-fit", nargs="*", default=[])
    select.add_argument("--theme", nargs="*", default=[])
    select.add_argument("--tone", nargs="*", default=[])
    select.add_argument("--tag", nargs="*", default=[])
    select.add_argument("--record-usage", action="store_true")
    select.add_argument("--stanzas", default=project_default("stanzas"))
    select.add_argument("--usage-history", default=project_default("usage_history"))
    select.add_argument("--rubric", default=project_default("rubric"))
    select.set_defaults(func=_cmd_select)

    review = subparsers.add_parser("review", help="Review extraction output: triage, extraction, or apply.")
    review_subparsers = review.add_subparsers(dest="review_command", required=True)

    review_triage = review_subparsers.add_parser("triage", help="Bucket extraction review items for manual triage.")
    review_triage.add_argument("--review", default=project_default("review"))
    review_triage.add_argument("--out", default=str(resolve_project_path(DEFAULT_REVIEW_TRIAGE_PATH)))
    review_triage.set_defaults(func=_cmd_triage_review)

    review_extraction = review_subparsers.add_parser("extraction", help="Interactively review extraction items.")
    review_extraction.add_argument("--review", default=project_default("review"))
    review_extraction.add_argument("--resolved", default=str(resolve_project_path(DEFAULT_EXTRACTION_REVIEW_RESOLVED_PATH)))
    review_extraction.add_argument("--triage", default=str(resolve_project_path(DEFAULT_REVIEW_TRIAGE_PATH)))
    review_extraction.add_argument(
        "--bucket",
        choices=["probable_junk", "probable_safe_irregular_lyric", "truly_ambiguous"],
    )
    review_extraction.set_defaults(func=_cmd_review_extraction)

    review_apply = review_subparsers.add_parser("apply", help="Apply extraction review resolutions.")
    review_apply.add_argument("--stanzas", default=project_default("stanzas"))
    review_apply.add_argument("--review", default=project_default("review"))
    review_apply.add_argument("--resolved", default=project_default("review_resolved"))
    review_apply.add_argument("--text-equivalence-overrides", default=project_default("text_equivalence_overrides"))
    review_apply.set_defaults(func=_cmd_apply_review)

    near_duplicates = subparsers.add_parser("near-duplicate-report", help="Report high-similarity stanza pairs that remain separate.")
    near_duplicates.add_argument("--stanzas", default=project_default("stanzas"))
    near_duplicates.add_argument("--out", default=project_default("near_duplicate_report"))
    near_duplicates.add_argument("--threshold", type=float, default=0.88)
    near_duplicates.set_defaults(func=_cmd_near_duplicate_report)

    fetch = subparsers.add_parser("fetch-missing-cooper", help="Fetch missing Texas Fasola Cooper poetry pages.")
    fetch.add_argument("--review", default=project_default("review"))
    fetch.add_argument("--cooper-first-lines", default=project_default("cooper_first_lines_csv"))
    fetch.add_argument("--cache-dir", default=project_default("cooper_cache_dir"))
    fetch.add_argument("--report-out", default=project_default("cooper_fetch_report"))
    fetch.set_defaults(func=_cmd_fetch_missing_cooper)

    from .bremen import cli as bremen_cli

    bremen_cli.register_subcommands(subparsers, handler=_cmd_bremen)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
