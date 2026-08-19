#!/usr/bin/env python3
"""Build the generated Sacred Harp Obsidian notes from a source snapshot.

The source snapshot stays outside this repository because it may contain
private vault material. This entry point keeps all paths explicit and makes the
generation step reproducible from a clean checkout.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def existing_path(source_dir: Path, name: str) -> Path | None:
    candidate = source_dir / name
    return candidate if candidate.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Sacred Harp song and text-hub notes in an Obsidian vault."
    )
    parser.add_argument("--sources", required=True, type=Path, help="Source snapshot directory")
    parser.add_argument("--vault", required=True, type=Path, help="Destination Obsidian vault")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / "work",
        help="Generated reports and caches (default: vault_reproduction/work)",
    )
    parser.add_argument(
        "--first-lines",
        type=Path,
        help="Canonical first-lines CSV; defaults to <sources>/combined_first_lines_canonicalized.csv",
    )
    parser.add_argument(
        "--extra-lyrics",
        action="append",
        type=Path,
        default=[],
        help="Additional lyrics CSV; may be supplied more than once",
    )
    args = parser.parse_args()

    sources = args.sources.expanduser().resolve()
    vault = args.vault.expanduser().resolve()
    work = args.work_dir.expanduser().resolve()
    first_lines = (args.first_lines or sources / "combined_first_lines_canonicalized.csv").resolve()

    if not sources.is_dir():
        parser.error(f"source directory does not exist: {sources}")
    if not first_lines.is_file():
        parser.error(f"canonical first-lines CSV does not exist: {first_lines}")

    vault.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(ROOT / "sh_text_hubs.py"),
        "--vault",
        str(vault),
        "--first-lines",
        str(first_lines),
        "--report",
        str(work / "sh_text_hub_report.csv"),
        "--review",
        str(work / "needs_review_groups.csv"),
        "--changed-across-editions",
        str(work / "changed_across_editions.csv"),
        "--title-cache",
        str(work / "title_cache.json"),
    ]

    required_inputs = {
        "--stanzas": "data/stanzas.yaml",
        "--lyrics-csv": "bremen_lyrics_fixed4.csv",
    }
    for option, name in required_inputs.items():
        path = existing_path(sources, name)
        if path is None:
            parser.error(f"required source input does not exist: {sources / name}")
        command.extend([option, str(path)])

    tune_comparison = existing_path(sources, "texasfasola_tunecomparison.csv")
    if tune_comparison is not None:
        command.extend(["--tune-comparison", str(tune_comparison)])

    for lyrics in args.extra_lyrics:
        path = lyrics.expanduser().resolve()
        if not path.is_file():
            parser.error(f"lyrics CSV does not exist: {path}")
        command.extend(["--extra-lyrics-csv", str(path)])

    print("[run]", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"[ok] generated Sacred Harp notes under {vault / '04 Music/shape-note'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
