from __future__ import annotations

import os
from pathlib import Path

DEFAULT_PATHS = {
    "lyrics_csv": "bremen_lyrics_fixed4.csv",
    "cooper_first_lines_csv": "cooper2012_first_lines_raw_texasfasola.csv",
    "cooper_cache_dir": ".cache/texasfasola_cooper/resources.texasfasola.org/poetry/cooper",
    "stanzas": "data/stanzas.yaml",
    "review": "data/extraction_review.yaml",
    "rubric": "data/tagging_rubric.yaml",
    "usage_history": "data/usage_history.yaml",
    "classification_review_dir": "data/classification_reviews",
    "review_triage": "data/review_triage.yaml",
    "review_resolved": "data/extraction_review_resolved.yaml",
    "text_equivalence_overrides": "data/text_equivalence_overrides.yaml",
    "near_duplicate_report": "data/near_duplicate_report.yaml",
    "cooper_fetch_report": "data/cooper_fetch_report.yaml",
    "bremen_cache": "bremen_first_lines.csv",
    "combined_cache": "combined_first_lines_canonicalized.csv",
    "bremen_lyrics": "bremen_lyrics.csv",
    "bremen_lyrics_fixed": "bremen_lyrics_fixed.csv",
    "bremen_overrides": "lyrics_overrides.csv",
}


def project_root() -> Path:
    override = os.environ.get("SH_CORPUS_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    candidate = Path(__file__).resolve().parents[2]
    if (candidate / "data").exists() or (candidate / "README.md").exists():
        return candidate

    return Path.cwd()


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return project_root() / candidate


def project_default(name: str) -> str:
    return str(resolve_project_path(DEFAULT_PATHS[name]))
