from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import Any

from .storage import (
    DEFAULT_CLASSIFICATION_REVIEW_DIR,
    DEFAULT_RUBRIC_PATH,
    DEFAULT_STANZAS_PATH,
    load_data_file,
    save_data_file,
)


def load_rubric(path: str | Path = DEFAULT_RUBRIC_PATH) -> dict[str, Any]:
    rubric = load_data_file(path, default={})
    rubric.setdefault("allowed_tags", {})
    rubric.setdefault("rules", {})
    return rubric


def _match_patterns(text: str, patterns: list[str]) -> bool:
    lowered = text.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def classify_stanza_record(stanza: dict[str, Any], rubric: dict[str, Any]) -> dict[str, Any]:
    text = f"{stanza['first_line']}\n{stanza['text']}".lower()
    allowed = rubric["allowed_tags"]
    rules = rubric.get("rules", {})

    theme_tags = [
        tag for tag in allowed.get("theme_tags", [])
        if _match_patterns(text, rules.get("theme_tags", {}).get(tag, {}).get("patterns", []))
    ]
    tone_tags = [
        tag for tag in allowed.get("tone_tags", [])
        if _match_patterns(text, rules.get("tone_tags", {}).get(tag, {}).get("patterns", []))
    ]

    operational_fit: list[str] = []
    for tag in allowed.get("operational_fit", []):
        rule = rules.get("operational_fit", {}).get(tag, {})
        required_theme = set(rule.get("theme_tags", []))
        required_tone = set(rule.get("tone_tags", []))
        direct_patterns = rule.get("patterns", [])
        if (required_theme and required_theme.intersection(theme_tags)) or (required_tone and required_tone.intersection(tone_tags)):
            operational_fit.append(tag)
            continue
        if direct_patterns and _match_patterns(text, direct_patterns):
            operational_fit.append(tag)

    for bucket_name, bucket_tags in (
        ("theme_tags", theme_tags),
        ("tone_tags", tone_tags),
        ("operational_fit", operational_fit),
    ):
        unknown = set(bucket_tags) - set(allowed.get(bucket_name, []))
        if unknown:
            raise ValueError(f"Unknown {bucket_name}: {sorted(unknown)}")

    rationale_bits = []
    if theme_tags:
        rationale_bits.append(f"theme={', '.join(theme_tags)}")
    if tone_tags:
        rationale_bits.append(f"tone={', '.join(tone_tags)}")
    if operational_fit:
        rationale_bits.append(f"fit={', '.join(operational_fit)}")
    rationale = "; ".join(rationale_bits) if rationale_bits else "No rubric rule matched strongly enough to tag this stanza."

    updated = dict(stanza)
    updated["theme_tags"] = theme_tags
    updated["tone_tags"] = tone_tags
    updated["operational_fit"] = operational_fit
    updated["tag_rationale"] = rationale
    return updated


def build_batch_reviews(
    stanza_rows: list[dict[str, Any]],
    rubric: dict[str, Any],
    batch_size: int = 75,
    review_dir: str | Path = DEFAULT_CLASSIFICATION_REVIEW_DIR,
) -> list[Path]:
    output_dir = Path(review_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    total = len(stanza_rows)
    batches = ceil(total / batch_size) if total else 0
    for index in range(batches):
        start = index * batch_size
        batch = stanza_rows[start : start + batch_size]
        review_rows = [classify_stanza_record(row, rubric) for row in batch]
        path = output_dir / f"batch_{index + 1:03d}.yaml"
        save_data_file(path, review_rows)
        paths.append(path)
    return paths


def classify_stanzas_file(
    stanzas_path: str | Path = DEFAULT_STANZAS_PATH,
    rubric_path: str | Path = DEFAULT_RUBRIC_PATH,
    batch_size: int = 75,
    apply: bool = False,
    review_dir: str | Path = DEFAULT_CLASSIFICATION_REVIEW_DIR,
) -> list[Path]:
    stanzas = load_data_file(stanzas_path, default=[])
    rubric = load_rubric(rubric_path)
    paths = build_batch_reviews(stanzas, rubric, batch_size=batch_size, review_dir=review_dir)
    if apply:
        updated = [classify_stanza_record(row, rubric) for row in stanzas]
        save_data_file(stanzas_path, updated)
    return paths
