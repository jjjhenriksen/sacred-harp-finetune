from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

from .models import EnrichedStanza, UsageEntry
from .storage import (
    DEFAULT_RUBRIC_PATH,
    DEFAULT_STANZAS_PATH,
    DEFAULT_USAGE_HISTORY_PATH,
    enrich_stanzas_with_usage,
    load_data_file,
    load_usage_history,
)


def top_band_size(candidate_count: int) -> int:
    return max(3, min(10, math.ceil(0.05 * candidate_count)))


def days_since(iso_date: str | None, today: date) -> int | None:
    if not iso_date:
        return None
    return (today - date.fromisoformat(iso_date)).days


def novelty_score(stanza: EnrichedStanza, today: date) -> int:
    age = days_since(stanza.last_used_date, today)
    if stanza.times_used == 0:
        return 3
    if stanza.times_used == 1 and age is not None and age > 90:
        return 2
    if stanza.times_used <= 3 and age is not None and age > 30:
        return 1
    return 0


def underuse_score(stanza: EnrichedStanza) -> int:
    if stanza.times_used == 0:
        return 3
    if stanza.times_used == 1:
        return 2
    if stanza.times_used <= 3:
        return 1
    return 0


def recency_penalty(stanza: EnrichedStanza, today: date) -> int:
    age = days_since(stanza.last_used_date, today)
    if age is None:
        return 0
    if age <= 7:
        return 100
    if age <= 30:
        return 5
    if age <= 90:
        return 2
    return 0


def fit_score(stanza: EnrichedStanza, operational_fit: list[str], theme_tags: list[str], tone_tags: list[str]) -> int:
    return (
        3 * len(set(operational_fit).intersection(stanza.operational_fit))
        + 2 * len(set(theme_tags).intersection(stanza.theme_tags))
        + 1 * len(set(tone_tags).intersection(stanza.tone_tags))
    )


def recent_theme_counts(
    stanza_rows: list[dict[str, Any]],
    usage_history_rows: list[dict[str, str]],
    window_size: int = 7,
) -> Counter[str]:
    themes_by_id = {row["id"]: list(row.get("theme_tags", [])) for row in stanza_rows}
    recent_entries = sorted(
        usage_history_rows,
        key=lambda row: row["date_used"],
        reverse=True,
    )[:window_size]
    counts: Counter[str] = Counter()
    for entry in recent_entries:
        for theme in themes_by_id.get(entry["stanza_id"], []):
            counts[theme] += 1
    return counts


def diversity_score(stanza: EnrichedStanza, theme_counts: Counter[str], fit: int) -> int:
    if fit == 0:
        return 0
    if not stanza.theme_tags:
        return 0
    score = 0
    for theme in set(stanza.theme_tags):
        seen = theme_counts.get(theme, 0)
        if seen == 0:
            score += 1
        elif seen == 1:
            score += 0
        else:
            score -= 1
    return max(-2, min(1, score))


def validate_selector_inputs(
    rubric: dict[str, Any],
    operational_fit: list[str],
    theme_tags: list[str],
    tone_tags: list[str],
) -> None:
    allowed = rubric.get("allowed_tags", {})
    for bucket_name, values in (
        ("operational_fit", operational_fit),
        ("theme_tags", theme_tags),
        ("tone_tags", tone_tags),
    ):
        invalid = set(values) - set(allowed.get(bucket_name, []))
        if invalid:
            raise ValueError(f"Invalid {bucket_name}: {sorted(invalid)}")


def _blocked_recently(stanza: EnrichedStanza, today: date) -> bool:
    age = days_since(stanza.last_used_date, today)
    return age is not None and age <= 7


def _strip_recent_block(candidates: list[dict[str, Any]], today: date) -> list[dict[str, Any]]:
    unblocked = [candidate for candidate in candidates if not _blocked_recently(candidate["stanza"], today)]
    return unblocked or candidates


def _rank_candidates(
    stanzas: list[EnrichedStanza],
    stanza_rows: list[dict[str, Any]],
    usage_history_rows: list[dict[str, str]],
    today: date,
    operational_fit: list[str],
    theme_tags: list[str],
    tone_tags: list[str],
) -> list[dict[str, Any]]:
    theme_counts = recent_theme_counts(stanza_rows, usage_history_rows)
    ranked = []
    for stanza in stanzas:
        fit = fit_score(stanza, operational_fit, theme_tags, tone_tags)
        novelty = novelty_score(stanza, today)
        underuse = underuse_score(stanza)
        diversity = diversity_score(stanza, theme_counts, fit)
        penalty = recency_penalty(stanza, today)
        ranked.append(
            {
                "stanza": stanza,
                "fit_score": fit,
                "novelty_score": novelty,
                "underuse_score": underuse,
                "diversity_score": diversity,
                "recency_penalty": penalty,
                "candidate_score": fit + novelty + underuse + diversity - penalty,
            }
        )
    return ranked


def select_stanza(
    *,
    operational_fit: list[str],
    theme_tags: list[str],
    tone_tags: list[str],
    today: date | None = None,
    rng: random.Random | None = None,
    usage_history: list[dict[str, str]] | None = None,
    stanzas_path: str | Path = DEFAULT_STANZAS_PATH,
    usage_history_path: str | Path = DEFAULT_USAGE_HISTORY_PATH,
    rubric_path: str | Path = DEFAULT_RUBRIC_PATH,
) -> dict[str, Any]:
    actual_today = today or date.today()
    actual_rng = rng or random.Random()
    if not operational_fit and not theme_tags and not tone_tags:
        operational_fit = ["ordinary-day"]

    rubric = load_data_file(rubric_path, default={})
    validate_selector_inputs(rubric, operational_fit, theme_tags, tone_tags)

    stanza_rows = load_data_file(stanzas_path, default=[])
    history_rows = usage_history if usage_history is not None else [
        asdict(entry) for entry in load_usage_history(usage_history_path)
    ]
    usage_entries = [UsageEntry(stanza_id=row["stanza_id"], date_used=row["date_used"]) for row in history_rows]
    enriched = enrich_stanzas_with_usage(stanza_rows, usage_entries)
    if not enriched:
        raise ValueError("No stanza records available for selection.")

    ranked = _rank_candidates(enriched, stanza_rows, history_rows, actual_today, operational_fit, theme_tags, tone_tags)
    strong_pool = [candidate for candidate in ranked if candidate["fit_score"] > 0] or ranked
    strong_pool = _strip_recent_block(strong_pool, actual_today)
    strong_sorted = sorted(
        strong_pool,
        key=lambda candidate: (
            candidate["candidate_score"],
            candidate["fit_score"],
            candidate["novelty_score"],
            candidate["underuse_score"],
            candidate["stanza"].id,
        ),
        reverse=True,
    )
    strong_band = strong_sorted[: top_band_size(len(strong_sorted))]

    surprise_roll = actual_rng.random()
    if surprise_roll < 0.1:
        strong_ids = {candidate["stanza"].id for candidate in strong_band}
        surprise_candidates = [candidate for candidate in ranked if candidate["stanza"].id not in strong_ids]
        surprise_candidates = _strip_recent_block(surprise_candidates, actual_today)
        if surprise_candidates:
            surprise_sorted = sorted(
                surprise_candidates,
                key=lambda candidate: (
                    candidate["novelty_score"] + candidate["underuse_score"] + 0.5 * candidate["fit_score"],
                    candidate["novelty_score"],
                    candidate["underuse_score"],
                    candidate["fit_score"],
                    candidate["stanza"].id,
                ),
                reverse=True,
            )
            surprise_band = surprise_sorted[: top_band_size(len(surprise_sorted))]
            return actual_rng.choice(surprise_band)["stanza"].to_dict()

    return actual_rng.choice(strong_band)["stanza"].to_dict()
