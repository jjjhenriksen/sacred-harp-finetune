from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .models import AssociatedTune, EnrichedStanza, StanzaRecord, UsageEntry

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


DEFAULT_STANZAS_PATH = Path("data/stanzas.yaml")
DEFAULT_USAGE_HISTORY_PATH = Path("data/usage_history.yaml")
DEFAULT_RUBRIC_PATH = Path("data/tagging_rubric.yaml")
DEFAULT_EXTRACTION_REVIEW_PATH = Path("data/extraction_review.yaml")
DEFAULT_CLASSIFICATION_REVIEW_DIR = Path("data/classification_reviews")


def _normalize_for_dump(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize_for_dump(asdict(value))
    if isinstance(value, dict):
        return {k: _normalize_for_dump(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_for_dump(v) for v in value]
    return value


def load_data_file(path: str | Path, default: Any = None) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        return [] if default is None else default
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        return [] if default is None else default
    if yaml is not None:
        loaded = yaml.safe_load(text)
        if loaded is None:
            return [] if default is None else default
        return loaded
    return json.loads(text)


def save_data_file(path: str | Path, value: Any) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_for_dump(value)
    if yaml is not None:
        text = yaml.safe_dump(normalized, sort_keys=False, allow_unicode=True)
    else:
        text = json.dumps(normalized, ensure_ascii=False, indent=2) + "\n"
    file_path.write_text(text, encoding="utf-8")


def load_rubric(path: str | Path = DEFAULT_RUBRIC_PATH) -> dict[str, Any]:
    return load_data_file(path, default={})


def load_stanzas(path: str | Path = DEFAULT_STANZAS_PATH) -> list[dict[str, Any]]:
    rows = load_data_file(path, default=[])
    return list(rows)


def _associated_tunes_from_row(data: dict[str, Any]) -> list[AssociatedTune]:
    tunes: list[AssociatedTune] = []
    for row in data.get("associated_tunes", []) or []:
        tunes.append(
            AssociatedTune(
                book=str(row.get("book", "")),
                page=str(row.get("page", "")),
                tune=str(row.get("tune", "")),
                edition=str(row.get("edition", "")),
                url=str(row.get("url", "")),
            )
        )
    return tunes


def _associated_songs_from_data(data: dict[str, Any], tunes: list[AssociatedTune]) -> list[str]:
    existing = [str(item) for item in data.get("associated_songs", []) or [] if str(item)]
    if existing:
        return existing
    return sorted({f"{item.book}:{item.page}" for item in tunes if item.book and item.page})


def _source_books_from_data(data: dict[str, Any], tunes: list[AssociatedTune]) -> list[str]:
    existing = [str(item) for item in data.get("source_books", []) or [] if str(item)]
    if existing:
        return existing
    return sorted({item.book for item in tunes if item.book})


def load_usage_history(path: str | Path = DEFAULT_USAGE_HISTORY_PATH) -> list[UsageEntry]:
    rows = load_data_file(path, default=[])
    history: list[UsageEntry] = []
    for row in rows:
        history.append(
            UsageEntry(
                stanza_id=str(row["stanza_id"]),
                date_used=str(row["date_used"]),
            )
        )
    return history


def enrich_stanzas_with_usage(
    stanza_rows: list[dict[str, Any]] | list[StanzaRecord],
    usage_history: list[UsageEntry],
) -> list[EnrichedStanza]:
    counts = Counter(entry.stanza_id for entry in usage_history)
    last_dates: dict[str, str] = {}
    for entry in usage_history:
        current = last_dates.get(entry.stanza_id)
        if current is None or entry.date_used > current:
            last_dates[entry.stanza_id] = entry.date_used
    enriched: list[EnrichedStanza] = []
    for row in stanza_rows:
        data = asdict(row) if is_dataclass(row) else dict(row)
        associated_tunes = _associated_tunes_from_row(data)
        enriched.append(
            EnrichedStanza(
                id=data.get("id") or data["text_id"],
                text_id=data.get("text_id") or data["id"],
                incipit=data.get("incipit") or data["first_line"],
                first_line=data.get("first_line") or data.get("incipit") or "",
                text=data["text"],
                associated_tunes=associated_tunes,
                associated_songs=_associated_songs_from_data(data, associated_tunes),
                source_books=_source_books_from_data(data, associated_tunes),
                theme_tags=list(data.get("theme_tags", [])),
                tone_tags=list(data.get("tone_tags", [])),
                operational_fit=list(data.get("operational_fit", [])),
                tag_rationale=data.get("tag_rationale", ""),
                last_used_date=last_dates.get(data.get("id") or data["text_id"]),
                times_used=counts.get(data.get("id") or data["text_id"], 0),
            )
        )
    return enriched


def append_usage_entry(
    stanza_id: str,
    date_used: str | None = None,
    path: str | Path = DEFAULT_USAGE_HISTORY_PATH,
) -> UsageEntry:
    entry = UsageEntry(stanza_id=stanza_id, date_used=date_used or date.today().isoformat())
    history = load_data_file(path, default=[])
    history.append(asdict(entry))
    save_data_file(path, history)
    return entry
