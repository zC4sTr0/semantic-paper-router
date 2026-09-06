"""Loading and validating the small reference corpus."""

from __future__ import annotations

import json
from pathlib import Path

from .classifier import Reference


def load_references(path: str | Path) -> tuple[Reference, ...]:
    """Load labeled references from a UTF-8 JSON array."""
    source = Path(path)
    try:
        records = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid corpus JSON: {source}") from exc
    if not isinstance(records, list) or not records:
        raise ValueError("corpus must be a non-empty JSON array")

    references = tuple(_reference_from_record(record) for record in records)
    ids = [reference.id for reference in references]
    if len(set(ids)) != len(ids):
        raise ValueError("reference ids must be unique")
    return references


def _reference_from_record(record: object) -> Reference:
    if not isinstance(record, dict):
        raise ValueError("each corpus item must be an object")
    try:
        reference = Reference(
            id=_required_text(record, "id"),
            category=_required_text(record, "category"),
            text=_required_text(record, "text"),
        )
    except KeyError as exc:
        raise ValueError(f"corpus item is missing {exc.args[0]!r}") from exc
    return reference


def _required_text(record: dict[str, object], field: str) -> str:
    value = record[field]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"corpus field {field!r} must be non-empty text")
    return value.strip()
