from __future__ import annotations

import re

from app.retrieval.filters import parse_date

__all__ = ["version_key", "version_sort_key"]


def version_key(value: object) -> tuple[int, ...]:
    """Extract a comparable version tuple from a free-form version string.

    Shared by the retrieval freshness ranking and the wiki version ledger so
    both apply the same "higher numeric version wins" rule.
    """
    if not isinstance(value, str):
        return ()
    parts = re.findall(r"\d+", value)
    if not parts:
        return ()
    return tuple(int(part) for part in parts)


def version_sort_key(metadata: dict[str, object], *, score: object = 0.0) -> tuple[object, ...]:
    """Deterministic "newest applicable version first" ordering key.

    A version with an effective date outranks one without; then the later date
    wins; then the higher version tuple; then the caller-supplied tiebreak
    (retrieval score in the dense path, ISO updated_at string in the wiki
    ledger). This is the evidence-backed primary sort for multi-version
    selection — it is metadata driven, never an LLM judgment.
    """
    effective_date = parse_date(metadata.get("effective_date"))
    return (
        1 if effective_date else 0,
        effective_date or "",
        version_key(metadata.get("version")),
        score,
    )
