from __future__ import annotations

from app.retrieval.filters import parse_date, version_key

__all__ = ["version_key", "version_sort_key"]


def version_sort_key(metadata: dict[str, object], *, score: object = 0.0) -> tuple[object, ...]:
    """Deterministic "newest applicable version first" ordering key.

    A version with an effective date outranks one without; then the later date
    wins; then the higher version tuple; then the caller-supplied tiebreak
    (BM25 discovery score in the wiki path, ISO updated_at string in the wiki
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
