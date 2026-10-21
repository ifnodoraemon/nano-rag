"""Retrieval-stage evaluation: did discovery surface the right document?

The pipeline is "BM25 document discovery -> deterministic version filter ->
LLM reading plan -> deep read". Everything downstream is wasted if the first
hop misses the document, yet the existing evaluation only scores the *answer*
(exact match, reference-context recall, deepeval). Those metrics cannot tell
you whether a change to tokenization, ``discovery_top_k`` or the version
filter improved discovery or merely produced a differently-worded answer.

This module scores the deterministic discovery stages alone:

- ``recall@k`` — of the documents a question needs, how many are in the top k;
- ``hit@k``    — did *any* expected document make the top k;
- ``MRR``      — how high the first expected document ranked.

It never calls a model, so it is cheap, reproducible, and usable as a gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable

from app.core.exceptions import StoreError

DEFAULT_CUTOFFS: tuple[int, ...] = (1, 4, 12)


@dataclass(frozen=True)
class RetrievedDocument:
    """One candidate as the reading plan would see it."""

    doc_id: str
    title: str = ""
    source_key: str | None = None
    score: float = 0.0


@dataclass(frozen=True)
class RetrievalRecord:
    """One evaluation question.

    Three ways to identify the expected document, any of which counts as a
    match: ``expected_doc_ids`` (exact, but changes when a corpus is rebuilt),
    ``expected_source_keys`` (the version-ledger identity, derived from the
    ingest path so it differs between a path ingest and an upload), and
    ``expected_titles`` (stable for a given corpus and therefore the sensible
    choice for a dataset committed next to the code).

    At least one must be present: a row with no expectation cannot be scored
    and is a dataset error, not a zero.
    """

    sample_id: str
    query: str
    kb_id: str = "default"
    expected_doc_ids: tuple[str, ...] = ()
    expected_source_keys: tuple[str, ...] = ()
    expected_titles: tuple[str, ...] = ()
    metadata_filters: dict[str, object] | None = None

    @property
    def expectations(self) -> tuple[tuple[str, str], ...]:
        """Distinct (kind, value) identifiers this row expects to retrieve.

        Recall counts *identifiers covered*, not *documents matched*. Counting
        documents let a corpus with several pages sharing one title inflate
        recall above 1.0 (7 matching pages / 1 expectation).
        """
        identifiers: list[tuple[str, str]] = []
        for kind, values in (
            ("doc_id", self.expected_doc_ids),
            ("source_key", self.expected_source_keys),
            ("title", self.expected_titles),
        ):
            for value in values:
                identifier = (kind, value.casefold())
                if identifier not in identifiers:
                    identifiers.append(identifier)
        return tuple(identifiers)

    @property
    def expected_count(self) -> int:
        return len(self.expectations) or 1


Discover = Callable[[RetrievalRecord], Awaitable[list[RetrievedDocument]]]


def parse_retrieval_record(raw: dict, index: int) -> RetrievalRecord:
    """Validate one dataset row. Malformed rows raise instead of scoring 0."""
    if not isinstance(raw, dict):
        raise StoreError(f"retrieval eval row {index} is not an object")
    query = str(raw.get("query") or "").strip()
    if not query:
        raise StoreError(f"retrieval eval row {index} has an empty query")

    def _ids(key: str) -> tuple[str, ...]:
        value = raw.get(key) or []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise StoreError(f"retrieval eval row {index}: {key} must be a list")
        return tuple(str(item).strip() for item in value if str(item).strip())

    expected_doc_ids = _ids("expected_doc_ids")
    expected_source_keys = _ids("expected_source_keys")
    expected_titles = _ids("expected_titles")
    if not expected_doc_ids and not expected_source_keys and not expected_titles:
        raise StoreError(
            f"retrieval eval row {index} ({query!r}) declares no expectation; "
            "set expected_doc_ids, expected_source_keys or expected_titles"
        )
    metadata_filters = raw.get("metadata_filters")
    if metadata_filters is not None and not isinstance(metadata_filters, dict):
        raise StoreError(
            f"retrieval eval row {index}: metadata_filters must be an object"
        )
    return RetrievalRecord(
        sample_id=str(raw.get("sample_id") or f"retrieval-{index + 1}"),
        query=query,
        kb_id=str(raw.get("kb_id") or "default"),
        expected_doc_ids=expected_doc_ids,
        expected_source_keys=expected_source_keys,
        expected_titles=expected_titles,
        metadata_filters=metadata_filters,
    )


def _satisfied_expectations(
    document: RetrievedDocument, record: RetrievalRecord
) -> set[tuple[str, str]]:
    """Which of the row's expectations this document satisfies."""
    satisfied: set[tuple[str, str]] = set()
    doc_id = (("doc_id", document.doc_id.casefold()),)
    if doc_id[0] in record.expectations:
        satisfied.update(doc_id)
    if document.source_key:
        identifier = ("source_key", document.source_key.casefold())
        if identifier in record.expectations:
            satisfied.add(identifier)
    if document.title:
        identifier = ("title", document.title.strip().casefold())
        if identifier in record.expectations:
            satisfied.add(identifier)
    return satisfied


def _is_relevant(document: RetrievedDocument, record: RetrievalRecord) -> bool:
    return bool(_satisfied_expectations(document, record))


def _first_relevant_rank(
    documents: list[RetrievedDocument], record: RetrievalRecord
) -> int | None:
    for position, document in enumerate(documents, start=1):
        if _is_relevant(document, record):
            return position
    return None


@dataclass
class _RowScore:
    record: RetrievalRecord
    ranking: list[str] = field(default_factory=list)
    first_relevant_rank: int | None = None
    recall_at_k: dict[str, float] = field(default_factory=dict)
    hit_at_k: dict[str, float] = field(default_factory=dict)


def _score_row(
    record: RetrievalRecord,
    documents: list[RetrievedDocument],
    cutoffs: tuple[int, ...],
) -> _RowScore:
    row = _RowScore(record=record, ranking=[document.doc_id for document in documents])
    row.first_relevant_rank = _first_relevant_rank(documents, record)
    expected = set(record.expectations)
    for cutoff in cutoffs:
        covered: set[tuple[str, str]] = set()
        for document in documents[:cutoff]:
            covered |= _satisfied_expectations(document, record)
        # Intersect defensively: recall above 1.0 is never meaningful.
        row.recall_at_k[str(cutoff)] = round(
            len(covered & expected) / record.expected_count, 4
        )
        row.hit_at_k[str(cutoff)] = 1.0 if covered & expected else 0.0
    return row


async def evaluate_retrieval(
    discover: Discover,
    records: list[RetrievalRecord],
    cutoffs: tuple[int, ...] = DEFAULT_CUTOFFS,
) -> dict[str, object]:
    """Score the discovery stage over a dataset.

    Returns a report shaped like the answer-level ones (mode/records/aggregate/
    results) so it can go through the same report gates.
    """
    if not records:
        raise StoreError("retrieval evaluation needs at least one record")

    ordered_cutoffs = tuple(sorted({cutoff for cutoff in cutoffs if cutoff > 0}))
    if not ordered_cutoffs:
        raise StoreError("retrieval evaluation needs at least one positive cutoff")

    rows: list[_RowScore] = []
    for record in records:
        documents = await discover(record)
        rows.append(_score_row(record, documents, ordered_cutoffs))

    aggregate: dict[str, object] = {
        "mrr": round(
            sum(1.0 / row.first_relevant_rank for row in rows if row.first_relevant_rank)
            / len(rows),
            4,
        ),
        # A query whose expected document never appears anywhere in the
        # candidate list is the interesting failure: no cutoff can rescue it.
        "unretrieved_records": sum(
            1 for row in rows if row.first_relevant_rank is None
        ),
        "candidate_count_avg": round(
            sum(len(row.ranking) for row in rows) / len(rows), 4
        ),
    }
    for cutoff in ordered_cutoffs:
        key = str(cutoff)
        aggregate[f"recall@{cutoff}"] = round(
            sum(row.recall_at_k[key] for row in rows) / len(rows), 4
        )
        aggregate[f"hit@{cutoff}"] = round(
            sum(row.hit_at_k[key] for row in rows) / len(rows), 4
        )

    return {
        "status": "ok",
        "mode": "deterministic",
        "stage": "discovery",
        "records": len(rows),
        "cutoffs": list(ordered_cutoffs),
        "aggregate": aggregate,
        "results": [
            {
                "sample_id": row.record.sample_id,
                "query": row.record.query,
                "kb_id": row.record.kb_id,
                "expected_doc_ids": list(row.record.expected_doc_ids),
                "expected_source_keys": list(row.record.expected_source_keys),
                "expected_titles": list(row.record.expected_titles),
                "first_relevant_rank": row.first_relevant_rank,
                "recall_at_k": row.recall_at_k,
                "hit_at_k": row.hit_at_k,
                "ranking": row.ranking,
            }
            for row in rows
        ],
    }


def failed_retrieval_gates(
    aggregate: dict[str, object],
    *,
    min_recall_at: dict[int, float] | None = None,
    min_mrr: float | None = None,
) -> list[str]:
    """Gate a retrieval report; a missing metric fails instead of passing.

    Same rule as ``scripts/run_eval.py``: a gate that cannot be evaluated is a
    failure, because "we could not measure it" is exactly when a regression
    slips through.
    """
    failures: list[str] = []
    for cutoff, threshold in (min_recall_at or {}).items():
        key = f"recall@{cutoff}"
        value = aggregate.get(key)
        if value is None:
            failures.append(f"{key} is missing from the report; gate cannot be evaluated")
            continue
        if float(value) < threshold:  # type: ignore[arg-type]
            failures.append(f"{key}={float(value):.4f} is below threshold {threshold:.4f}")  # type: ignore[arg-type]
    if min_mrr is not None:
        value = aggregate.get("mrr")
        if value is None:
            failures.append("mrr is missing from the report; gate cannot be evaluated")
        elif float(value) < min_mrr:  # type: ignore[arg-type]
            failures.append(f"mrr={float(value):.4f} is below threshold {min_mrr:.4f}")  # type: ignore[arg-type]
    return failures
