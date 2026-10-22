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

import math

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
    # Optional graded relevance, keyed by the same identifier values used above
    # (doc_id / source_key / title). Datasets that carry grades - T2Retrieval's
    # qrels do - should pass them: recall and MRR both throw the grades away,
    # while nDCG uses them to ask "are the *most* valuable results at the top".
    relevance_grades: dict[str, int] = field(default_factory=dict)
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
    grades_raw = raw.get("relevance_grades")
    if grades_raw is None:
        grades_raw = raw.get("grades")
    relevance_grades: dict[str, int] = {}
    if grades_raw is not None:
        if not isinstance(grades_raw, dict):
            raise StoreError(
                f"retrieval eval row {index}: relevance_grades must be an object "
                "mapping identifier -> integer grade"
            )
        for identifier, grade in grades_raw.items():
            try:
                numeric = int(grade)
            except (TypeError, ValueError) as exc:
                raise StoreError(
                    f"retrieval eval row {index}: grade for {identifier!r} must be "
                    f"an integer, got {grade!r}"
                ) from exc
            if numeric > 0:
                relevance_grades[str(identifier).strip().casefold()] = numeric
    return RetrievalRecord(
        sample_id=str(raw.get("sample_id") or f"retrieval-{index + 1}"),
        query=query,
        kb_id=str(raw.get("kb_id") or "default"),
        expected_doc_ids=expected_doc_ids,
        expected_source_keys=expected_source_keys,
        expected_titles=expected_titles,
        relevance_grades=relevance_grades,
        metadata_filters=metadata_filters,
    )


def _gain(document: RetrievedDocument, record: RetrievalRecord) -> int:
    """Relevance gain of one candidate.

    With graded relevance supplied, the gain is the best grade among the
    identifiers this document satisfies; without grades, every expected document
    is worth 1 (binary relevance). Non-expected documents gain nothing.
    """
    satisfied = _satisfied_expectations(document, record)
    if not satisfied:
        return 0
    if not record.relevance_grades:
        return 1
    grades = [
        record.relevance_grades.get(value)
        for _kind, value in satisfied
    ]
    numeric = [grade for grade in grades if isinstance(grade, int) and grade > 0]
    return max(numeric) if numeric else 1


def _dcg(gains: list[int]) -> float:
    # 2^rel - 1 is the standard NDCG gain, which makes a grade of 2 worth far
    # more than two grade-1 documents - that is the whole point of using grades.
    return sum(
        (2**gain - 1) / math.log2(index + 2) for index, gain in enumerate(gains)
    )


def _ndcg(
    documents: list[RetrievedDocument], record: RetrievalRecord, cutoff: int
) -> float:
    """Ranking quality at ``cutoff``.

    Recall@k cannot tell "the best document is first" from "it is tenth"; MRR
    sees only the first relevant hit and ignores the rest. nDCG scores the whole
    ordering with position discounts, so it is the metric that answers "is the
    overall ranking good, and are the high-value results near the top".
    """
    actual = _dcg([_gain(document, record) for document in documents[:cutoff]])
    if actual == 0.0:
        return 0.0
    # Ideal ordering: the best possible gains this corpus could produce.
    grade_pool = (
        sorted((grade for grade in record.relevance_grades.values() if grade > 0), reverse=True)
        if record.relevance_grades
        else [1] * record.expected_count
    )
    ideal = _dcg(grade_pool[:cutoff])
    if ideal == 0.0:
        return 0.0
    return min(1.0, actual / ideal)


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
    ndcg_at_k: dict[str, float] = field(default_factory=dict)
    # How clearly the right document beat the best wrong one. Recall cannot see
    # this: two configurations can both rank the answer first while one leaves
    # the runner-up almost tied (which is one query wording away from a miss).
    # The ratio is scale-free, so it is comparable across tokenizers whose raw
    # BM25 scores live on different scales (a tokenizer that emits more tokens
    # per character necessarily produces larger absolute scores).
    separation_margin: float | None = None
    separation_ratio: float | None = None


def _separation(
    documents: list[RetrievedDocument], record: RetrievalRecord
) -> tuple[float | None, float | None]:
    """(margin, ratio) describing how far the right document beat the best wrong one.

    margin = score(relevant) - score(best irrelevant), raw BM25 units: only
    comparable between runs of the *same* configuration.

    ratio  = 1 - score(best irrelevant) / score(relevant): scale-free, 0 means a
    tie and 1 means the right document completely dominates. This is the figure
    to compare across tokenizers, whose absolute scores are not on the same
    scale.
    """
    relevant = [d.score for d in documents if _is_relevant(d, record)]
    irrelevant = [d.score for d in documents if not _is_relevant(d, record)]
    if not relevant:
        return None, None
    best_relevant = max(relevant)
    best_irrelevant = max(irrelevant) if irrelevant else 0.0
    margin = round(best_relevant - best_irrelevant, 6)
    if best_relevant <= 0:
        return margin, None
    ratio = round(1.0 - (best_irrelevant / best_relevant), 6)
    return margin, ratio


def _score_row(
    record: RetrievalRecord,
    documents: list[RetrievedDocument],
    cutoffs: tuple[int, ...],
) -> _RowScore:
    row = _RowScore(record=record, ranking=[document.doc_id for document in documents])
    row.first_relevant_rank = _first_relevant_rank(documents, record)
    row.separation_margin, row.separation_ratio = _separation(documents, record)
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
        row.ndcg_at_k[str(cutoff)] = round(
            _ndcg(documents, record, cutoff), 4
        )
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
        # Higher is more robust; see _separation. The ratio is the one to
        # compare across configurations, the margin only across runs of one.
        "separation_ratio_avg": _mean_optional(
            [row.separation_ratio for row in rows]
        ),
        "separation_margin_avg": _mean_optional(
            [row.separation_margin for row in rows]
        ),
        "rows_without_scored_relevant": sum(
            1 for row in rows if row.separation_margin is None
        ),
    }
    for cutoff in ordered_cutoffs:
        key = str(cutoff)
        aggregate[f"ndcg@{cutoff}"] = round(
            sum(row.ndcg_at_k[key] for row in rows) / len(rows), 4
        )
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
                "ndcg_at_k": row.ndcg_at_k,
                "mrr": (
                    round(1.0 / row.first_relevant_rank, 4)
                    if row.first_relevant_rank
                    else 0.0
                ),
                "ranking": row.ranking,
                "separation_margin": row.separation_margin,
                "separation_ratio": row.separation_ratio,
            }
            for row in rows
        ],
    }


def _mean_optional(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return round(sum(present) / len(present), 4)


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
