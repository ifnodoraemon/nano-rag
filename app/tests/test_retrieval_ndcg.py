"""nDCG for retrieval ranking, and graded relevance support.

Recall@k answers "was it found"; MRR answers "how early was the *first* hit".
Neither sees the rest of the ordering, so a run that puts the best three results
in the top three and one that scatters them score identically on MRR.
"""

from __future__ import annotations

import math

import pytest

from app.core.exceptions import StoreError
from app.eval.retrieval import (
    RetrievedDocument,
    RetrievalRecord,
    evaluate_retrieval,
    parse_retrieval_record,
)


def _doc(doc_id: str, score: float = 1.0) -> RetrievedDocument:
    return RetrievedDocument(doc_id=doc_id, title=doc_id, score=score)


def _record(doc_ids: tuple[str, ...], grades: dict[str, int] | None = None) -> RetrievalRecord:
    return RetrievalRecord(
        sample_id="s1",
        query="q",
        expected_doc_ids=doc_ids,
        relevance_grades=grades or {},
    )


async def _evaluate(documents: list[RetrievedDocument], record: RetrievalRecord, cutoffs=(1, 4, 12)):
    async def discover(_record):  # noqa: ANN001, ANN202
        return documents

    return await evaluate_retrieval(discover, [record], cutoffs)


@pytest.mark.asyncio
async def test_ndcg_is_one_when_the_only_relevant_document_is_first() -> None:
    report = await _evaluate([_doc("a"), _doc("b")], _record(("a",)))
    assert report["aggregate"]["ndcg@1"] == 1.0
    assert report["aggregate"]["ndcg@4"] == 1.0


@pytest.mark.asyncio
async def test_ndcg_discounts_position() -> None:
    """The same relevant document at rank 2 must score lower than at rank 1."""
    first = await _evaluate([_doc("a"), _doc("x")], _record(("a",)))
    second = await _evaluate([_doc("x"), _doc("a")], _record(("a",)))
    assert first["aggregate"]["ndcg@4"] == 1.0
    assert 0.0 < second["aggregate"]["ndcg@4"] < 1.0
    # Standard DCG discount for rank 2: 1 / log2(3).
    assert second["aggregate"]["ndcg@4"] == pytest.approx(1 / math.log2(3), abs=1e-4)


@pytest.mark.asyncio
async def test_ndcg_sees_ordering_that_mrr_cannot() -> None:
    """Two relevant documents: MRR is identical, nDCG is not.

    This is the reason to carry both - MRR only cares about the first hit.
    """
    clustered = await _evaluate([_doc("a"), _doc("b"), _doc("x")], _record(("a", "b")))
    scattered = await _evaluate([_doc("a"), _doc("x"), _doc("b")], _record(("a", "b")))

    assert clustered["aggregate"]["mrr"] == scattered["aggregate"]["mrr"] == 1.0
    assert clustered["aggregate"]["ndcg@12"] > scattered["aggregate"]["ndcg@12"]


@pytest.mark.asyncio
async def test_ndcg_is_zero_when_nothing_relevant_is_retrieved() -> None:
    report = await _evaluate([_doc("x"), _doc("y")], _record(("a",)))
    assert report["aggregate"]["ndcg@4"] == 0.0
    assert report["aggregate"]["unretrieved_records"] == 1


@pytest.mark.asyncio
async def test_graded_relevance_prefers_the_high_value_document() -> None:
    """With grades, rank 1 matters more: a grade-3 doc must beat a grade-1 doc.

    Binary relevance would score both runs identically (one relevant doc, rank 1).
    """
    graded = {"a": 3, "b": 1}
    high_first = await _evaluate(
        [_doc("a"), _doc("b")], _record(("a", "b"), graded), cutoffs=(1, 2)
    )
    low_first = await _evaluate(
        [_doc("b"), _doc("a")], _record(("a", "b"), graded), cutoffs=(1, 2)
    )
    assert high_first["aggregate"]["recall@1"] == low_first["aggregate"]["recall@1"]
    assert high_first["aggregate"]["ndcg@2"] > low_first["aggregate"]["ndcg@2"]


@pytest.mark.asyncio
async def test_report_rows_carry_ndcg_and_mrr_per_row() -> None:
    report = await _evaluate([_doc("a")], _record(("a",)))
    row = report["results"][0]
    assert row["ndcg_at_k"]["1"] == 1.0
    assert row["mrr"] == 1.0


def test_dataset_parser_reads_graded_relevance() -> None:
    record = parse_retrieval_record(
        {
            "query": "q",
            "expected_doc_ids": ["doc-1"],
            "relevance_grades": {"doc-1": 2, "doc-ignored": 0},
        },
        0,
    )
    assert record.relevance_grades == {"doc-1": 2}


def test_dataset_parser_rejects_a_non_integer_grade() -> None:
    with pytest.raises(StoreError, match="must be an integer"):
        parse_retrieval_record(
            {
                "query": "q",
                "expected_doc_ids": ["doc-1"],
                "relevance_grades": {"doc-1": "high"},
            },
            0,
        )
