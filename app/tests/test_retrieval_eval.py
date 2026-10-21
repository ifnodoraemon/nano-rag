"""Retrieval-stage evaluation (discovery recall@k / MRR).

These pin the metric definitions and the "no silent zero" rules, plus an
end-to-end run over a real temporary wiki index so the harness is exercised
against the actual BM25 + version-filter stages rather than a fake.
"""

from __future__ import annotations

import pytest

from app.core.exceptions import StoreError
from app.eval.retrieval import (
    RetrievedDocument,
    evaluate_retrieval,
    failed_retrieval_gates,
    parse_retrieval_record,
)
from app.wiki.compiler import WikiCompiler
from app.wiki.search import WikiSearcher


def _doc(doc_id: str, source_key: str | None = None) -> RetrievedDocument:
    return RetrievedDocument(doc_id=doc_id, title=doc_id, source_key=source_key)


# --------------------------------------------------------------------------- #
# Dataset parsing: a row with no expectation is an error, not a zero          #
# --------------------------------------------------------------------------- #

def test_parse_record_accepts_doc_ids_or_source_keys() -> None:
    by_id = parse_retrieval_record({"query": "q", "expected_doc_ids": ["d1"]}, 0)
    by_key = parse_retrieval_record({"query": "q", "expected_source_keys": ["k1"]}, 1)
    assert by_id.expected_doc_ids == ("d1",)
    assert by_key.expected_source_keys == ("k1",)


def test_parse_record_rejects_row_without_expectation() -> None:
    with pytest.raises(StoreError):
        parse_retrieval_record({"query": "q"}, 0)


def test_parse_record_rejects_empty_query() -> None:
    with pytest.raises(StoreError):
        parse_retrieval_record({"query": "  ", "expected_doc_ids": ["d1"]}, 0)


def test_parse_record_rejects_non_list_expectation() -> None:
    with pytest.raises(StoreError):
        parse_retrieval_record({"query": "q", "expected_doc_ids": {"d1": True}}, 0)


# --------------------------------------------------------------------------- #
# Metric definitions                                                          #
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_recall_and_mrr_on_known_rankings() -> None:
    record = parse_retrieval_record(
        {"sample_id": "r1", "query": "q", "expected_doc_ids": ["hit"]}, 0
    )

    async def discover(_record):  # noqa: ANN001, ANN202
        return [_doc("a"), _doc("b"), _doc("hit"), _doc("c")]

    report = await evaluate_retrieval(discover, [record], cutoffs=(1, 4))
    aggregate = report["aggregate"]

    # Ranked 3rd: missed at k=1, found at k=4.
    assert aggregate["recall@1"] == 0.0
    assert aggregate["hit@1"] == 0.0
    assert aggregate["recall@4"] == 1.0
    assert aggregate["hit@4"] == 1.0
    # MRR is the reciprocal of the first relevant rank.
    assert aggregate["mrr"] == pytest.approx(1 / 3, abs=1e-4)
    assert report["results"][0]["first_relevant_rank"] == 3


@pytest.mark.asyncio
async def test_missing_document_is_counted_as_unretrieved() -> None:
    record = parse_retrieval_record(
        {"query": "q", "expected_doc_ids": ["absent"]}, 0
    )

    async def discover(_record):  # noqa: ANN001, ANN202
        return [_doc("a"), _doc("b")]

    report = await evaluate_retrieval(discover, [record], cutoffs=(1, 4, 12))
    aggregate = report["aggregate"]

    assert aggregate["unretrieved_records"] == 1
    assert aggregate["mrr"] == 0.0
    assert aggregate["recall@12"] == 0.0
    assert report["results"][0]["first_relevant_rank"] is None


@pytest.mark.asyncio
async def test_source_key_match_is_case_insensitive() -> None:
    record = parse_retrieval_record(
        {"query": "q", "expected_source_keys": ["Policies Travel"]}, 0
    )

    async def discover(_record):  # noqa: ANN001, ANN202
        return [_doc("d1", "policies travel")]

    report = await evaluate_retrieval(discover, [record], cutoffs=(1,))
    assert report["aggregate"]["recall@1"] == 1.0


@pytest.mark.asyncio
async def test_duplicate_titles_cannot_inflate_recall_above_one() -> None:
    # A corpus with several pages sharing one title must not produce
    # recall@k > 1.0: recall counts identifiers covered, not documents matched.
    record = parse_retrieval_record({"query": "q", "expected_titles": ["handbook"]}, 0)

    async def discover(_record):  # noqa: ANN001, ANN202
        return [
            RetrievedDocument(doc_id=f"d{i}", title="handbook") for i in range(7)
        ]

    report = await evaluate_retrieval(discover, [record], cutoffs=(1, 4, 12))
    for cutoff in (1, 4, 12):
        assert report["aggregate"][f"recall@{cutoff}"] == 1.0
    assert report["aggregate"]["mrr"] == 1.0


@pytest.mark.asyncio
async def test_title_matching_is_used_when_configured() -> None:
    record = parse_retrieval_record(
        {"query": "q", "expected_titles": ["Employee Handbook"]}, 0
    )

    async def discover(_record):  # noqa: ANN001, ANN202
        return [RetrievedDocument(doc_id="x", title="employee_handbook")]

    report = await evaluate_retrieval(discover, [record], cutoffs=(1,))
    # Title comparison is whitespace/case-normalized but not punctuation-blind.
    assert report["aggregate"]["recall@1"] == 0.0


@pytest.mark.asyncio
async def test_partial_recall_with_multiple_expected_documents() -> None:
    record = parse_retrieval_record(
        {"query": "q", "expected_doc_ids": ["d1", "d2"]}, 0
    )

    async def discover(_record):  # noqa: ANN001, ANN202
        return [_doc("d1"), _doc("x")]

    report = await evaluate_retrieval(discover, [record], cutoffs=(4,))
    assert report["aggregate"]["recall@4"] == 0.5
    assert report["aggregate"]["hit@4"] == 1.0


# --------------------------------------------------------------------------- #
# Gates fail closed                                                           #
# --------------------------------------------------------------------------- #

def test_gate_fails_on_missing_metric() -> None:
    failures = failed_retrieval_gates({}, min_recall_at={12: 0.9}, min_mrr=0.5)
    assert len(failures) == 2
    assert all("cannot be evaluated" in failure for failure in failures)


def test_gate_fails_below_threshold_and_passes_above() -> None:
    below = failed_retrieval_gates(
        {"recall@12": 0.5, "mrr": 0.4}, min_recall_at={12: 0.9}, min_mrr=0.7
    )
    assert len(below) == 2

    above = failed_retrieval_gates(
        {"recall@12": 0.95, "mrr": 0.8}, min_recall_at={12: 0.9}, min_mrr=0.7
    )
    assert above == []


# --------------------------------------------------------------------------- #
# End-to-end over a real wiki index                                           #
# --------------------------------------------------------------------------- #

def _write_page(root, doc_id: str, title: str, source_key: str, body: str) -> None:
    (root / "sources" / f"{doc_id}.md").write_text(
        "---\n"
        f"doc_id: {doc_id}\n"
        f"title: {title}\n"
        "kb_id: default\n"
        f"source_key: {source_key}\n"
        f"is_latest_version: true\n"
        "---\n"
        f"# {title}\n\n{body}\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_end_to_end_recall_over_real_index(tmp_path) -> None:
    root = tmp_path / "wiki"
    WikiCompiler(root)
    _write_page(
        root,
        "travel",
        "Travel Reimbursement",
        "policies travel",
        "Submit travel expense claims within 15 calendar days of the trip.",
    )
    _write_page(
        root,
        "leave",
        "Annual Leave",
        "policies leave",
        "Employees accrue 15 days of annual leave per year.",
    )
    searcher = WikiSearcher(root)

    record = parse_retrieval_record(
        {
            "sample_id": "r-travel",
            "query": "travel expense claim deadline",
            "expected_source_keys": ["policies travel"],
        },
        0,
    )

    async def discover(_record):  # noqa: ANN001, ANN202
        hits = searcher.search(_record.query, top_k=12, kb_id=_record.kb_id)
        return [
            RetrievedDocument(
                doc_id=hit.chunk.doc_id,
                title=hit.chunk.title,
                source_key=(hit.chunk.metadata or {}).get("source_key"),
                score=hit.score,
            )
            for hit in hits
            if (hit.chunk.metadata or {}).get("wiki_kind") == "source"
        ]

    report = await evaluate_retrieval(discover, [record], cutoffs=(1, 4, 12))

    assert report["stage"] == "discovery"
    assert report["aggregate"]["hit@4"] == 1.0
    assert report["aggregate"]["unretrieved_records"] == 0
    assert report["results"][0]["first_relevant_rank"] is not None
