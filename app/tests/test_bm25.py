from app.retrieval.bm25 import BM25Config, BM25Index

import pytest

from app.core.exceptions import RetrievalError


def test_bm25_index_add_and_search() -> None:
    index = BM25Index()
    index.add_document("doc1", "hello world")
    index.add_document("doc2", "hello python")
    index.add_document("doc3", "world python")
    results = index.search("hello", top_k=2)
    assert len(results) == 2
    doc_ids = {doc_id for doc_id, _ in results}
    assert "doc1" in doc_ids
    assert "doc2" in doc_ids


def test_bm25_index_remove_document() -> None:
    index = BM25Index()
    index.add_document("doc1", "hello world")
    index.add_document("doc2", "hello python")
    index.remove_document("doc1")
    results = index.search("hello", top_k=5)
    assert len(results) == 1
    assert results[0][0] == "doc2"


def test_bm25_index_clear() -> None:
    index = BM25Index()
    index.add_document("doc1", "hello world")
    index.clear()
    results = index.search("hello", top_k=5)
    assert results == []


def test_bm25_index_empty_query() -> None:
    index = BM25Index()
    index.add_document("doc1", "hello world")
    results = index.search("", top_k=5)
    assert results == []


def test_bm25_index_no_documents() -> None:
    index = BM25Index()
    results = index.search("hello", top_k=5)
    assert results == []


def test_bm25_config_custom_values() -> None:
    config = BM25Config(k1=2.0, b=0.5)
    assert config.k1 == 2.0
    assert config.b == 0.5


def test_bm25_index_capacity_raises_instead_of_silent_drop() -> None:
    # Regression: add_document used to silently return at max_documents, so
    # new documents were quietly invisible to search. It must raise.
    index = BM25Index(BM25Config(max_documents=2))
    index.add_document("doc1", "hello world")
    index.add_document("doc2", "hello python")
    with pytest.raises(RetrievalError, match="capacity reached"):
        index.add_document("doc3", "hello rust")
    # The index still serves the documents it accepted.
    assert {doc_id for doc_id, _ in index.search("hello", top_k=10)} == {"doc1", "doc2"}


def test_bm25_index_readd_replaces_document() -> None:
    index = BM25Index()
    index.add_document("doc1", "hello world")
    index.add_document("doc1", "completely different terms")
    results = index.search("hello", top_k=5)
    assert results == []
    results = index.search("different", top_k=5)
    assert [doc_id for doc_id, _ in results] == ["doc1"]
    assert index.document_count == 1


def test_bm25_index_single_digit_tokens_are_searchable() -> None:
    # "第5条" must match documents containing "5": single digits were
    # previously dropped by the tokenizer, breaking clause-number recall.
    index = BM25Index()
    index.add_document("doc1", "第5条 员工享有年假。")
    results = index.search("第5条", top_k=5)
    assert [doc_id for doc_id, _ in results] == ["doc1"]


def test_bm25_index_search_is_scoped_by_allowed_doc_ids() -> None:
    index = BM25Index()
    index.add_document("doc1", "hello world")
    index.add_document("doc2", "hello python")
    index.add_document("doc3", "hello rust")
    results = index.search("hello", top_k=10, allowed_doc_ids={"doc2", "doc3"})
    assert {doc_id for doc_id, _ in results} == {"doc2", "doc3"}


def test_filters_version_semantics() -> None:
    from app.retrieval.filters import infer_metadata_filters, match_metadata_filters

    # "v2" matches "2.0" — one shared numeric-tuple semantic for filter and
    # ranking (previously the filter used string equality and hard-excluded).
    assert match_metadata_filters({"version": "2.0"}, {"version": "v2"}) is True
    assert match_metadata_filters({"version": "v2.1"}, {"version": "2.1"}) is True
    assert match_metadata_filters({"version": "2.0"}, {"version": "3.0"}) is False

    # A bare year must NOT expand into a hard date range that would exclude
    # the current version of a document ("2013年制定的规定现在还有效吗").
    inferred = infer_metadata_filters("2013年制定的规定现在还有效吗")
    assert "effective_date_from" not in inferred
    assert "effective_date_to" not in inferred

    # Explicit full dates still produce a soft date window.
    inferred = infer_metadata_filters("2024年6月1日之后生效的政策")
    assert inferred.get("effective_date_to") == "2024-06-01"

    # Soft doc-type mode no longer bypasses the version/date checks.
    assert (
        match_metadata_filters(
            {"doc_type": None, "version": "2.0"},
            {"doc_types": ["policy"], "doc_type_match_mode": "soft", "version": "3.0"},
        )
        is False
    )
