from app.retrieval.bm25 import BM25Config, BM25Index


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
    assert len(results) == 0


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
