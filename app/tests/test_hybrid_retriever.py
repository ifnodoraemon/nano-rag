import os

from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.bm25 import BM25Index
from app.schemas.chunk import Chunk


def test_hybrid_retriever_index_chunk(monkeypatch) -> None:
    monkeypatch.setenv("RAG_HYBRID_SEARCH_ENABLED", "true")
    retriever = HybridRetriever(
        repository=None,
        embedding_client=None,
    )
    chunk = Chunk(
        chunk_id="chunk1",
        doc_id="doc1",
        chunk_index=0,
        text="hello world",
        source_path="test.txt",
    )
    retriever.index_chunk(chunk)
    results = retriever.bm25_index.search("hello", top_k=5)
    assert len(results) == 1
    assert results[0][0] == "chunk1"


def test_hybrid_retriever_remove_chunk(monkeypatch) -> None:
    monkeypatch.setenv("RAG_HYBRID_SEARCH_ENABLED", "true")
    retriever = HybridRetriever(
        repository=None,
        embedding_client=None,
    )
    chunk = Chunk(
        chunk_id="chunk1",
        doc_id="doc1",
        chunk_index=0,
        text="hello world",
        source_path="test.txt",
    )
    retriever.index_chunk(chunk)
    retriever.remove_chunk("chunk1")
    results = retriever.bm25_index.search("hello", top_k=5)
    assert len(results) == 0


def test_hybrid_retriever_clear_index(monkeypatch) -> None:
    monkeypatch.setenv("RAG_HYBRID_SEARCH_ENABLED", "true")
    retriever = HybridRetriever(
        repository=None,
        embedding_client=None,
    )
    chunk1 = Chunk(
        chunk_id="chunk1",
        doc_id="doc1",
        chunk_index=0,
        text="hello world",
        source_path="test.txt",
    )
    chunk2 = Chunk(
        chunk_id="chunk2",
        doc_id="doc1",
        chunk_index=1,
        text="hello python",
        source_path="test.txt",
    )
    retriever.index_chunk(chunk1)
    retriever.index_chunk(chunk2)
    retriever.clear_index()
    results = retriever.bm25_index.search("hello", top_k=5)
    assert len(results) == 0
