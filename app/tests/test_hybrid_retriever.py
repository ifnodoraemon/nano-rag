import os

import pytest

from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.bm25 import BM25Index
from app.schemas.chunk import Chunk
from app.vectorstore.repository import SearchHit


class FakeEmbeddingClient:
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:  # noqa: ARG002
        return [[1.0, 0.0]]


class NativeHybridRepository:
    def __init__(self) -> None:
        self.called_with: dict[str, object] = {}

    def native_hybrid_search(
        self,
        vector,
        query,
        top_k,
        kb_id="default",
        tenant_id=None,
        metadata_filters=None,
        dense_weight=0.7,
        sparse_weight=0.3,
    ):
        self.called_with = {
            "vector": vector,
            "query": query,
            "top_k": top_k,
            "kb_id": kb_id,
            "tenant_id": tenant_id,
            "metadata_filters": metadata_filters,
            "dense_weight": dense_weight,
            "sparse_weight": sparse_weight,
        }
        return [
            SearchHit(
                chunk=Chunk(
                    chunk_id="chunk-native",
                    doc_id="doc1",
                    chunk_index=0,
                    text="native hybrid hit",
                    source_path="test.txt",
                ),
                score=1.0,
            )
        ]


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


@pytest.mark.asyncio
async def test_hybrid_retriever_uses_native_repository_hybrid(monkeypatch) -> None:
    monkeypatch.setenv("RAG_HYBRID_SEARCH_ENABLED", "true")
    repository = NativeHybridRepository()
    retriever = HybridRetriever(
        repository=repository,
        embedding_client=FakeEmbeddingClient(),
    )

    hits = await retriever.retrieve(
        "policy query",
        top_k=3,
        kb_id="kb-a",
        tenant_id="tenant-a",
        metadata_filters={"doc_types": ["policy"]},
    )

    assert [hit.chunk.chunk_id for hit in hits] == ["chunk-native"]
    assert repository.called_with["query"] == "policy query"
    assert repository.called_with["kb_id"] == "kb-a"
    assert repository.called_with["tenant_id"] == "tenant-a"
