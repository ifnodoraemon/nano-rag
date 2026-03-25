import pytest

from app.core.config import AppConfig
from app.core.tracing import TraceStore, TracingManager
from app.model_client.embeddings import EmbeddingClient
from app.model_client.rerank import RerankClient
from app.retrieval.pipeline import RetrievalPipeline
from app.schemas.chunk import Chunk
from app.schemas.document import Document
from app.vectorstore.repository import InMemoryVectorRepository


class FakeEmbeddingClient(EmbeddingClient):
    def __init__(self) -> None:
        pass

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vectors.append([float(len(text)), float(text.count("a") + 1)])
        return vectors


class FakeRerankClient(RerankClient):
    def __init__(self) -> None:
        pass

    async def rerank(self, query: str, documents: list[str], top_k: int):
        ordered = sorted(range(len(documents)), key=lambda index: len(documents[index]), reverse=True)
        return [
            type("RerankResult", (), {"index": index, "score": float(len(documents[index])), "document": documents[index]})
            for index in ordered[:top_k]
        ]


class ExplodingRerankClient(RerankClient):
    def __init__(self) -> None:
        pass

    async def rerank(self, query: str, documents: list[str], top_k: int):
        raise AssertionError("rerank should be disabled for this test")


@pytest.mark.asyncio
async def test_retrieval_pipeline_returns_contexts() -> None:
    repository = InMemoryVectorRepository()
    document = Document(doc_id="doc", source_path="/tmp/a.txt", title="A", content="...", metadata={"kb_id": "default"})
    chunks = [
        Chunk(chunk_id="c1", doc_id="doc", chunk_index=0, text="aaa", source_path="/tmp/a.txt", title="A", metadata={"kb_id": "default"}),
        Chunk(chunk_id="c2", doc_id="doc", chunk_index=1, text="aaaaaa", source_path="/tmp/a.txt", title="A", metadata={"kb_id": "default"}),
    ]
    repository.upsert(document, chunks, [[3.0, 4.0], [6.0, 7.0]])
    config = AppConfig(
        config_dir=None,  # type: ignore[arg-type]
        settings={"retrieval": {"top_k": 2, "rerank_top_k": 2, "final_contexts": 1}},
        models={
            "model_gateway": {"base_url": "", "api_key": ""},
            "rerank": {"default_alias": "test-rerank"},
        },
        prompts={},
    )
    pipeline = RetrievalPipeline(
        config,
        repository,
        FakeEmbeddingClient(),
        FakeRerankClient(),
        TraceStore(),
        TracingManager("test-service", ""),
    )

    contexts, trace = await pipeline.run("aaa", 2)

    assert len(contexts) == 1
    assert contexts[0]["chunk_id"] == "c2"
    assert trace["trace_id"]


@pytest.mark.asyncio
async def test_retrieval_pipeline_skips_rerank_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("DISABLE_RERANK", "1")
    repository = InMemoryVectorRepository()
    document = Document(doc_id="doc", source_path="/tmp/a.txt", title="A", content="...", metadata={"kb_id": "default"})
    chunks = [
        Chunk(chunk_id="c1", doc_id="doc", chunk_index=0, text="aaa", source_path="/tmp/a.txt", title="A", metadata={"kb_id": "default"}),
        Chunk(chunk_id="c2", doc_id="doc", chunk_index=1, text="aaaaaa", source_path="/tmp/a.txt", title="A", metadata={"kb_id": "default"}),
    ]
    repository.upsert(document, chunks, [[3.0, 4.0], [6.0, 7.0]])
    config = AppConfig(
        config_dir=None,  # type: ignore[arg-type]
        settings={"retrieval": {"top_k": 2, "rerank_top_k": 2, "final_contexts": 1}},
        models={
            "model_gateway": {"base_url": "", "api_key": ""},
            "rerank": {"default_alias": "qwen3-rerank"},
        },
        prompts={},
    )
    pipeline = RetrievalPipeline(
        config,
        repository,
        FakeEmbeddingClient(),
        ExplodingRerankClient(),
        TraceStore(),
        TracingManager("test-service", ""),
    )

    contexts, trace = await pipeline.run("aaa", 2)

    assert len(contexts) == 1
    assert contexts[0]["chunk_id"] == "c1"
    assert trace["reranked_chunk_ids"] == ["c1", "c2"]


@pytest.mark.asyncio
async def test_retrieval_pipeline_scopes_results_by_kb_and_tenant() -> None:
    repository = InMemoryVectorRepository()
    repository.upsert(
        Document(doc_id="doc-a", source_path="/tmp/a.txt", title="A", content="...", metadata={"kb_id": "kb-a", "tenant_id": "tenant-a"}),
        [
            Chunk(
                chunk_id="c1",
                doc_id="doc-a",
                chunk_index=0,
                text="aaa",
                source_path="/tmp/a.txt",
                title="A",
                metadata={"kb_id": "kb-a", "tenant_id": "tenant-a"},
            )
        ],
        [[3.0, 4.0]],
    )
    repository.upsert(
        Document(doc_id="doc-b", source_path="/tmp/b.txt", title="B", content="...", metadata={"kb_id": "kb-b", "tenant_id": "tenant-b"}),
        [
            Chunk(
                chunk_id="c2",
                doc_id="doc-b",
                chunk_index=0,
                text="aaaaaa",
                source_path="/tmp/b.txt",
                title="B",
                metadata={"kb_id": "kb-b", "tenant_id": "tenant-b"},
            )
        ],
        [[6.0, 7.0]],
    )
    config = AppConfig(
        config_dir=None,  # type: ignore[arg-type]
        settings={"retrieval": {"top_k": 2, "rerank_top_k": 2, "final_contexts": 1}},
        models={"model_gateway": {"base_url": "", "api_key": ""}},
        prompts={},
    )
    pipeline = RetrievalPipeline(
        config,
        repository,
        FakeEmbeddingClient(),
        FakeRerankClient(),
        TraceStore(),
        TracingManager("test-service", ""),
    )

    contexts, trace = await pipeline.run("aaa", 2, kb_id="kb-a", tenant_id="tenant-a")

    assert len(contexts) == 1
    assert contexts[0]["chunk_id"] == "c1"
    assert trace["kb_id"] == "kb-a"
    assert trace["tenant_id"] == "tenant-a"
