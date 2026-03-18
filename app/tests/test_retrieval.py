import pytest

from app.core.config import AppConfig
from app.core.tracing import TraceStore
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


@pytest.mark.asyncio
async def test_retrieval_pipeline_returns_contexts() -> None:
    repository = InMemoryVectorRepository()
    document = Document(doc_id="doc", source_path="/tmp/a.txt", title="A", content="...", metadata={})
    chunks = [
        Chunk(chunk_id="c1", doc_id="doc", chunk_index=0, text="aaa", source_path="/tmp/a.txt", title="A"),
        Chunk(chunk_id="c2", doc_id="doc", chunk_index=1, text="aaaaaa", source_path="/tmp/a.txt", title="A"),
    ]
    repository.upsert(document, chunks, [[3.0, 4.0], [6.0, 7.0]])
    config = AppConfig(
        config_dir=None,  # type: ignore[arg-type]
        settings={"retrieval": {"top_k": 2, "rerank_top_k": 2, "final_contexts": 1}},
        models={"model_gateway": {"base_url": "", "api_key": ""}},
        prompts={},
    )
    pipeline = RetrievalPipeline(config, repository, FakeEmbeddingClient(), FakeRerankClient(), TraceStore())

    contexts, trace = await pipeline.run("aaa", 2)

    assert len(contexts) == 1
    assert contexts[0]["chunk_id"] == "c2"
    assert trace["trace_id"]
