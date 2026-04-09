import json

import pytest

from app.core.config import AppConfig
from app.core.tracing import TraceStore, TracingManager
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.query_rewriter import QueryRewriter, QueryRewriterConfig
from app.schemas.chunk import Chunk
from app.schemas.document import Document
from app.vectorstore.repository import InMemoryVectorRepository
from app.wiki.compiler import WikiCompiler


class FakeEmbeddingClient:
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            if lowered == "pto carryover":
                vectors.append([1.0, 0.0])
                continue
            vectors.append(
                [
                    1.0 if "vacation" in lowered else 0.0,
                    1.0 if "pto" in lowered else 0.0,
                ]
            )
        return vectors


class FakeRerankClient:
    async def rerank(self, query: str, documents: list[str], top_k: int):  # noqa: ARG002
        return [
            type(
                "RerankResult",
                (),
                {"index": index, "score": float(top_k - index), "document": document},
            )
            for index, document in enumerate(documents[:top_k])
        ]


class FakeGenerationClient:
    async def generate(self, messages: list[dict[str, str]]) -> dict[str, str]:
        prompt = messages[0]["content"]
        if "Rewritten query:" in prompt:
            return {"content": "vacation carryover"}
        if "Generate 2 queries" in prompt:
            return {"content": "1. PTO carryover\n2. leave rollover"}
        return {"content": "internal PTO carryover handbook section"}


def _build_config(tmp_path) -> AppConfig:
    return AppConfig(
        config_dir=tmp_path,
        settings={"retrieval": {"top_k": 4, "rerank_top_k": 4, "final_contexts": 2}},
        models={
            "model_gateway": {"base_url": "", "api_key": ""},
            "rerank": {"default_alias": "disabled"},
        },
        prompts={},
    )


@pytest.mark.asyncio
async def test_retrieval_pipeline_uses_hybrid_search_to_surface_bm25_only_hits(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("RAG_HYBRID_SEARCH_ENABLED", "true")
    monkeypatch.setenv("RAG_HYBRID_VECTOR_WEIGHT", "0.1")
    monkeypatch.setenv("RAG_HYBRID_BM25_WEIGHT", "0.9")
    repository = InMemoryVectorRepository()
    document = Document(
        doc_id="doc-1",
        source_path="data/raw/policy.md",
        title="Policy",
        content="...",
        metadata={"kb_id": "default"},
    )
    chunks = [
        Chunk(
            chunk_id="c-vacation",
            doc_id="doc-1",
            chunk_index=0,
            text="Vacation policy overview",
            source_path="data/raw/policy.md",
            title="Policy",
            metadata={"kb_id": "default"},
        ),
        Chunk(
            chunk_id="c-pto",
            doc_id="doc-1",
            chunk_index=1,
            text="PTO carryover rules",
            source_path="data/raw/policy.md",
            title="Policy",
            metadata={"kb_id": "default"},
        ),
    ]
    repository.upsert(document, chunks, [[1.0, 0.0], [0.0, 0.0]])
    hybrid = HybridRetriever(repository=repository, embedding_client=FakeEmbeddingClient())
    hybrid.index_chunks(chunks)
    pipeline = RetrievalPipeline(
        config=_build_config(tmp_path),
        repository=repository,
        embedding_client=FakeEmbeddingClient(),
        rerank_client=FakeRerankClient(),
        trace_store=TraceStore(),
        tracing_manager=TracingManager("test-service", ""),
        hybrid_retriever=hybrid,
    )

    contexts, trace = await pipeline.run("pto carryover", 2)

    assert [context["chunk_id"] for context in contexts] == ["c-pto", "c-vacation"]
    assert trace["retrieved_chunk_ids"][:2] == ["c-pto", "c-vacation"]


def test_hybrid_retriever_bootstraps_from_parsed_artifacts(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RAG_HYBRID_SEARCH_ENABLED", "true")
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    artifact = {
        "document": {
            "doc_id": "doc-1",
            "source_path": "data/raw/guide.md",
            "title": "Guide",
            "content": "...",
            "metadata": {"kb_id": "kb-a", "tenant_id": "tenant-a"},
        },
        "chunks": [
            {
                "chunk_id": "doc-1:0",
                "doc_id": "doc-1",
                "chunk_index": 0,
                "text": "vacation and reimbursement policy",
                "source_path": "data/raw/guide.md",
                "title": "Guide",
                "metadata": {"kb_id": "kb-a", "tenant_id": "tenant-a"},
            }
        ],
    }
    (parsed_dir / "doc-1.json").write_text(json.dumps(artifact), encoding="utf-8")

    hybrid = HybridRetriever(
        repository=InMemoryVectorRepository(),
        embedding_client=FakeEmbeddingClient(),
    )

    count = hybrid.bootstrap_from_parsed_dir(parsed_dir)
    results = hybrid.bm25_index.search(
        "vacation", top_k=5, allowed_doc_ids={"doc-1:0"}
    )

    assert count == 1
    assert len(results) == 1
    assert results[0][0] == "doc-1:0"


@pytest.mark.asyncio
async def test_hybrid_retriever_respects_kb_and_tenant_scope(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("RAG_HYBRID_SEARCH_ENABLED", "true")
    repository = InMemoryVectorRepository()
    repository.upsert(
        Document(
            doc_id="doc-a",
            source_path="data/raw/a.md",
            title="A",
            content="...",
            metadata={"kb_id": "kb-a", "tenant_id": "tenant-a"},
        ),
        [
            Chunk(
                chunk_id="chunk-a",
                doc_id="doc-a",
                chunk_index=0,
                text="PTO carryover rule",
                source_path="data/raw/a.md",
                title="A",
                metadata={"kb_id": "kb-a", "tenant_id": "tenant-a"},
            )
        ],
        [[0.0, 0.0]],
    )
    repository.upsert(
        Document(
            doc_id="doc-b",
            source_path="data/raw/b.md",
            title="B",
            content="...",
            metadata={"kb_id": "kb-b", "tenant_id": "tenant-b"},
        ),
        [
            Chunk(
                chunk_id="chunk-b",
                doc_id="doc-b",
                chunk_index=0,
                text="PTO carryover rule",
                source_path="data/raw/b.md",
                title="B",
                metadata={"kb_id": "kb-b", "tenant_id": "tenant-b"},
            )
        ],
        [[0.0, 0.0]],
    )
    hybrid = HybridRetriever(
        repository=repository,
        embedding_client=FakeEmbeddingClient(),
    )
    hybrid.index_chunks(
        [
            Chunk(
                chunk_id="chunk-a",
                doc_id="doc-a",
                chunk_index=0,
                text="PTO carryover rule",
                source_path="data/raw/a.md",
                title="A",
                metadata={"kb_id": "kb-a", "tenant_id": "tenant-a"},
            ),
            Chunk(
                chunk_id="chunk-b",
                doc_id="doc-b",
                chunk_index=0,
                text="PTO carryover rule",
                source_path="data/raw/b.md",
                title="B",
                metadata={"kb_id": "kb-b", "tenant_id": "tenant-b"},
            ),
        ]
    )
    pipeline = RetrievalPipeline(
        config=_build_config(tmp_path),
        repository=repository,
        embedding_client=FakeEmbeddingClient(),
        rerank_client=FakeRerankClient(),
        trace_store=TraceStore(),
        tracing_manager=TracingManager("test-service", ""),
        hybrid_retriever=hybrid,
    )

    contexts, _ = await pipeline.run(
        "pto carryover",
        2,
        kb_id="kb-a",
        tenant_id="tenant-a",
    )

    assert [context["chunk_id"] for context in contexts] == ["chunk-a"]


@pytest.mark.asyncio
async def test_ingestion_pipeline_updates_hybrid_index(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RAG_HYBRID_SEARCH_ENABLED", "true")
    monkeypatch.setenv("PARSED_OUTPUT_DIR", str(tmp_path / "parsed"))
    monkeypatch.setattr(
        "app.ingestion.pipeline.discover_files", lambda path: [tmp_path / "doc.txt"]
    )

    async def fake_parse_document(path, document_parser=None):  # noqa: ANN001, ARG001
        return "PTO carryover details"

    monkeypatch.setattr("app.ingestion.pipeline.parse_document", fake_parse_document)

    repository = InMemoryVectorRepository()
    hybrid = HybridRetriever(
        repository=repository,
        embedding_client=FakeEmbeddingClient(),
    )
    config = AppConfig(
        config_dir=tmp_path,
        settings={"chunk": {"size": 200, "overlap": 20}},
        models={"model_gateway": {"base_url": "", "api_key": ""}},
        prompts={},
    )
    pipeline = IngestionPipeline(
        config=config,
        repository=repository,
        embedding_client=FakeEmbeddingClient(),
        tracing_manager=TracingManager("test-service", ""),
        hybrid_retriever=hybrid,
        wiki_compiler=WikiCompiler(tmp_path / "wiki"),
    )

    response = await pipeline.run(str(tmp_path), kb_id="default")

    assert response.documents == 1
    assert hybrid.bm25_index.search("pto", top_k=5)
    assert (tmp_path / "wiki" / "sources").exists()
    artifact = json.loads(next((tmp_path / "parsed").glob("*.json")).read_text(encoding="utf-8"))
    assert artifact["document"]["metadata"]["doc_type"] == "document"
    assert artifact["document"]["metadata"]["section_count"] == 1
    assert artifact["chunks"][0]["metadata"]["parent_chunk_id"].startswith("doc-")
    assert artifact["chunks"][0]["metadata"]["section_path"] == ["doc"]


@pytest.mark.asyncio
async def test_retrieval_pipeline_records_query_expansion_metadata(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("RAG_HYBRID_SEARCH_ENABLED", "true")
    monkeypatch.setenv("RAG_HYBRID_VECTOR_WEIGHT", "0.1")
    monkeypatch.setenv("RAG_HYBRID_BM25_WEIGHT", "0.9")
    repository = InMemoryVectorRepository()
    document = Document(
        doc_id="doc-2",
        source_path="data/raw/handbook.md",
        title="Handbook",
        content="...",
        metadata={"kb_id": "default"},
    )
    chunks = [
        Chunk(
            chunk_id="carryover",
            doc_id="doc-2",
            chunk_index=0,
            text="PTO carryover rules",
            source_path="data/raw/handbook.md",
            title="Handbook",
            metadata={"kb_id": "default"},
        ),
        Chunk(
            chunk_id="vacation",
            doc_id="doc-2",
            chunk_index=1,
            text="Vacation policy overview",
            source_path="data/raw/handbook.md",
            title="Handbook",
            metadata={"kb_id": "default"},
        ),
    ]
    repository.upsert(document, chunks, [[0.0, 0.0], [1.0, 0.0]])
    hybrid = HybridRetriever(
        repository=repository,
        embedding_client=FakeEmbeddingClient(),
    )
    hybrid.index_chunks(chunks)
    query_rewriter = QueryRewriter(
        generation_client=FakeGenerationClient(),
        config=QueryRewriterConfig(
            enable_rewrite=True,
            enable_multi_query=True,
            multi_query_count=2,
            enable_hyde=True,
        ),
    )
    trace_store = TraceStore()
    pipeline = RetrievalPipeline(
        config=_build_config(tmp_path),
        repository=repository,
        embedding_client=FakeEmbeddingClient(),
        rerank_client=FakeRerankClient(),
        trace_store=trace_store,
        tracing_manager=TracingManager("test-service", ""),
        query_rewriter=query_rewriter,
        hybrid_retriever=hybrid,
    )

    contexts, trace = await pipeline.run("vacation policy", 2)
    record = trace_store.get(trace["trace_id"])

    assert [context["chunk_id"] for context in contexts] == ["carryover", "vacation"]
    assert record is not None
    assert record.rewritten_query == "vacation carryover"
    assert record.expanded_queries == [
        "vacation carryover",
        "PTO carryover",
        "leave rollover",
    ]
    assert record.hyde_query == "internal PTO carryover handbook section"
