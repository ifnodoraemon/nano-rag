import pytest

from app.core.config import AppConfig
from app.core.tracing import TraceStore, TracingManager
from app.model_client.embeddings import EmbeddingClient
from app.model_client.rerank import RerankClient
from app.retrieval.context_builder import build_contexts
from app.retrieval.pipeline import RetrievalPipeline
from app.schemas.chunk import Chunk
from app.schemas.document import Document
from app.vectorstore.repository import InMemoryVectorRepository, SearchHit


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
async def test_retrieval_pipeline_scopes_results_by_kb() -> None:
    repository = InMemoryVectorRepository()
    repository.upsert(
        Document(doc_id="doc-a", source_path="/tmp/a.txt", title="A", content="...", metadata={"kb_id": "kb-a"}),
        [
            Chunk(
                chunk_id="c1",
                doc_id="doc-a",
                chunk_index=0,
                text="aaa",
                source_path="/tmp/a.txt",
                title="A",
                metadata={"kb_id": "kb-a"},
            )
        ],
        [[3.0, 4.0]],
    )
    repository.upsert(
        Document(doc_id="doc-b", source_path="/tmp/b.txt", title="B", content="...", metadata={"kb_id": "kb-b"}),
        [
            Chunk(
                chunk_id="c2",
                doc_id="doc-b",
                chunk_index=0,
                text="aaaaaa",
                source_path="/tmp/b.txt",
                title="B",
                metadata={"kb_id": "kb-b"},
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

    contexts, trace = await pipeline.run("aaa", 2, kb_id="kb-a")

    assert len(contexts) == 1
    assert contexts[0]["chunk_id"] == "c1"
    assert trace["kb_id"] == "kb-a"


@pytest.mark.asyncio
async def test_retrieval_pipeline_promotes_parent_section_context() -> None:
    repository = InMemoryVectorRepository()
    document = Document(
        doc_id="doc-parent",
        source_path="/tmp/policy.txt",
        title="Policy",
        content="...",
        metadata={"kb_id": "default"},
    )
    parent_text = "Leave Policy section. Carryover is allowed up to 5 days. Applies to full-time employees."
    chunks = [
        Chunk(
            chunk_id="c1",
            doc_id="doc-parent",
            chunk_index=0,
            text="Carryover is allowed up to 5 days.",
            source_path="/tmp/policy.txt",
            title="Policy",
            metadata={
                "kb_id": "default",
                "parent_chunk_id": "doc-parent:parent:0",
                "parent_text": parent_text,
                "section_path": ["Policy", "Leave Policy"],
                "doc_type": "policy",
            },
        ),
        Chunk(
            chunk_id="c2",
            doc_id="doc-parent",
            chunk_index=1,
            text="Applies to full-time employees.",
            source_path="/tmp/policy.txt",
            title="Policy",
            metadata={
                "kb_id": "default",
                "parent_chunk_id": "doc-parent:parent:0",
                "parent_text": parent_text,
                "section_path": ["Policy", "Leave Policy"],
                "doc_type": "policy",
            },
        ),
    ]
    repository.upsert(document, chunks, [[8.0, 4.0], [7.0, 3.0]])
    config = AppConfig(
        config_dir=None,  # type: ignore[arg-type]
        settings={"retrieval": {"top_k": 2, "rerank_top_k": 2, "final_contexts": 2}},
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

    contexts, _ = await pipeline.run("carryover policy", 2)

    assert len(contexts) == 1
    assert contexts[0]["parent_chunk_id"] == "doc-parent:parent:0"
    assert contexts[0]["text"] == parent_text
    assert contexts[0]["title"] == "Policy > Leave Policy"
    assert contexts[0]["supporting_chunk_id"] in {"c1", "c2"}


def test_context_builder_uses_child_text_when_parent_preview_is_truncated() -> None:
    hit = SearchHit(
        chunk=Chunk(
            chunk_id="table:8",
            doc_id="doc-table",
            chunk_index=8,
            text="| 中卫市 | 中宁县 | Ⅲ | 31800 | 喊叫水乡、徐套乡 |",
            source_path="/tmp/table.pdf",
            title="Price Table",
            metadata={
                "kb_id": "default",
                "parent_chunk_id": "doc-table:parent:1",
                "parent_text": "| 银川市 | ...",
                "section_path": ["Attachment", "Price Table"],
                "chunk_kind": "child",
                "child_chunk_index": 8,
            },
        ),
        score=0.9,
    )

    contexts = build_contexts([hit], limit=1)

    assert contexts[0]["text"] == "| 中卫市 | 中宁县 | Ⅲ | 31800 | 喊叫水乡、徐套乡 |"
    assert "_dedupe_key" not in contexts[0]


@pytest.mark.asyncio
async def test_retrieval_pipeline_applies_explicit_metadata_filters() -> None:
    repository = InMemoryVectorRepository()
    repository.upsert(
        Document(
            doc_id="policy-doc",
            source_path="/tmp/policy.md",
            title="Policy",
            content="...",
            metadata={"kb_id": "default"},
        ),
        [
            Chunk(
                chunk_id="policy-1",
                doc_id="policy-doc",
                chunk_index=0,
                text="Policy carryover rules",
                source_path="/tmp/policy.md",
                title="Policy",
                metadata={"kb_id": "default", "doc_type": "policy", "effective_date": "2026-01-15"},
            )
        ],
        [[8.0, 4.0]],
    )
    repository.upsert(
        Document(
            doc_id="faq-doc",
            source_path="/tmp/faq.md",
            title="FAQ",
            content="...",
            metadata={"kb_id": "default"},
        ),
        [
            Chunk(
                chunk_id="faq-1",
                doc_id="faq-doc",
                chunk_index=0,
                text="FAQ carryover answer",
                source_path="/tmp/faq.md",
                title="FAQ",
                metadata={"kb_id": "default", "doc_type": "faq", "effective_date": "2026-01-15"},
            )
        ],
        [[8.0, 4.0]],
    )
    config = AppConfig(
        config_dir=None,  # type: ignore[arg-type]
        settings={"retrieval": {"top_k": 2, "rerank_top_k": 2, "final_contexts": 2}},
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

    contexts, trace = await pipeline.run(
        "carryover rules",
        2,
        metadata_filters={"doc_types": ["policy"]},
    )

    assert [context["chunk_id"] for context in contexts] == ["policy-1"]
    assert trace["retrieval_params"]["metadata_filters"] == {"doc_types": ["policy"]}


@pytest.mark.asyncio
async def test_retrieval_pipeline_infers_doc_type_and_year_filters_from_query() -> None:
    repository = InMemoryVectorRepository()
    repository.upsert(
        Document(
            doc_id="policy-2025",
            source_path="/tmp/policy-2025.md",
            title="Policy 2025",
            content="...",
            metadata={"kb_id": "default"},
        ),
        [
            Chunk(
                chunk_id="policy-2025:0",
                doc_id="policy-2025",
                chunk_index=0,
                text="2025 carryover policy",
                source_path="/tmp/policy-2025.md",
                title="Policy 2025",
                metadata={"kb_id": "default", "doc_type": "policy", "effective_date": "2025-01-15"},
            )
        ],
        [[8.0, 4.0]],
    )
    repository.upsert(
        Document(
            doc_id="policy-2026",
            source_path="/tmp/policy-2026.md",
            title="Policy 2026",
            content="...",
            metadata={"kb_id": "default"},
        ),
        [
            Chunk(
                chunk_id="policy-2026:0",
                doc_id="policy-2026",
                chunk_index=0,
                text="2026 carryover policy",
                source_path="/tmp/policy-2026.md",
                title="Policy 2026",
                metadata={"kb_id": "default", "doc_type": "policy", "effective_date": "2026-02-01"},
            )
        ],
        [[8.0, 4.0]],
    )
    config = AppConfig(
        config_dir=None,  # type: ignore[arg-type]
        settings={"retrieval": {"top_k": 2, "rerank_top_k": 2, "final_contexts": 2}},
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

    contexts, trace = await pipeline.run("2026 policy carryover", 2)

    assert [context["chunk_id"] for context in contexts] == ["policy-2026:0"]
    assert trace["retrieval_params"]["metadata_filters"] == {
        "doc_types": ["policy"],
        "effective_date_from": "2026-01-01",
        "effective_date_to": "2026-12-31",
    }


@pytest.mark.asyncio
async def test_retrieval_pipeline_prefers_latest_effective_version_in_contexts() -> None:
    repository = InMemoryVectorRepository()
    repository.upsert(
        Document(
            doc_id="policy-old",
            source_path="/tmp/leave-policy-2025.md",
            title="Leave Policy 2025",
            content="...",
            metadata={"kb_id": "default"},
        ),
        [
            Chunk(
                chunk_id="policy-old:0",
                doc_id="policy-old",
                chunk_index=0,
                text="Carryover is allowed up to 3 days.",
                source_path="/tmp/leave-policy-2025.md",
                title="Leave Policy 2025",
                metadata={
                    "kb_id": "default",
                    "source_key": "leave policy",
                    "section_path": ["Handbook", "Carryover"],
                    "section_path_text": "Handbook > Carryover",
                    "effective_date": "2025-01-01",
                    "version": "v1.0",
                },
            )
        ],
        [[8.0, 4.0]],
    )
    repository.upsert(
        Document(
            doc_id="policy-new",
            source_path="/tmp/leave-policy-2026.md",
            title="Leave Policy 2026",
            content="...",
            metadata={"kb_id": "default"},
        ),
        [
            Chunk(
                chunk_id="policy-new:0",
                doc_id="policy-new",
                chunk_index=0,
                text="Carryover is allowed up to 5 days.",
                source_path="/tmp/leave-policy-2026.md",
                title="Leave Policy 2026",
                metadata={
                    "kb_id": "default",
                    "source_key": "leave policy",
                    "section_path": ["Handbook", "Carryover"],
                    "section_path_text": "Handbook > Carryover",
                    "effective_date": "2026-01-01",
                    "version": "v2.0",
                },
            )
        ],
        [[8.0, 4.0]],
    )
    config = AppConfig(
        config_dir=None,  # type: ignore[arg-type]
        settings={"retrieval": {"top_k": 2, "rerank_top_k": 2, "final_contexts": 2}},
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

    contexts, trace = await pipeline.run("carryover policy", 2)

    assert [context["chunk_id"] for context in contexts] == ["policy-new:0"]
    assert contexts[0]["is_latest_version"] is True
    assert contexts[0]["freshness_tier"] == "primary"
    assert contexts[0]["evidence_role"] == "primary"
    assert trace["freshness_ranked_chunk_ids"] == ["policy-new:0"]


@pytest.mark.asyncio
async def test_retrieval_pipeline_orders_primary_supporting_and_conflicting_contexts() -> None:
    repository = InMemoryVectorRepository()
    repository.upsert(
        Document(
            doc_id="doc",
            source_path="/tmp/doc.md",
            title="Doc",
            content="...",
            metadata={"kb_id": "default"},
        ),
        [
            Chunk(
                chunk_id="supporting-1",
                doc_id="doc",
                chunk_index=0,
                text="Manager approval is required.",
                source_path="/tmp/doc.md",
                title="Guide",
                metadata={"kb_id": "default"},
            ),
            Chunk(
                chunk_id="primary-1",
                doc_id="doc",
                chunk_index=1,
                text="Carryover is allowed up to 5 days.",
                source_path="/tmp/doc.md",
                title="Leave Policy 2026",
                metadata={
                    "kb_id": "default",
                    "source_key": "leave policy",
                    "section_path": ["Handbook", "Carryover"],
                    "section_path_text": "Handbook > Carryover",
                    "effective_date": "2026-01-01",
                    "version": "v2.0",
                },
            ),
            Chunk(
                chunk_id="conflict-1",
                doc_id="doc",
                chunk_index=2,
                text="Older policy says carryover is not allowed.",
                source_path="/tmp/doc.md",
                title="Leave Policy Conflict",
                metadata={
                    "kb_id": "default",
                    "wiki_kind": "topic",
                    "wiki_status": "conflicting",
                },
            ),
        ],
        [[6.0, 1.0], [5.0, 1.0], [4.0, 1.0]],
    )
    config = AppConfig(
        config_dir=None,  # type: ignore[arg-type]
        settings={"retrieval": {"top_k": 3, "rerank_top_k": 3, "final_contexts": 3}},
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

    contexts, _ = await pipeline.run("carryover", 3)

    assert [context["chunk_id"] for context in contexts] == [
        "primary-1",
        "supporting-1",
        "conflict-1",
    ]
    assert [context["evidence_role"] for context in contexts] == [
        "primary",
        "supporting",
        "conflicting",
    ]
