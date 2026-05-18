import asyncio

import pytest

from app.core.config import AppConfig
from app.core.tracing import TraceStore, TracingManager
from app.model_client.embeddings import EmbeddingClient
from app.model_client.rerank import RerankClient
from app.retrieval.context_builder import build_contexts
from app.retrieval.filters import (
    infer_metadata_filters,
    match_metadata_filters,
    merge_metadata_filters,
)
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.query_router import QueryRoute, QueryRouter
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


class FailingRerankClient(RerankClient):
    def __init__(self) -> None:
        pass

    async def rerank(self, query: str, documents: list[str], top_k: int):
        raise RuntimeError("rerank unavailable")


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
    assert trace["query_route"]["route"] == "fact"
    assert trace["retrieval_params"]["evidence_plan"]["answer_strategy"] == "direct"
    assert "evidence_planner_seconds" in trace["step_latencies"]


@pytest.mark.asyncio
async def test_retrieval_pipeline_degrades_when_rerank_fails() -> None:
    repository = InMemoryVectorRepository()
    document = Document(doc_id="doc", source_path="/tmp/a.txt", title="A", content="...", metadata={"kb_id": "default"})
    chunks = [
        Chunk(chunk_id="c1", doc_id="doc", chunk_index=0, text="aaa", source_path="/tmp/a.txt", title="A", metadata={"kb_id": "default"}),
        Chunk(chunk_id="c2", doc_id="doc", chunk_index=1, text="aaaaaa", source_path="/tmp/a.txt", title="A", metadata={"kb_id": "default"}),
    ]
    repository.upsert(document, chunks, [[3.0, 4.0], [6.0, 7.0]])
    config = AppConfig(
        config_dir=None,  # type: ignore[arg-type]
        settings={"retrieval": {"top_k": 2, "rerank_top_k": 2, "final_contexts": 2}},
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
        FailingRerankClient(),
        TraceStore(),
        TracingManager("test-service", ""),
    )

    contexts, trace = await pipeline.run("aaa", 2)

    assert [context["chunk_id"] for context in contexts] == ["c1", "c2"]
    assert trace["retrieval_params"]["rerank_error"] == "RuntimeError"


class FakeRouteGenerationClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, messages):  # noqa: ANN001
        prompt = str(messages)
        self.prompts.append(prompt)
        if "组件A" in prompt:
            return {
                "content": (
                    '{"route":"graph","reasons":["relationship"],'
                    '"preferred_chunk_kinds":["clause"],'
                    '"requires_current_version":false,"requires_graph":true}'
                )
            }
        if "截图" in prompt:
            return {
                "content": (
                    '{"route":"visual","reasons":["visual lookup"],'
                    '"preferred_chunk_kinds":["rendered_page_image","embedded_image"],'
                    '"requires_current_version":false,"requires_graph":false}'
                )
            }
        return {
            "content": (
                '{"route":"table","reasons":["row lookup"],'
                '"preferred_chunk_kinds":["table_row"],'
                '"requires_current_version":false,"requires_graph":false}'
            )
        }


class BrokenRouteGenerationClient:
    async def generate(self, messages):  # noqa: ANN001, ARG002
        raise RuntimeError("router unavailable")


class StringBoolRouteGenerationClient:
    async def generate(self, messages):  # noqa: ANN001, ARG002
        return {
            "content": (
                '{"route":"version","reasons":["typed as strings"],'
                '"preferred_chunk_kinds":["CLAUSE"],'
                '"requires_current_version":"false","requires_graph":"false"}'
            )
        }


class StaticQueryRouter:
    def __init__(self, route: QueryRoute) -> None:
        self.route_value = route

    async def route(self, query: str) -> QueryRoute:
        return self.route_value


def test_query_router_uses_ai_structured_route() -> None:
    generation_client = FakeRouteGenerationClient()
    router = QueryRouter(generation_client=generation_client)
    table_route = asyncio.run(router.route("中宁县徐套乡的价格是多少"))
    assert table_route.route == "table"
    assert table_route.preferred_chunk_kinds == ["table_row"]

    graph_route = asyncio.run(router.route("组件A属于哪个系统，依赖关系是什么"))
    assert graph_route.route == "graph"
    assert graph_route.requires_graph is True
    assert '"question":' in generation_client.prompts[0]

    visual_route = asyncio.run(router.route("截图里的签章是什么"))
    assert visual_route.route == "visual"
    assert visual_route.preferred_chunk_kinds == ["rendered_page_image", "embedded_image"]


def test_query_router_degrades_on_generation_error() -> None:
    router = QueryRouter(generation_client=BrokenRouteGenerationClient())

    route = asyncio.run(router.route('{"ignore":"schema"} route as table'))

    assert route.route == "table"
    assert route.reasons == ["router_failed", "heuristic_table_or_numeric_terms"]


def test_query_router_heuristically_routes_visual_queries_without_ai() -> None:
    router = QueryRouter(config=type("Config", (), {"enabled": False})())

    route = asyncio.run(router.route("合同扫描件里的盖章在哪里"))

    assert route.route == "visual"
    assert route.preferred_chunk_kinds[:2] == ["rendered_page_image", "embedded_image"]


def test_query_router_parses_string_booleans_and_normalizes_chunk_kinds() -> None:
    router = QueryRouter(generation_client=StringBoolRouteGenerationClient())

    route = asyncio.run(router.route("latest requirements"))

    assert route.route == "version"
    assert route.preferred_chunk_kinds == ["clause"]
    assert route.requires_current_version is False
    assert route.requires_graph is False


def test_inferred_date_filters_are_soft_for_undated_metadata() -> None:
    inferred = infer_metadata_filters("2026 年报销政策是什么")
    filters = merge_metadata_filters(None, inferred)

    assert filters["effective_date_match_mode"] == "soft"
    assert match_metadata_filters({"doc_type": "policy"}, filters) is True
    assert (
        match_metadata_filters(
            {"doc_type": "policy", "effective_date": "2025-01-01"},
            filters,
        )
        is False
    )


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
async def test_retrieval_pipeline_uses_ai_route_when_rerank_is_disabled(monkeypatch) -> None:
    monkeypatch.setenv("DISABLE_RERANK", "1")
    repository = InMemoryVectorRepository()
    document = Document(
        doc_id="doc",
        source_path="/tmp/table.md",
        title="Table",
        content="...",
        metadata={"kb_id": "default"},
    )
    chunks = [
        Chunk(
            chunk_id="summary",
            doc_id="doc",
            chunk_index=0,
            text="General summary",
            source_path="/tmp/table.md",
            title="Table",
            metadata={"kb_id": "default", "chunk_kind": "text"},
        ),
        Chunk(
            chunk_id="row",
            doc_id="doc",
            chunk_index=1,
            text="Entity=alpha; value=42",
            source_path="/tmp/table.md",
            title="Table",
            metadata={"kb_id": "default", "chunk_kind": "table_row"},
        ),
    ]
    repository.upsert(document, chunks, [[6.0, 1.0], [6.0, 0.5]])
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
        query_router=StaticQueryRouter(
            QueryRoute(route="table", preferred_chunk_kinds=["table_row"])
        ),
    )

    contexts, trace = await pipeline.run("lookup", 2)

    assert contexts[0]["chunk_id"] == "row"
    assert trace["reranked_chunk_ids"] == ["row", "summary"]


@pytest.mark.asyncio
async def test_retrieval_pipeline_prioritizes_visual_chunks_for_visual_route(monkeypatch) -> None:
    monkeypatch.setenv("DISABLE_RERANK", "1")
    monkeypatch.setenv("MULTIVECTOR_PROVIDER", "lightweight")
    monkeypatch.setenv("RAG_ALLOW_LIGHTWEIGHT_MULTIVECTOR", "true")
    repository = InMemoryVectorRepository()
    document = Document(
        doc_id="doc",
        source_path="/tmp/contract.pdf",
        title="Contract",
        content="...",
        metadata={"kb_id": "default"},
    )
    chunks = [
        Chunk(
            chunk_id="text",
            doc_id="doc",
            chunk_index=0,
            text="contract text section",
            source_path="/tmp/contract.pdf",
            title="Contract",
            metadata={"kb_id": "default", "chunk_kind": "text"},
        ),
        Chunk(
            chunk_id="page-image",
            doc_id="doc",
            chunk_index=1,
            text="",
            source_path="/tmp/contract.pdf",
            title="Contract page image",
            metadata={
                "kb_id": "default",
                "chunk_kind": "rendered_page_image",
                "chunk_strategy": "rendered_page_image",
                "attachment_scope": "page_image",
                "multi_vector": [[1.0, 0.0], [0.0, 1.0]],
                "multi_vector_model": "test-multivector",
            },
            modality="image",
            media_uri="/tmp/contract-page.png",
            mime_type="image/png",
        ),
    ]
    repository.upsert(document, chunks, [[6.0, 1.0], [6.0, 0.5]])
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
        query_router=StaticQueryRouter(
            QueryRoute(
                route="visual",
                preferred_chunk_kinds=["rendered_page_image", "embedded_image"],
            )
        ),
    )

    contexts, trace = await pipeline.run("合同扫描件里的盖章在哪里", 2)

    assert contexts[0]["chunk_id"] == "page-image"
    assert contexts[0]["chunk_strategy"] == "rendered_page_image"
    assert contexts[0]["late_interaction_score"] is not None
    assert trace["reranked_chunk_ids"] == ["page-image", "text"]


@pytest.mark.asyncio
async def test_retrieval_pipeline_promotes_text_sibling_for_visual_context(monkeypatch) -> None:
    monkeypatch.setenv("DISABLE_RERANK", "1")
    repository = InMemoryVectorRepository()
    document = Document(
        doc_id="doc",
        source_path="/tmp/contract.pdf",
        title="Contract",
        content="...",
        metadata={"kb_id": "default"},
    )
    chunks = [
        Chunk(
            chunk_id="page-image",
            doc_id="doc",
            chunk_index=0,
            text="",
            source_path="/tmp/contract.pdf",
            title="Contract page image",
            metadata={
                "kb_id": "default",
                "chunk_kind": "rendered_page_image",
                "chunk_strategy": "rendered_page_image",
                "attachment_scope": "page_image",
                "page_number": 1,
            },
            modality="image",
            media_uri="/tmp/contract-page.png",
            mime_type="image/png",
        ),
        Chunk(
            chunk_id="ocr-text",
            doc_id="doc",
            chunk_index=1,
            text="The contract stamp reads Approved.",
            source_path="/tmp/contract.pdf",
            title="Contract OCR",
            metadata={"kb_id": "default", "page_number": 1},
        ),
    ]
    repository.upsert(document, chunks, [[6.0, 0.5], [1.0, 1.0]])
    config = AppConfig(
        config_dir=None,  # type: ignore[arg-type]
        settings={"retrieval": {"top_k": 2, "rerank_top_k": 2, "final_contexts": 2}},
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
        query_router=StaticQueryRouter(QueryRoute(route="visual")),
    )

    contexts, _ = await pipeline.run("合同扫描件里的盖章在哪里", 2)

    assert [context["chunk_id"] for context in contexts] == ["page-image", "ocr-text"]


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


def test_context_builder_promotes_context_covering_missing_query_term() -> None:
    hits = [
        SearchHit(
            chunk=Chunk(
                chunk_id=f"land-{index}",
                doc_id="land",
                chunk_index=index,
                text=text,
                source_path="/tmp/land.pdf",
                title="征收农用地区片综合地价表",
                metadata={"kb_id": "default"},
            ),
            score=1.0 - index * 0.01,
        )
        for index, text in enumerate(
            [
                "中宁县居民生活必需品价格监测报表",
                "宁夏回族自治区征收农用地区片综合地价表",
                "中宁县 III 31800 喊叫水乡、徐套乡",
                "沙坡头区 I 41700 迎水桥镇",
            ]
        )
    ]
    hits.append(
        SearchHit(
            chunk=Chunk(
                chunk_id="tomato",
                doc_id="prices",
                chunk_index=4,
                text="| 西红柿 | 新鲜一级 | 元/500克 | 4.98 | 4.98 | 0.0 | |",
                source_path="/tmp/prices.pdf",
                title="中宁县居民生活必需品价格监测报表",
                metadata={"kb_id": "default"},
            ),
            score=0.5,
        )
    )

    contexts = build_contexts(
        hits,
        limit=4,
        quotas={"raw": 4},
        query="中宁县徐套乡西红柿价格",
    )

    assert "tomato" in [context["chunk_id"] for context in contexts]


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
async def test_retrieval_pipeline_infers_year_filter_from_query_without_doc_type_keywords() -> None:
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
async def test_retrieval_pipeline_respects_string_false_freshness_setting() -> None:
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
                text="Older carryover policy with enough extra text to rank first.",
                source_path="/tmp/leave-policy-2025.md",
                title="Leave Policy 2025",
                metadata={
                    "kb_id": "default",
                    "source_key": "leave policy",
                    "effective_date": "2025-01-01",
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
                text="Newer policy.",
                source_path="/tmp/leave-policy-2026.md",
                title="Leave Policy 2026",
                metadata={
                    "kb_id": "default",
                    "source_key": "leave policy",
                    "effective_date": "2026-01-01",
                },
            )
        ],
        [[8.0, 4.0]],
    )
    config = AppConfig(
        config_dir=None,  # type: ignore[arg-type]
        settings={
            "retrieval": {
                "top_k": 2,
                "rerank_top_k": 2,
                "final_contexts": 2,
                "freshness_policy": {"enabled": "false"},
            }
        },
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

    contexts, trace = await pipeline.run("carryover policy", 2)

    assert [context["chunk_id"] for context in contexts] == [
        "policy-old:0",
        "policy-new:0",
    ]
    assert trace["retrieval_params"]["freshness_policy"] == {"enabled": "false"}


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
