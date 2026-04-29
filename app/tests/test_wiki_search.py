import pytest

from app.core.tracing import TraceStore, TracingManager
from app.retrieval.pipeline import RetrievalPipeline
from app.schemas.chunk import Chunk
from app.schemas.document import Document
from app.vectorstore.repository import InMemoryVectorRepository
from app.wiki.compiler import WikiCompiler
from app.wiki.search import WikiSearcher
from app.core.config import AppConfig


class FakeEmbeddingClient:
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if "vacation" in lowered else 0.0,
                    1.0 if "expense" in lowered else 0.0,
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


def _config(tmp_path) -> AppConfig:
    return AppConfig(
        config_dir=tmp_path,
        settings={"retrieval": {"top_k": 4, "rerank_top_k": 4, "final_contexts": 2}},
        models={
            "model_gateway": {"base_url": "", "api_key": ""},
            "rerank": {"default_alias": "disabled"},
        },
        prompts={},
    )


def test_wiki_search_shared_scope_does_not_return_tenant_pages(tmp_path) -> None:
    wiki_compiler = WikiCompiler(tmp_path / "wiki")
    wiki_compiler.upsert_document(
        Document(
            doc_id="shared",
            source_path="uploads/default/__shared__/policy.md",
            title="Shared Policy",
            content="# Policy\n\nShared vacation rules.",
            metadata={"kb_id": "default", "tenant_id": None},
        ),
        [
            Chunk(
                chunk_id="shared:0",
                doc_id="shared",
                chunk_index=0,
                text="Shared vacation rules.",
                source_path="uploads/default/__shared__/policy.md",
                metadata={"kb_id": "default", "tenant_id": None},
            )
        ],
    )
    wiki_compiler.upsert_document(
        Document(
            doc_id="tenant",
            source_path="uploads/default/tenant-a/policy.md",
            title="Tenant Policy",
            content="# Policy\n\nTenant vacation rules.",
            metadata={"kb_id": "default", "tenant_id": "tenant-a"},
        ),
        [
            Chunk(
                chunk_id="tenant:0",
                doc_id="tenant",
                chunk_index=0,
                text="Tenant vacation rules.",
                source_path="uploads/default/tenant-a/policy.md",
                metadata={"kb_id": "default", "tenant_id": "tenant-a"},
            )
        ],
    )
    wiki_searcher = WikiSearcher(tmp_path / "wiki")

    shared_hits = wiki_searcher.search("vacation", top_k=10, kb_id="default")
    tenant_hits = wiki_searcher.search(
        "vacation", top_k=10, kb_id="default", tenant_id="tenant-a"
    )

    assert shared_hits
    assert tenant_hits
    assert all(hit.chunk.metadata.get("tenant_id") is None for hit in shared_hits)
    assert all(
        hit.chunk.metadata.get("tenant_id") == "tenant-a" for hit in tenant_hits
    )


@pytest.mark.asyncio
async def test_retrieval_pipeline_prefers_wiki_hits_before_raw_hits(tmp_path) -> None:
    wiki_compiler = WikiCompiler(tmp_path / "wiki")
    wiki_compiler.upsert_document(
        Document(
            doc_id="doc-1",
            source_path="data/raw/handbook.md",
            title="Employee Handbook",
            content=(
                "# Leave Policy\n\n"
                "Employees can carry over PTO into the next year subject to manager approval."
            ),
            metadata={"kb_id": "default"},
        ),
        [
            Chunk(
                chunk_id="doc-1:0",
                doc_id="doc-1",
                chunk_index=0,
                text="Employees can carry over PTO into the next year subject to manager approval.",
                source_path="data/raw/handbook.md",
                title="Employee Handbook",
                metadata={"kb_id": "default"},
            )
        ],
    )
    wiki_searcher = WikiSearcher(tmp_path / "wiki")
    repository = InMemoryVectorRepository()
    repository.upsert(
        Document(
            doc_id="raw-1",
            source_path="data/raw/expense.md",
            title="Expense Guide",
            content="...",
            metadata={"kb_id": "default"},
        ),
        [
            Chunk(
                chunk_id="raw-1:0",
                doc_id="raw-1",
                chunk_index=0,
                text="Expense reimbursements must be filed within 30 days.",
                source_path="data/raw/expense.md",
                title="Expense Guide",
                metadata={"kb_id": "default"},
            )
        ],
        [[0.0, 1.0]],
    )
    config = _config(tmp_path)
    config.settings["retrieval"]["final_contexts"] = 3
    pipeline = RetrievalPipeline(
        config=config,
        repository=repository,
        embedding_client=FakeEmbeddingClient(),
        rerank_client=FakeRerankClient(),
        trace_store=TraceStore(),
        tracing_manager=TracingManager("test-service", ""),
        wiki_searcher=wiki_searcher,
    )

    contexts, trace = await pipeline.run("carry over pto", 2)

    assert contexts[0]["chunk_id"].startswith("wiki:topic:")
    assert contexts[0]["source"] == "wiki/topics/default--leave-policy.md"
    assert "Leave Policy" in contexts[0]["text"]
    assert any(
        chunk_id.startswith("wiki:topic:") for chunk_id in trace["retrieved_chunk_ids"]
    )


@pytest.mark.asyncio
async def test_retrieval_pipeline_includes_raw_hits_when_wiki_is_insufficient(tmp_path) -> None:
    wiki_compiler = WikiCompiler(tmp_path / "wiki")
    wiki_compiler.upsert_document(
        Document(
            doc_id="doc-1",
            source_path="data/raw/handbook.md",
            title="Employee Handbook",
            content="# Leave Policy\n\nPTO carryover is allowed.",
            metadata={"kb_id": "default"},
        ),
        [
            Chunk(
                chunk_id="doc-1:0",
                doc_id="doc-1",
                chunk_index=0,
                text="PTO carryover is allowed.",
                source_path="data/raw/handbook.md",
                title="Employee Handbook",
                metadata={"kb_id": "default"},
            )
        ],
    )
    wiki_searcher = WikiSearcher(tmp_path / "wiki")
    repository = InMemoryVectorRepository()
    repository.upsert(
        Document(
            doc_id="raw-1",
            source_path="data/raw/expense.md",
            title="Expense Guide",
            content="...",
            metadata={"kb_id": "default"},
        ),
        [
            Chunk(
                chunk_id="raw-1:0",
                doc_id="raw-1",
                chunk_index=0,
                text="Expense reimbursements must be filed within 30 days.",
                source_path="data/raw/expense.md",
                title="Expense Guide",
                metadata={"kb_id": "default"},
            )
        ],
        [[0.0, 1.0]],
    )
    config = _config(tmp_path)
    config.settings["retrieval"]["final_contexts"] = 3
    pipeline = RetrievalPipeline(
        config=config,
        repository=repository,
        embedding_client=FakeEmbeddingClient(),
        rerank_client=FakeRerankClient(),
        trace_store=TraceStore(),
        tracing_manager=TracingManager("test-service", ""),
        wiki_searcher=wiki_searcher,
    )

    config.settings["retrieval"]["final_contexts"] = 4
    contexts, trace = await pipeline.run("pto expense", 4)

    assert any(context["chunk_id"].startswith("wiki:") for context in contexts)
    assert any(context["chunk_id"] == "raw-1:0" for context in contexts)
    assert "raw-1:0" in trace["retrieved_chunk_ids"]


@pytest.mark.asyncio
async def test_retrieval_pipeline_balances_topic_and_raw_contexts_with_quota(tmp_path) -> None:
    wiki_compiler = WikiCompiler(tmp_path / "wiki")
    wiki_compiler.upsert_document(
        Document(
            doc_id="doc-1",
            source_path="data/raw/handbook.md",
            title="Employee Handbook",
            content="# Leave Policy\n\nPTO carryover is allowed.\n\n## Expense Rules\n\nExpense policy summary.",
            metadata={"kb_id": "default"},
        ),
        [
            Chunk(
                chunk_id="doc-1:0",
                doc_id="doc-1",
                chunk_index=0,
                text="PTO carryover is allowed.",
                source_path="data/raw/handbook.md",
                title="Employee Handbook",
                metadata={"kb_id": "default"},
            )
        ],
    )
    wiki_searcher = WikiSearcher(tmp_path / "wiki")
    repository = InMemoryVectorRepository()
    repository.upsert(
        Document(
            doc_id="raw-1",
            source_path="data/raw/expense.md",
            title="Expense Guide",
            content="...",
            metadata={"kb_id": "default"},
        ),
        [
            Chunk(
                chunk_id="raw-1:0",
                doc_id="raw-1",
                chunk_index=0,
                text="Expense reimbursements must be filed within 30 days.",
                source_path="data/raw/expense.md",
                title="Expense Guide",
                metadata={"kb_id": "default"},
            )
        ],
        [[0.0, 1.0]],
    )
    config = _config(tmp_path)
    config.settings["retrieval"]["final_contexts"] = 2
    config.settings["retrieval"]["context_quota"] = {
        "topic": 1,
        "raw": 1,
        "source": 1,
        "index": 0,
    }
    pipeline = RetrievalPipeline(
        config=config,
        repository=repository,
        embedding_client=FakeEmbeddingClient(),
        rerank_client=FakeRerankClient(),
        trace_store=TraceStore(),
        tracing_manager=TracingManager("test-service", ""),
        wiki_searcher=wiki_searcher,
    )

    contexts, _ = await pipeline.run("pto expense", 4)

    assert len(contexts) == 2
    assert any(context["chunk_id"].startswith("wiki:topic:") for context in contexts)
    assert any(context["chunk_id"] == "raw-1:0" for context in contexts)
