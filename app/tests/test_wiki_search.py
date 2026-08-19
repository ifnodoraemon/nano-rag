import json

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


def test_wiki_search_filters_pages_by_kb(tmp_path) -> None:
    wiki_compiler = WikiCompiler(tmp_path / "wiki")
    wiki_compiler.upsert_document(
        Document(
            doc_id="default-doc",
            source_path="uploads/default/policy.md",
            title="Default Policy",
            content="# Policy\n\nDefault vacation rules.",
            metadata={"kb_id": "default"},
        ),
        [
            Chunk(
                chunk_id="default:0",
                doc_id="default-doc",
                chunk_index=0,
                text="Default vacation rules.",
                source_path="uploads/default/policy.md",
                metadata={"kb_id": "default"},
            )
        ],
    )
    wiki_compiler.upsert_document(
        Document(
            doc_id="other-doc",
            source_path="uploads/other/policy.md",
            title="Other Policy",
            content="# Policy\n\nOther vacation rules.",
            metadata={"kb_id": "other"},
        ),
        [
            Chunk(
                chunk_id="other:0",
                doc_id="other-doc",
                chunk_index=0,
                text="Other vacation rules.",
                source_path="uploads/other/policy.md",
                metadata={"kb_id": "other"},
            )
        ],
    )
    wiki_searcher = WikiSearcher(tmp_path / "wiki")

    default_hits = wiki_searcher.search("vacation", top_k=10, kb_id="default")
    other_hits = wiki_searcher.search("vacation", top_k=10, kb_id="other")

    assert default_hits
    assert other_hits
    assert all(hit.chunk.metadata.get("kb_id") == "default" for hit in default_hits)
    assert all(hit.chunk.metadata.get("kb_id") == "other" for hit in other_hits)


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


def _wiki_doc(doc_id: str, source_key: str, version: str, effective_date: str) -> Document:
    return Document(
        doc_id=doc_id,
        source_path=f"uploads/default/{doc_id}.md",
        title=f"{source_key.title()} {version}",
        content=f"# {source_key.title()}\n\n{version} rules apply.",
        metadata={
            "kb_id": "default",
            "source_key": source_key,
            "doc_type": "document",
            "effective_date": effective_date,
            "version": version,
            "source_content_hash": f"hash-{doc_id}",
        },
    )


def _wiki_chunk(doc_id: str) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}:0",
        doc_id=doc_id,
        chunk_index=0,
        text=f"{doc_id} body",
        source_path=f"uploads/default/{doc_id}.md",
        title=doc_id,
        metadata={"kb_id": "default"},
    )


def test_wiki_source_page_frontmatter_carries_version_fields(tmp_path) -> None:
    compiler = WikiCompiler(tmp_path / "wiki")
    compiler.upsert_document(_wiki_doc("doc-a", "travel policy", "v1", None), [_wiki_chunk("doc-a")])

    metadata, _ = WikiCompiler.read_frontmatter(tmp_path / "wiki" / "sources" / "doc-a.md")
    assert metadata["content_hash"] == "hash-doc-a"
    assert metadata["is_latest_version"] is True
    assert metadata["superseded_by"] is None
    assert metadata["version"] == "v1"
    assert metadata["source_key"] == "travel policy"


def test_wiki_version_chain_marks_superseded_pages(tmp_path) -> None:
    compiler = WikiCompiler(tmp_path / "wiki")
    compiler.upsert_document(_wiki_doc("doc-v1", "employee manual", "v1", "2024-01-01"), [_wiki_chunk("doc-v1")])
    compiler.upsert_document(_wiki_doc("doc-v2", "employee manual", "v2", "2025-06-01"), [_wiki_chunk("doc-v2")])
    compiler.upsert_document(_wiki_doc("doc-solo", "travel policy", "v1", None), [_wiki_chunk("doc-solo")])

    old, _ = WikiCompiler.read_frontmatter(tmp_path / "wiki" / "sources" / "doc-v1.md")
    new, _ = WikiCompiler.read_frontmatter(tmp_path / "wiki" / "sources" / "doc-v2.md")
    solo, _ = WikiCompiler.read_frontmatter(tmp_path / "wiki" / "sources" / "doc-solo.md")

    assert new["is_latest_version"] is True
    assert new["superseded_by"] is None
    assert old["is_latest_version"] is False
    assert old["superseded_by"] == "doc-v2"
    assert solo["is_latest_version"] is True
    assert solo["superseded_by"] is None


def test_wiki_version_chain_prefers_date_then_version(tmp_path) -> None:
    compiler = WikiCompiler(tmp_path / "wiki")
    # Two docs, same source_key, same version string, different dates.
    compiler.upsert_document(_wiki_doc("doc-d1", "hr policy", "v1", "2023-01-01"), [_wiki_chunk("doc-d1")])
    compiler.upsert_document(_wiki_doc("doc-d2", "hr policy", "v1", "2024-05-05"), [_wiki_chunk("doc-d2")])

    d1, _ = WikiCompiler.read_frontmatter(tmp_path / "wiki" / "sources" / "doc-d1.md")
    d2, _ = WikiCompiler.read_frontmatter(tmp_path / "wiki" / "sources" / "doc-d2.md")
    assert d2["is_latest_version"] is True
    assert d1["is_latest_version"] is False
    assert d1["superseded_by"] == "doc-d2"


def test_wiki_version_chain_recomputed_after_remove(tmp_path) -> None:
    compiler = WikiCompiler(tmp_path / "wiki")
    compiler.upsert_document(_wiki_doc("doc-v1", "employee manual", "v1", "2024-01-01"), [_wiki_chunk("doc-v1")])
    compiler.upsert_document(_wiki_doc("doc-v2", "employee manual", "v2", "2025-06-01"), [_wiki_chunk("doc-v2")])

    compiler.remove_document("doc-v2")

    d1, _ = WikiCompiler.read_frontmatter(tmp_path / "wiki" / "sources" / "doc-v1.md")
    assert d1["is_latest_version"] is True
    assert d1["superseded_by"] is None


def test_wiki_search_hit_metadata_includes_version_fields(tmp_path) -> None:
    compiler = WikiCompiler(tmp_path / "wiki")
    compiler.upsert_document(_wiki_doc("doc-v1", "employee manual", "v1", "2024-01-01"), [_wiki_chunk("doc-v1")])
    compiler.upsert_document(_wiki_doc("doc-v2", "employee manual", "v2", "2025-06-01"), [_wiki_chunk("doc-v2")])
    searcher = WikiSearcher(tmp_path / "wiki")

    hits = searcher.search("employee manual rules", top_k=10, kb_id="default")
    by_id = {hit.chunk.doc_id: hit for hit in hits if hit.chunk.doc_id in {"doc-v1", "doc-v2"}}

    assert by_id["doc-v2"].chunk.metadata["is_latest_version"] is True
    assert by_id["doc-v2"].chunk.metadata["superseded_by"] is None
    assert by_id["doc-v1"].chunk.metadata["is_latest_version"] is False
    assert by_id["doc-v1"].chunk.metadata["superseded_by"] == "doc-v2"


def test_wiki_bootstrap_rebuilds_from_parsed_artifacts(tmp_path) -> None:
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    for doc_id, version, effective_date in [
        ("doc-v1", "v1", "2024-01-01"),
        ("doc-v2", "v2", "2025-06-01"),
    ]:
        document = _wiki_doc(doc_id, "employee manual", version, effective_date)
        (parsed_dir / f"{doc_id}.json").write_text(
            json.dumps(
                {
                    "document": document.model_dump(),
                    "chunks": [_wiki_chunk(doc_id).model_dump()],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # A fresh wiki dir (as after a container rebuild) rebuilds from artifacts.
    wiki_dir = tmp_path / "wiki"
    compiler = WikiCompiler(wiki_dir)
    count = compiler.bootstrap_from_parsed_dir(parsed_dir)

    assert count == 2
    assert (wiki_dir / "sources" / "doc-v1.md").exists()
    assert (wiki_dir / "sources" / "doc-v2.md").exists()

    old, _ = WikiCompiler.read_frontmatter(wiki_dir / "sources" / "doc-v1.md")
    new, _ = WikiCompiler.read_frontmatter(wiki_dir / "sources" / "doc-v2.md")
    assert new["is_latest_version"] is True
    assert old["is_latest_version"] is False
    assert old["superseded_by"] == "doc-v2"

    # The searcher re-indexes the rebuilt pages and can find them.
    searcher = WikiSearcher(wiki_dir)
    hits = searcher.search("employee manual rules", top_k=10, kb_id="default")
    source_hits = {
        hit.chunk.doc_id for hit in hits if hit.chunk.metadata.get("wiki_kind") == "source"
    }
    assert source_hits == {"doc-v1", "doc-v2"}

    # Bootstrap is idempotent and does not append to the ingest log.
    assert (wiki_dir / "log.md").read_text(encoding="utf-8").count("ingest") == 0
    assert compiler.bootstrap_from_parsed_dir(parsed_dir) == 2


def test_wiki_bootstrap_skips_malformed_artifacts(tmp_path) -> None:
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    (parsed_dir / "good.json").write_text(
        json.dumps(
            {
                "document": _wiki_doc("doc-a", "travel policy", "v1", None).model_dump(),
                "chunks": [_wiki_chunk("doc-a").model_dump()],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (parsed_dir / "broken.json").write_text("{not valid json", encoding="utf-8")
    (parsed_dir / "missing-doc.json").write_text(json.dumps({"chunks": []}), encoding="utf-8")

    wiki_dir = tmp_path / "wiki"
    compiler = WikiCompiler(wiki_dir)
    count = compiler.bootstrap_from_parsed_dir(parsed_dir)

    assert count == 1
    assert (wiki_dir / "sources" / "doc-a.md").exists()
    assert not list((wiki_dir / "sources").glob("broken*"))
