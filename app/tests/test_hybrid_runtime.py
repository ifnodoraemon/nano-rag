import json

import pytest

from app.core.config import AppConfig
from app.core.exceptions import ModelGatewayError
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

    async def embed_items(self, items):  # noqa: ANN001
        return await self.embed_texts([
            "\n".join(getattr(item, "text", "") for item in batch)
            for batch in items
        ])


class FailingEmbeddingClient:
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:  # noqa: ARG002
        raise ModelGatewayError("embedding service unavailable")

    async def embed_items(self, items):  # noqa: ANN001, ARG002
        raise ModelGatewayError("embedding service unavailable")


class SelectiveFailingEmbeddingClient:
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if any("FAIL_EMBED" in text for text in texts):
            raise ModelGatewayError("embedding service unavailable")
        return [[1.0, 0.0] for _ in texts]

    async def embed_items(self, items):  # noqa: ANN001
        return await self.embed_texts([
            "\n".join(getattr(item, "text", "") for item in batch)
            for batch in items
        ])


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


class FakeImageTextParser:
    enabled = True

    def supports(self, path):  # noqa: ANN001
        return path.suffix.lower() == ".png"

    async def parse_file(self, path):  # noqa: ANN001, ARG002
        return "# Label\n\nInvoice total: 99 USD"


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


def test_hybrid_search_enabled_respects_string_false_config(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("RAG_HYBRID_SEARCH_ENABLED", raising=False)
    config = AppConfig(
        config_dir=tmp_path,
        settings={"hybrid_search": {"enabled": "false"}},
        models={"model_gateway": {"base_url": "", "api_key": ""}},
        prompts={},
    )

    assert config.hybrid_search_enabled is False


class FailingUpsertRepository(InMemoryVectorRepository):
    def __init__(self, fail_on_call: int) -> None:
        super().__init__()
        self._fail_on_call = fail_on_call
        self._upsert_calls = 0

    def upsert(
        self, document: Document, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        self._upsert_calls += 1
        if self._upsert_calls == self._fail_on_call:
            raise RuntimeError("repository upsert failed")
        super().upsert(document, chunks, embeddings)


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
            "metadata": {"kb_id": "kb-a"},
        },
        "chunks": [
            {
                "chunk_id": "doc-1:0",
                "doc_id": "doc-1",
                "chunk_index": 0,
                "text": "vacation and reimbursement policy",
                "source_path": "data/raw/guide.md",
                "title": "Guide",
                "metadata": {"kb_id": "kb-a"},
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
async def test_hybrid_retriever_respects_kb_scope(
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
            metadata={"kb_id": "kb-a"},
        ),
        [
            Chunk(
                chunk_id="chunk-a",
                doc_id="doc-a",
                chunk_index=0,
                text="PTO carryover rule",
                source_path="data/raw/a.md",
                title="A",
                metadata={"kb_id": "kb-a"},
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
            metadata={"kb_id": "kb-b"},
        ),
        [
            Chunk(
                chunk_id="chunk-b",
                doc_id="doc-b",
                chunk_index=0,
                text="PTO carryover rule",
                source_path="data/raw/b.md",
                title="B",
                metadata={"kb_id": "kb-b"},
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
                metadata={"kb_id": "kb-a"},
            ),
            Chunk(
                chunk_id="chunk-b",
                doc_id="doc-b",
                chunk_index=0,
                text="PTO carryover rule",
                source_path="data/raw/b.md",
                title="B",
                metadata={"kb_id": "kb-b"},
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
    )

    assert [context["chunk_id"] for context in contexts] == ["chunk-a"]


@pytest.mark.asyncio
async def test_ingestion_pipeline_updates_hybrid_index(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RAG_HYBRID_SEARCH_ENABLED", "true")
    monkeypatch.setenv("PARSED_OUTPUT_DIR", str(tmp_path / "parsed"))
    (tmp_path / "doc.txt").write_text("PTO carryover details", encoding="utf-8")
    monkeypatch.setattr(
        "app.ingestion.pipeline.discover_files", lambda path: [tmp_path / "doc.txt"]
    )

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
        generation_client=FakeGenerationClient(),
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
    assert artifact["chunks"][0]["metadata"]["node_id"].startswith("doc-")
    assert artifact["structured_document"]["root"]["children"]


@pytest.mark.asyncio
async def test_ingestion_pipeline_routes_image_audio_video_modalities(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("PARSED_OUTPUT_DIR", str(tmp_path / "parsed"))
    image_path = tmp_path / "logo.png"
    audio_path = tmp_path / "clip.mp3"
    video_path = tmp_path / "demo.mp4"
    image_path.write_bytes(b"\x89PNGfake")
    audio_path.write_bytes(b"ID3audio")
    video_path.write_bytes(b"\x00\x00mp4")
    monkeypatch.setattr(
        "app.ingestion.pipeline.discover_files",
        lambda path: [image_path, audio_path, video_path],
    )

    repository = InMemoryVectorRepository()
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
        generation_client=FakeGenerationClient(),
        tracing_manager=TracingManager("test-service", ""),
    )

    response = await pipeline.run(str(tmp_path), kb_id="default")

    assert response.documents == 3
    assert response.chunks == 3
    assert len(repository.entries) == 3
    modalities = sorted(chunk.modality for chunk, _ in repository.entries)
    assert modalities == ["audio", "image", "video"]
    for chunk, _ in repository.entries:
        assert chunk.media_uri is not None
        assert chunk.mime_type is not None
        assert chunk.text == ""


@pytest.mark.asyncio
async def test_uploaded_media_chunks_point_to_persistent_upload_uri(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("PARSED_OUTPUT_DIR", str(tmp_path / "parsed"))
    upload_dir = tmp_path / "uploads"
    image_path = tmp_path / "batch" / "logo.png"
    image_path.parent.mkdir()
    image_path.write_bytes(b"\x89PNGfake")
    monkeypatch.setattr("app.ingestion.pipeline.discover_files", lambda path: [image_path])

    repository = InMemoryVectorRepository()
    config = AppConfig(
        config_dir=tmp_path,
        settings={"chunk": {"size": 200, "overlap": 20}},
        models={"model_gateway": {"base_url": "", "api_key": ""}},
        prompts={},
    )
    monkeypatch.setenv("UPLOAD_OUTPUT_DIR", str(upload_dir))
    pipeline = IngestionPipeline(
        config=config,
        repository=repository,
        embedding_client=FakeEmbeddingClient(),
        generation_client=FakeGenerationClient(),
        tracing_manager=TracingManager("test-service", ""),
    )
    stable_source = "uploads/default/__shared__/logo.png"

    await pipeline.run(
        str(image_path.parent),
        kb_id="default",
        source_path_overrides={str(image_path.resolve()): stable_source},
    )

    chunk = repository.entries[0][0]
    assert chunk.source_path == stable_source
    assert chunk.media_uri == str(upload_dir / "default" / "__shared__" / "logo.png")


@pytest.mark.asyncio
async def test_image_ingest_creates_visual_and_text_chunks_when_parser_available(
    monkeypatch, tmp_path
) -> None:
    image_path = tmp_path / "receipt.png"
    image_path.write_bytes(b"\x89PNGfake")
    monkeypatch.setattr("app.ingestion.pipeline.discover_files", lambda path: [image_path])
    repository = InMemoryVectorRepository()
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
        generation_client=FakeGenerationClient(),
        tracing_manager=TracingManager("test-service", ""),
        document_parser=FakeImageTextParser(),
    )

    response = await pipeline.run(str(tmp_path), kb_id="default")

    assert response.documents == 1
    assert response.chunks == 2
    chunks = [chunk for chunk, _ in repository.entries]
    assert [chunk.modality for chunk in chunks] == ["image", "text"]
    assert chunks[0].metadata["chunk_kind"] == "media_object"
    assert chunks[0].metadata["chunk_strategy"] == "media_object"
    assert chunks[0].metadata["media_text_extraction"] == "ok"
    assert chunks[1].text == "Invoice total: 99 USD"
    assert chunks[1].metadata["chunk_strategy"] == "definition_leaf"
    assert chunks[1].metadata["source_modality"] == "image"


def test_media_reembed_uses_media_uri(tmp_path) -> None:
    durable_path = tmp_path / "uploads" / "default" / "logo.png"
    durable_path.parent.mkdir(parents=True)
    durable_path.write_bytes(b"image")
    config = AppConfig(
        config_dir=tmp_path,
        settings={"chunk": {"size": 200, "overlap": 20}},
        models={"model_gateway": {"base_url": "", "api_key": ""}},
        prompts={},
    )
    pipeline = IngestionPipeline(
        config=config,
        repository=InMemoryVectorRepository(),
        embedding_client=FakeEmbeddingClient(),
        generation_client=FakeGenerationClient(),
        tracing_manager=TracingManager("test-service", ""),
    )
    chunk = Chunk(
        chunk_id="c-image",
        doc_id="doc-image",
        chunk_index=0,
        text="",
        source_path="uploads/default/logo.png",
        title="logo",
        metadata={"kb_id": "default"},
        modality="image",
        media_uri=str(durable_path),
        mime_type="image/png",
    )

    inputs = pipeline._chunks_to_embed_inputs([chunk])  # noqa: SLF001

    assert inputs[0][0].data == b"image"


@pytest.mark.asyncio
async def test_ingestion_pipeline_does_not_write_parsed_artifact_when_embeddings_fail(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("RAG_HYBRID_SEARCH_ENABLED", "true")
    monkeypatch.setenv("PARSED_OUTPUT_DIR", str(tmp_path / "parsed"))
    (tmp_path / "doc.txt").write_text("PTO carryover details", encoding="utf-8")
    monkeypatch.setattr(
        "app.ingestion.pipeline.discover_files", lambda path: [tmp_path / "doc.txt"]
    )

    repository = InMemoryVectorRepository()
    hybrid = HybridRetriever(
        repository=repository,
        embedding_client=FailingEmbeddingClient(),
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
        embedding_client=FailingEmbeddingClient(),
        generation_client=FakeGenerationClient(),
        tracing_manager=TracingManager("test-service", ""),
        hybrid_retriever=hybrid,
        wiki_compiler=WikiCompiler(tmp_path / "wiki"),
    )

    with pytest.raises(ModelGatewayError):
        await pipeline.run(str(tmp_path), kb_id="default")

    parsed_dir = tmp_path / "parsed"
    assert not parsed_dir.exists() or not any(parsed_dir.glob("*.json"))
    assert not hybrid.bm25_index.search("pto", top_k=5)


@pytest.mark.asyncio
async def test_ingestion_pipeline_uses_stable_source_path_overrides(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("PARSED_OUTPUT_DIR", str(tmp_path / "parsed"))
    source_file = tmp_path / "upload-a.txt"
    source_file.write_text("first upload", encoding="utf-8")
    monkeypatch.setattr("app.ingestion.pipeline.discover_files", lambda path: [source_file])

    repository = InMemoryVectorRepository()
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
        generation_client=FakeGenerationClient(),
        tracing_manager=TracingManager("test-service", ""),
    )

    stable_source = "uploads/default/__shared__/policy.txt"
    await pipeline.run(
        str(tmp_path),
        kb_id="default",
        source_path_overrides={str(source_file.resolve()): stable_source},
    )

    first_doc = next(iter(repository.documents.values()))
    assert first_doc.source_path == stable_source

    replacement_file = tmp_path / "upload-b.txt"
    replacement_file.write_text("second upload replacement", encoding="utf-8")
    monkeypatch.setattr(
        "app.ingestion.pipeline.discover_files", lambda path: [replacement_file]
    )
    await pipeline.run(
        str(tmp_path),
        kb_id="default",
        source_path_overrides={str(replacement_file.resolve()): stable_source},
    )

    assert len(repository.documents) == 1
    replacement_doc = next(iter(repository.documents.values()))
    assert replacement_doc.source_path == stable_source
    assert replacement_doc.content == "second upload replacement"
    assert len(repository.entries) == 1


@pytest.mark.asyncio
async def test_ingestion_pipeline_preserves_existing_document_when_upsert_fails(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("PARSED_OUTPUT_DIR", str(tmp_path / "parsed"))
    source_file = tmp_path / "upload.txt"
    source_file.write_text("first committed content", encoding="utf-8")
    monkeypatch.setattr("app.ingestion.pipeline.discover_files", lambda path: [source_file])

    repository = FailingUpsertRepository(fail_on_call=2)
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
        generation_client=FakeGenerationClient(),
        tracing_manager=TracingManager("test-service", ""),
    )
    stable_source = "uploads/default/policy.txt"

    await pipeline.run(
        str(tmp_path),
        kb_id="default",
        source_path_overrides={str(source_file.resolve()): stable_source},
    )
    original_multivectors = {
        path.name for path in (tmp_path / "parsed" / "multivectors").glob("*.json")
    }
    source_file.write_text("second replacement content", encoding="utf-8")

    with pytest.raises(RuntimeError, match="repository upsert failed"):
        await pipeline.run(
            str(tmp_path),
            kb_id="default",
            source_path_overrides={str(source_file.resolve()): stable_source},
        )

    committed_doc = next(iter(repository.documents.values()))
    assert committed_doc.content == "first committed content"
    assert len(repository.entries) == 1
    parsed_artifact = json.loads(
        next((tmp_path / "parsed").glob("*.json")).read_text(encoding="utf-8")
    )
    assert parsed_artifact["document"]["content"] == "first committed content"
    assert not list((tmp_path / "parsed").glob("*.tmp"))
    assert {
        path.name for path in (tmp_path / "parsed" / "multivectors").glob("*.json")
    } == original_multivectors


@pytest.mark.asyncio
async def test_ingestion_pipeline_is_atomic_before_apply(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("RAG_HYBRID_SEARCH_ENABLED", "true")
    monkeypatch.setenv("PARSED_OUTPUT_DIR", str(tmp_path / "parsed"))
    files = [tmp_path / "ok.txt", tmp_path / "bad.txt"]
    for file_path in files:
        file_path.write_text(
            "FAIL_EMBED" if file_path.name == "bad.txt" else "safe content",
            encoding="utf-8",
        )
    monkeypatch.setattr("app.ingestion.pipeline.discover_files", lambda path: files)

    repository = InMemoryVectorRepository()
    hybrid = HybridRetriever(
        repository=repository,
        embedding_client=SelectiveFailingEmbeddingClient(),
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
        embedding_client=SelectiveFailingEmbeddingClient(),
        generation_client=FakeGenerationClient(),
        tracing_manager=TracingManager("test-service", ""),
        hybrid_retriever=hybrid,
        wiki_compiler=WikiCompiler(tmp_path / "wiki"),
    )

    with pytest.raises(ModelGatewayError):
        await pipeline.run(str(tmp_path), kb_id="default")

    parsed_dir = tmp_path / "parsed"
    assert not parsed_dir.exists() or not any(parsed_dir.glob("*.json"))
    assert not list((parsed_dir / "multivectors").glob("*.json"))
    assert not repository.documents
    assert not repository.entries
    assert not hybrid.bm25_index.search("safe", top_k=5)


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
