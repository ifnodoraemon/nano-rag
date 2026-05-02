from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from app.api.auth import RequestContext
from app.api.routes_business import (
    BenchmarkRunRequest,
    BusinessChatRequest,
    BusinessIngestRequest,
    BusinessRetrieveRequest,
    FeedbackRequest,
    KnowledgeBaseCreateRequest,
    _build_upload_source_path,
    rag_chat,
    rag_create_knowledge_base,
    rag_documents,
    rag_feedback,
    rag_ingest,
    rag_ingest_upload,
    rag_benchmark,
    rag_knowledge_bases,
    rag_retrieve,
    rag_trace,
)
from app.diagnostics.service import DiagnosisService
from app.core.tracing import FeedbackStore, TraceStore
from app.knowledge_bases.catalog import KnowledgeBaseRecord
from app.schemas.chat import Citation, ChatResponse
from app.schemas.trace import TraceRecord


CONTEXT = RequestContext(auth_mode="api_key")


class FakeKnowledgeBaseCatalog:
    def __init__(self) -> None:
        self.records = {
            "default": KnowledgeBaseRecord(
                kb_id="default",
                name="Default Knowledge Base",
                created_at=1.0,
                updated_at=1.0,
            )
        }

    def exists(self, kb_id: str) -> bool:
        return kb_id in self.records

    def list(self, allowed_kb_ids=None):  # noqa: ANN001
        records = list(self.records.values())
        if allowed_kb_ids is not None:
            records = [record for record in records if record.kb_id in allowed_kb_ids]
        return records

    def create(self, **kwargs):  # noqa: ANN003
        if kwargs["kb_id"] in self.records:
            raise ValueError(f"knowledge base already exists: {kwargs['kb_id']}")
        record = KnowledgeBaseRecord(
            created_at=2.0,
            updated_at=2.0,
            **kwargs,
        )
        self.records[record.kb_id] = record
        return record


def _request_with_container(container) -> SimpleNamespace:
    if not hasattr(container, "knowledge_base_catalog"):
        container.knowledge_base_catalog = FakeKnowledgeBaseCatalog()
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(container=container)))


@pytest.mark.asyncio
async def test_business_chat_uses_kb_scope_and_metadata_filters() -> None:
    async def fake_chat_run(payload):  # noqa: ANN001
        assert payload.kb_id == "default"
        assert payload.metadata_filters == {"doc_types": ["policy"]}
        return ChatResponse(
            answer="a1",
            citations=[Citation(chunk_id="c1", source="doc.md", score=1.0)],
            contexts=[{"chunk_id": "c1"}],
            trace_id="trace-1",
        )

    container = SimpleNamespace(chat_pipeline=SimpleNamespace(run=fake_chat_run))

    response = await rag_chat(
        BusinessChatRequest(
            query="policy",
            kb_id="default",
            session_id="session-a",
            metadata_filters={"doc_types": ["policy"]},
        ),
        _request_with_container(container),
        CONTEXT,
    )

    assert response.kb_id == "default"
    assert response.session_id == "session-a"
    assert response.trace_id == "trace-1"


@pytest.mark.asyncio
async def test_business_retrieve_uses_kb_scope_and_metadata_filters() -> None:
    async def fake_retrieve_debug(  # noqa: ANN001
        query,
        top_k=None,
        kb_id="default",
        session_id=None,
        metadata_filters=None,
    ):
        assert query == "policy"
        assert top_k == 5
        assert kb_id == "default"
        assert session_id == "session-a"
        assert metadata_filters == {"doc_types": ["policy"]}
        return SimpleNamespace(
            query=query,
            contexts=[{"chunk_id": "c1", "text": "answer context"}],
            retrieved=[{"chunk_id": "c1", "score": 0.9}],
            reranked=[{"chunk_id": "c1", "score": 0.95}],
            trace_id="trace-retrieve-1",
        )

    container = SimpleNamespace(
        retrieval_pipeline=SimpleNamespace(debug=fake_retrieve_debug)
    )

    response = await rag_retrieve(
        BusinessRetrieveRequest(
            query="policy",
            kb_id="default",
            session_id="session-a",
            top_k=5,
            metadata_filters={"doc_types": ["policy"]},
        ),
        _request_with_container(container),
        CONTEXT,
    )

    assert response.kb_id == "default"
    assert response.session_id == "session-a"
    assert response.trace_id == "trace-retrieve-1"
    assert response.contexts == [{"chunk_id": "c1", "text": "answer context"}]
    assert response.retrieved == [{"chunk_id": "c1", "score": 0.9}]
    assert response.reranked == [{"chunk_id": "c1", "score": 0.95}]


@pytest.mark.asyncio
async def test_business_retrieve_rejects_inaccessible_kb() -> None:
    container = SimpleNamespace(
        retrieval_pipeline=SimpleNamespace(debug=None),
    )

    with pytest.raises(HTTPException) as exc_info:
        await rag_retrieve(
            BusinessRetrieveRequest(query="policy", kb_id="default"),
            _request_with_container(container),
            RequestContext(auth_mode="api_key", allowed_kb_ids={"other"}),
        )

    assert exc_info.value.status_code == 403


def test_business_requests_normalize_blank_session() -> None:
    chat = BusinessChatRequest(query="q1", session_id="null")
    feedback = FeedbackRequest(trace_id="t1", rating="up", session_id="")

    assert chat.session_id is None
    assert feedback.session_id is None


@pytest.mark.asyncio
async def test_knowledge_base_catalog_routes_list_and_create() -> None:
    catalog = FakeKnowledgeBaseCatalog()
    container = SimpleNamespace(
        knowledge_base_catalog=catalog,
        config=SimpleNamespace(parsed_dir=SimpleNamespace(exists=lambda: False)),
        trace_store=TraceStore(),
    )

    created = await rag_create_knowledge_base(
        KnowledgeBaseCreateRequest(kb_id="policies", name="Policies"),
        _request_with_container(container),
        CONTEXT,
    )
    listed = await rag_knowledge_bases(_request_with_container(container), CONTEXT)

    assert created.kb_id == "policies"
    assert {item.kb_id for item in listed} == {"default", "policies"}


@pytest.mark.asyncio
async def test_business_ingest_wraps_ingest_response() -> None:
    async def fake_ingest_run(path, kb_id="default", source_path_overrides=None):  # noqa: ANN001
        assert path == "./data/raw"
        assert kb_id == "default"
        assert source_path_overrides is None
        return SimpleNamespace(documents=2, chunks=4)

    container = SimpleNamespace(ingestion_pipeline=SimpleNamespace(run=fake_ingest_run))

    response = await rag_ingest(
        BusinessIngestRequest(path="./data/raw", kb_id="default"),
        _request_with_container(container),
        CONTEXT,
    )

    assert response.status == "ok"
    assert response.documents == 2
    assert response.chunks == 4


@pytest.mark.asyncio
async def test_business_ingest_upload_wraps_ingest_response(tmp_path) -> None:
    captured: dict[str, object] = {}

    async def fake_ingest_run(path, kb_id="default", source_path_overrides=None):  # noqa: ANN001
        captured["path"] = path
        captured["kb_id"] = kb_id
        captured["source_path_overrides"] = source_path_overrides
        assert tmp_path.as_posix() in path
        return SimpleNamespace(documents=1, chunks=2)

    container = SimpleNamespace(
        ingestion_pipeline=SimpleNamespace(run=fake_ingest_run),
        config=SimpleNamespace(upload_dir=tmp_path),
    )

    upload = UploadFile(filename="policy.md", file=BytesIO(b"# Policy\nBody"))
    response = await rag_ingest_upload(
        _request_with_container(container),
        files=[upload],
        kb_id="default",
        context=CONTEXT,
    )

    assert response.status == "ok"
    assert response.uploaded_files == ["policy.md"]
    assert response.documents == 1
    assert response.chunks == 2
    assert captured["kb_id"] == "default"
    source_path_overrides = captured["source_path_overrides"]
    assert isinstance(source_path_overrides, dict)
    assert next(iter(source_path_overrides.values())) == "uploads/default/policy.md"
    assert (tmp_path / "default" / "policy.md").read_bytes() == b"# Policy\nBody"


@pytest.mark.asyncio
async def test_business_ingest_upload_stages_durable_copy_before_ingest(
    monkeypatch, tmp_path
) -> None:
    async def fake_ingest_run(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("ingestion should not run when durable staging fails")

    def fail_copy(*args, **kwargs):  # noqa: ANN002, ANN003
        raise OSError("disk full")

    monkeypatch.setattr("app.api.routes_business.shutil.copy2", fail_copy)
    container = SimpleNamespace(
        ingestion_pipeline=SimpleNamespace(run=fake_ingest_run),
        config=SimpleNamespace(upload_dir=tmp_path),
    )

    upload = UploadFile(filename="policy.md", file=BytesIO(b"# Policy\nBody"))
    with pytest.raises(OSError):
        await rag_ingest_upload(
            _request_with_container(container),
            files=[upload],
            kb_id="default",
            context=CONTEXT,
        )

    assert not list((tmp_path / "default").glob("*.tmp"))


@pytest.mark.asyncio
async def test_business_ingest_upload_restores_previous_file_when_ingest_fails(
    tmp_path,
) -> None:
    durable_path = tmp_path / "default" / "policy.md"
    durable_path.parent.mkdir(parents=True)
    durable_path.write_bytes(b"old policy")

    async def fake_ingest_run(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("indexing failed")

    container = SimpleNamespace(
        ingestion_pipeline=SimpleNamespace(run=fake_ingest_run),
        config=SimpleNamespace(upload_dir=tmp_path),
    )

    upload = UploadFile(filename="policy.md", file=BytesIO(b"new policy"))
    with pytest.raises(RuntimeError):
        await rag_ingest_upload(
            _request_with_container(container),
            files=[upload],
            kb_id="default",
            context=CONTEXT,
        )

    assert durable_path.read_bytes() == b"old policy"


@pytest.mark.asyncio
async def test_business_ingest_upload_ignores_backup_cleanup_failure(
    monkeypatch, tmp_path
) -> None:
    durable_path = tmp_path / "default" / "policy.md"
    durable_path.parent.mkdir(parents=True)
    durable_path.write_bytes(b"old policy")
    original_unlink = Path.unlink

    def flaky_unlink(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        if self.name.endswith(".bak"):
            raise OSError("cleanup failed")
        return original_unlink(self, *args, **kwargs)

    async def fake_ingest_run(*args, **kwargs):  # noqa: ANN002, ANN003
        return SimpleNamespace(documents=1, chunks=1)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)
    container = SimpleNamespace(
        ingestion_pipeline=SimpleNamespace(run=fake_ingest_run),
        config=SimpleNamespace(upload_dir=tmp_path),
    )

    response = await rag_ingest_upload(
        _request_with_container(container),
        files=[UploadFile(filename="policy.md", file=BytesIO(b"new policy"))],
        kb_id="default",
        context=CONTEXT,
    )

    assert response.status == "ok"
    assert durable_path.read_bytes() == b"new policy"


def test_upload_source_path_sanitizes_kb_component() -> None:
    assert _build_upload_source_path("政策 文档.pdf", "../kb/a") == (
        "uploads/kb_a/政策_文档.pdf"
    )


@pytest.mark.asyncio
async def test_business_ingest_upload_rejects_duplicate_filenames(tmp_path) -> None:
    container = SimpleNamespace(
        ingestion_pipeline=SimpleNamespace(run=None),
        config=SimpleNamespace(upload_dir=tmp_path),
    )

    with pytest.raises(HTTPException) as exc_info:
        await rag_ingest_upload(
            _request_with_container(container),
            files=[
                UploadFile(filename="policy.md", file=BytesIO(b"first")),
                UploadFile(filename="policy.md", file=BytesIO(b"second")),
            ],
            kb_id="default",
            context=CONTEXT,
        )

    assert exc_info.value.status_code == 400
    assert "duplicate upload filename" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_business_documents_lists_current_kb_only(tmp_path) -> None:
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    (parsed_dir / "doc-a.json").write_text(
        """
{
  "document": {
    "doc_id": "doc-a",
    "source_path": "uploads/default/policy.md",
    "title": "Policy",
    "content": "body",
    "metadata": {"kb_id": "default", "doc_type": "policy", "source_key": "policy"}
  },
  "chunks": [{}, {}]
}
        """.strip(),
        encoding="utf-8",
    )
    (parsed_dir / "doc-b.json").write_text(
        """
{
  "document": {
    "doc_id": "doc-b",
    "source_path": "uploads/other/guide.md",
    "title": "Guide",
    "content": "body",
    "metadata": {"kb_id": "other"}
  },
  "chunks": [{}]
}
        """.strip(),
        encoding="utf-8",
    )

    container = SimpleNamespace(config=SimpleNamespace(parsed_dir=parsed_dir))

    response = await rag_documents(
        _request_with_container(container),
        kb_id="default",
        context=CONTEXT,
    )

    assert [item.doc_id for item in response] == ["doc-a"]
    assert response[0].chunk_count == 2
    assert response[0].doc_type == "policy"


@pytest.mark.asyncio
async def test_feedback_is_saved() -> None:
    trace_store = TraceStore()
    trace_store.save_raw(
        TraceRecord(trace_id="trace-1", kb_id="default", session_id="session-a").model_dump()
    )
    feedback_store = FeedbackStore()
    container = SimpleNamespace(trace_store=trace_store, feedback_store=feedback_store)

    response = await rag_feedback(
        FeedbackRequest(
            trace_id="trace-1",
            rating="up",
            kb_id="default",
            session_id="session-a",
            comment="good answer",
        ),
        _request_with_container(container),
        CONTEXT,
    )

    assert response.status == "ok"
    assert feedback_store.list()[0].trace_id == "trace-1"


@pytest.mark.asyncio
async def test_business_trace_requires_matching_kb() -> None:
    trace_store = TraceStore()
    trace_store.save_raw(
        TraceRecord(trace_id="trace-1", kb_id="default", session_id="session-a").model_dump()
    )
    container = SimpleNamespace(trace_store=trace_store)

    record = await rag_trace(
        "trace-1",
        _request_with_container(container),
        kb_id="default",
        session_id="session-a",
        context=CONTEXT,
    )

    assert record.trace_id == "trace-1"

    with pytest.raises(HTTPException) as exc_info:
        await rag_trace(
            "trace-1",
            _request_with_container(container),
            kb_id="other",
            session_id="session-a",
            context=CONTEXT,
        )

    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_business_benchmark_rejects_dataset_kb_outside_context(
    monkeypatch, tmp_path
) -> None:
    dataset_dir = tmp_path / "eval"
    dataset_dir.mkdir()
    (dataset_dir / "sample.jsonl").write_text(
        '{"query":"q","kb_id":"other"}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("EVAL_DATASET_DIR", str(dataset_dir))

    container = SimpleNamespace(
        ragas_runner=SimpleNamespace(run=lambda records: {"results": records}),
        diagnosis_service=DiagnosisService(),
    )

    with pytest.raises(HTTPException) as exc_info:
        await rag_benchmark(
            BenchmarkRunRequest(dataset_path="sample.jsonl"),
            _request_with_container(container),
            RequestContext(auth_mode="api_key", allowed_kb_ids={"default"}),
        )

    assert exc_info.value.status_code == 403
