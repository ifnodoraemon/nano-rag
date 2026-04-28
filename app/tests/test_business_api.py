from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile

from app.api import routes_business
from app.api.routes_business import (
    BenchmarkRunRequest,
    BusinessChatRequest,
    BusinessIngestRequest,
    rag_documents,
    FeedbackRequest,
    rag_benchmark,
    rag_chat,
    rag_feedback,
    rag_ingest,
    rag_ingest_upload,
    rag_trace,
)
from app.core.tracing import FeedbackStore, TraceStore
from app.diagnostics.service import DiagnosisService
from app.schemas.chat import Citation, ChatResponse
from app.schemas.trace import TraceRecord


def _request_with_container(container) -> SimpleNamespace:
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(container=container)))


@pytest.mark.asyncio
async def test_business_chat_preserves_business_metadata() -> None:
    trace_store = TraceStore()
    trace_store.save_raw({"trace_id": "trace-1", "query": "q1"})

    async def fake_chat_run(payload):  # noqa: ANN001, ARG001
        return ChatResponse(
            answer="a1",
            citations=[Citation(chunk_id="c1", source="doc.md", score=1.0)],
            contexts=[{"chunk_id": "c1"}],
            trace_id="trace-1",
        )

    container = SimpleNamespace(
        chat_pipeline=SimpleNamespace(run=fake_chat_run),
        trace_store=trace_store,
    )

    response = await rag_chat(
        BusinessChatRequest(query="q1", kb_id="default", tenant_id="tenant-a", session_id="session-a"),
        _request_with_container(container),
    )

    assert response.kb_id == "default"
    assert response.tenant_id == "tenant-a"
    assert response.session_id == "session-a"
    assert response.trace_id == "trace-1"


@pytest.mark.asyncio
async def test_business_chat_passes_metadata_filters() -> None:
    trace_store = TraceStore()
    trace_store.save_raw({"trace_id": "trace-1", "query": "policy"})

    async def fake_chat_run(payload):  # noqa: ANN001
        assert payload.metadata_filters == {"doc_types": ["policy"]}
        return ChatResponse(
            answer="a1",
            citations=[Citation(chunk_id="c1", source="doc.md", score=1.0)],
            contexts=[{"chunk_id": "c1"}],
            trace_id="trace-1",
        )

    container = SimpleNamespace(
        chat_pipeline=SimpleNamespace(run=fake_chat_run),
        trace_store=trace_store,
    )

    response = await rag_chat(
        BusinessChatRequest(
            query="policy",
            kb_id="default",
            metadata_filters={"doc_types": ["policy"]},
        ),
        _request_with_container(container),
    )

    assert response.trace_id == "trace-1"


@pytest.mark.asyncio
async def test_business_ingest_wraps_ingest_response() -> None:
    async def fake_ingest_run(  # noqa: ANN001
        path, kb_id="default", tenant_id=None, source_path_overrides=None
    ):
        assert path == "./data/raw"
        assert kb_id == "default"
        assert tenant_id == "tenant-a"
        assert source_path_overrides is None
        return SimpleNamespace(documents=2, chunks=4)

    container = SimpleNamespace(ingestion_pipeline=SimpleNamespace(run=fake_ingest_run))

    response = await rag_ingest(
        BusinessIngestRequest(path="./data/raw", kb_id="default", tenant_id="tenant-a"),
        _request_with_container(container),
    )

    assert response.status == "ok"
    assert response.documents == 2
    assert response.chunks == 4


@pytest.mark.asyncio
async def test_business_ingest_upload_wraps_ingest_response(tmp_path) -> None:
    captured: dict[str, object] = {}

    async def fake_ingest_run(  # noqa: ANN001
        path, kb_id="default", tenant_id=None, source_path_overrides=None
    ):
        captured["path"] = path
        captured["kb_id"] = kb_id
        captured["tenant_id"] = tenant_id
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
        tenant_id="tenant-a",
    )

    assert response.status == "ok"
    assert response.source == "upload"
    assert response.uploaded_files == ["policy.md"]
    assert response.documents == 1
    assert response.chunks == 2
    assert captured["kb_id"] == "default"
    assert captured["tenant_id"] == "tenant-a"
    source_path_overrides = captured["source_path_overrides"]
    assert isinstance(source_path_overrides, dict)
    assert len(source_path_overrides) == 1
    assert next(iter(source_path_overrides.values())) == (
        "uploads/default/tenant-a/policy.md"
    )
    assert not any(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_business_ingest_upload_rejects_unsupported_extension(tmp_path) -> None:
    container = SimpleNamespace(
        ingestion_pipeline=SimpleNamespace(run=None),
        config=SimpleNamespace(upload_dir=tmp_path),
    )

    upload = UploadFile(filename="policy.exe", file=BytesIO(b"bad"))
    with pytest.raises(HTTPException) as exc_info:
        await rag_ingest_upload(
            _request_with_container(container),
            files=[upload],
            kb_id="default",
            tenant_id=None,
        )

    assert exc_info.value.status_code == 400
    assert not any(tmp_path.iterdir())


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
            tenant_id=None,
        )

    assert exc_info.value.status_code == 400
    assert "duplicate upload filename" in str(exc_info.value.detail)
    assert not any(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_business_ingest_upload_rejects_oversized_file_while_streaming(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(routes_business, "MAX_UPLOAD_BYTES", 4)
    container = SimpleNamespace(
        ingestion_pipeline=SimpleNamespace(run=None),
        config=SimpleNamespace(upload_dir=tmp_path),
    )

    upload = UploadFile(filename="large.md", file=BytesIO(b"12345"))
    with pytest.raises(HTTPException) as exc_info:
        await rag_ingest_upload(
            _request_with_container(container),
            files=[upload],
            kb_id="default",
            tenant_id=None,
        )

    assert exc_info.value.status_code == 413
    assert not any(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_business_documents_lists_current_scope_only(tmp_path) -> None:
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    (parsed_dir / "doc-a.json").write_text(
        """
{
  "document": {
    "doc_id": "doc-a",
    "source_path": "uploads/default/tenant-a/policy.md",
    "title": "Policy",
    "content": "body",
    "metadata": {
      "kb_id": "default",
      "tenant_id": "tenant-a",
      "doc_type": "policy",
      "source_key": "policy"
    }
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
    "source_path": "uploads/default/tenant-b/guide.md",
    "title": "Guide",
    "content": "body",
    "metadata": {
      "kb_id": "default",
      "tenant_id": "tenant-b"
    }
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
        tenant_id="tenant-a",
    )

    assert len(response) == 1
    assert response[0].doc_id == "doc-a"
    assert response[0].chunk_count == 2
    assert response[0].doc_type == "policy"
    assert response[0].source_path == "uploads/default/tenant-a/policy.md"


@pytest.mark.asyncio
async def test_feedback_is_saved() -> None:
    trace_store = TraceStore()
    trace_store.save_raw(
        TraceRecord(trace_id="trace-1", kb_id="default", tenant_id="tenant-a", session_id="session-a").model_dump()
    )
    feedback_store = FeedbackStore()
    container = SimpleNamespace(trace_store=trace_store, feedback_store=feedback_store)

    response = await rag_feedback(
        FeedbackRequest(
            trace_id="trace-1",
            rating="up",
            kb_id="default",
            tenant_id="tenant-a",
            session_id="session-a",
            comment="good answer",
        ),
        _request_with_container(container),
    )

    assert response.status == "ok"
    assert feedback_store.list()[0].trace_id == "trace-1"


@pytest.mark.asyncio
async def test_feedback_rejects_scope_mismatch() -> None:
    trace_store = TraceStore()
    trace_store.save_raw(
        TraceRecord(trace_id="trace-1", kb_id="default", tenant_id="tenant-a", session_id="session-a").model_dump()
    )
    container = SimpleNamespace(trace_store=trace_store, feedback_store=FeedbackStore())

    with pytest.raises(HTTPException) as exc_info:
        await rag_feedback(
            FeedbackRequest(trace_id="trace-1", rating="up", kb_id="default", tenant_id="tenant-b"),
            _request_with_container(container),
        )

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_business_trace_requires_matching_scope() -> None:
    trace_store = TraceStore()
    trace_store.save_raw(
        TraceRecord(trace_id="trace-1", kb_id="default", tenant_id="tenant-a", session_id="session-a").model_dump()
    )
    container = SimpleNamespace(trace_store=trace_store)

    record = await rag_trace(
        "trace-1",
        _request_with_container(container),
        kb_id="default",
        tenant_id="tenant-a",
        session_id="session-a",
    )

    assert record.trace_id == "trace-1"

    with pytest.raises(HTTPException) as exc_info:
        await rag_trace(
            "trace-1",
            _request_with_container(container),
            kb_id="default",
            tenant_id="tenant-b",
            session_id="session-a",
        )

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_benchmark_route_returns_report() -> None:
    trace_store = TraceStore()
    trace_store.save_raw(
        {
            "trace_id": "trace-1",
            "query": "q1",
            "latency_seconds": 0.7,
            "step_latencies": {"retrieval_seconds": 0.1, "generation_seconds": 0.6},
        }
    )
    dataset = [
        {
            "sample_id": "sample-1",
            "query": "q1",
            "reference_answer": "a1",
            "reference_contexts": ["ctx"],
        }
    ]
    container = SimpleNamespace(
        trace_store=trace_store,
        diagnosis_service=DiagnosisService(),
        ragas_runner=SimpleNamespace(
            run=lambda records: {
                "records": 1,
                "aggregate": {
                    "answer_exact_match": 1.0,
                    "reference_context_recall": 1.0,
                    "retrieved_context_count_avg": 1.0,
                },
                "results": [
                    {
                        "sample_id": records[0]["sample_id"],
                        "trace_id": records[0]["trace_id"],
                        "query": "q1",
                        "answer_exact_match": 1.0,
                        "reference_context_recall": 1.0,
                        "answer": "a1",
                        "reference_answer": "a1",
                    }
                ],
            }
        ),
    )

    async def fake_materialize_eval_records(container_arg, dataset_arg):  # noqa: ANN001
        assert container_arg is container
        assert dataset_arg == dataset
        return [{"sample_id": "sample-1", "trace_id": "trace-1"}]

    request = _request_with_container(container)

    from unittest.mock import patch

    with patch("app.api.routes_business.load_jsonl_dataset", return_value=dataset), patch(
        "app.api.routes_business.materialize_eval_records",
        new=fake_materialize_eval_records,
    ), patch("app.api.routes_business.save_json") as save_json:
        response = await rag_benchmark(
            BenchmarkRunRequest(
                dataset_path="data/eval/sample.jsonl",
                output_path="data/reports/eval/benchmarks/test.json",
            ),
            request,
        )

    assert response.status == "ok"
    assert response.report["aggregate"]["latency_seconds_avg"] == 0.7
    save_json.assert_called_once()


@pytest.mark.asyncio
async def test_benchmark_route_rejects_dataset_outside_eval_dir() -> None:
    container = SimpleNamespace(
        trace_store=TraceStore(),
        diagnosis_service=DiagnosisService(),
        ragas_runner=SimpleNamespace(run=lambda records: {"records": len(records), "aggregate": {}, "results": []}),
    )

    with pytest.raises(HTTPException) as exc_info:
        await rag_benchmark(
            BenchmarkRunRequest(dataset_path="../outside.jsonl"),
            _request_with_container(container),
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_benchmark_route_returns_503_when_eval_disabled() -> None:
    container = SimpleNamespace(
        trace_store=TraceStore(),
        diagnosis_service=DiagnosisService(),
        ragas_runner=None,
    )

    with pytest.raises(HTTPException) as exc_info:
        await rag_benchmark(
            BenchmarkRunRequest(dataset_path="data/eval/sample.jsonl"),
            _request_with_container(container),
        )

    assert exc_info.value.status_code == 503
    assert "RAG_EVAL_ENABLED=true" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_benchmark_route_returns_503_when_diagnosis_disabled() -> None:
    container = SimpleNamespace(
        trace_store=TraceStore(),
        diagnosis_service=None,
        ragas_runner=SimpleNamespace(run=lambda records: {"records": len(records)}),
    )

    with pytest.raises(HTTPException) as exc_info:
        await rag_benchmark(
            BenchmarkRunRequest(dataset_path="data/eval/sample.jsonl"),
            _request_with_container(container),
        )

    assert exc_info.value.status_code == 503
    assert "RAG_DIAGNOSIS_ENABLED=true" in str(exc_info.value.detail)
