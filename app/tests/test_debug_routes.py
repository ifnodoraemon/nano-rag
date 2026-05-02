import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api.routes_debug import (
    diagnose_auto,
    diagnose_eval,
    diagnose_trace,
    get_benchmark_reports,
    get_trace,
    get_eval_reports,
    get_benchmark_report_detail,
    get_eval_report_detail,
    list_traces,
    parsed_document_debug,
    retrieve_debug,
    run_eval,
    storage_debug,
)
from app.api.auth import RequestContext
from app.core.tracing import TraceStore
from app.eval.dataset import (
    list_benchmark_reports,
    list_eval_datasets,
    list_eval_reports,
    resolve_eval_dataset_path,
)
from app.eval.ragas_runner import RagasRunner
from app.schemas.diagnosis import AutoDiagnosisRequest, EvalDiagnosisRequest, TraceDiagnosisRequest
from app.schemas.eval import EvalRunRequest


def _request_with_container(container) -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(container=container))
    )


class FakeCatalog:
    def __init__(self, kb_ids=("default", "other")) -> None:
        self.kb_ids = set(kb_ids)

    def exists(self, kb_id: str) -> bool:
        return kb_id in self.kb_ids


@pytest.mark.asyncio
async def test_eval_report_detail_rejects_path_outside_eval_dir() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await get_eval_report_detail(_request_with_container(SimpleNamespace()), "../secret.json")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_benchmark_report_detail_rejects_path_outside_benchmark_dir() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await get_benchmark_report_detail(
            _request_with_container(SimpleNamespace()), "../not-benchmark.json"
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_run_eval_rejects_dataset_outside_eval_dir() -> None:
    container = SimpleNamespace(
        ragas_runner=SimpleNamespace(
            run=lambda records: {
                "status": "ok",
                "records": len(records),
                "aggregate": {},
                "results": [],
            }
        ),
    )

    with pytest.raises(HTTPException) as exc_info:
        await run_eval(
            EvalRunRequest(dataset_path="../outside.jsonl"),
            _request_with_container(container),
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_run_eval_uses_ragas_lib_when_requested(monkeypatch, tmp_path) -> None:
    dataset_dir = tmp_path / "eval"
    dataset_dir.mkdir()
    dataset_path = dataset_dir / "sample.jsonl"
    dataset_path.write_text(
        '{"query":"q","answer":"a","retrieved_contexts":["ctx"]}\n',
        encoding="utf-8",
    )
    output_dir = tmp_path / "reports"
    monkeypatch.setenv("EVAL_DATASET_DIR", str(dataset_dir))
    monkeypatch.setenv("EVAL_REPORT_DIR", str(output_dir))

    class FakeRunner:
        def run(self, records):  # noqa: ANN001
            return {"mode": "sync", "records": len(records)}

        async def run_async(self, records):  # noqa: ANN001
            return {"mode": "ragas", "records": len(records)}

    container = SimpleNamespace(ragas_runner=FakeRunner())

    response = await run_eval(
        EvalRunRequest(dataset_path=str(dataset_path), use_ragas_lib=True),
        _request_with_container(container),
    )

    assert response.report["mode"] == "ragas"


def test_eval_dataset_listing_supports_external_dir(monkeypatch, tmp_path) -> None:
    dataset_dir = tmp_path / "eval"
    dataset_dir.mkdir()
    (dataset_dir / "sample.jsonl").write_text('{"query":"q"}\n', encoding="utf-8")
    monkeypatch.setenv("EVAL_DATASET_DIR", str(dataset_dir))

    datasets = list_eval_datasets()

    assert datasets[0]["path"] == "sample.jsonl"
    assert resolve_eval_dataset_path("sample.jsonl") == dataset_dir / "sample.jsonl"


def test_eval_report_listing_supports_external_dir(monkeypatch, tmp_path) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / "report.json").write_text(
        '{"records":1,"status":"ok","aggregate":{}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("EVAL_REPORT_DIR", str(report_dir))

    reports = list_eval_reports()

    assert reports[0]["path"] == "report.json"


def test_benchmark_report_listing_supports_external_dir(monkeypatch, tmp_path) -> None:
    report_dir = tmp_path / "reports"
    benchmark_dir = report_dir / "benchmarks"
    benchmark_dir.mkdir(parents=True)
    (benchmark_dir / "benchmark.json").write_text(
        '{"records":1,"status":"ok","aggregate":{}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("EVAL_REPORT_DIR", str(report_dir))

    reports = list_benchmark_reports()

    assert reports[0]["path"] == "benchmark.json"


@pytest.mark.asyncio
async def test_report_detail_filters_results_by_allowed_kbs(monkeypatch, tmp_path) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / "report.json").write_text(
        """
{
  "records": 2,
  "aggregate": {"answer_exact_match": 0.5},
  "results": [
    {"sample_id": "default", "kb_id": "default", "answer_exact_match": 1.0},
    {"sample_id": "other", "kb_id": "other", "answer_exact_match": 0.0}
  ]
}
        """.strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("EVAL_REPORT_DIR", str(report_dir))
    container = SimpleNamespace(
        trace_store=TraceStore(),
        knowledge_base_catalog=FakeCatalog(),
    )
    context = RequestContext(auth_mode="api_key", allowed_kb_ids={"default"})

    detail = await get_eval_report_detail(
        _request_with_container(container),
        "report.json",
        context,
    )
    listed = await get_eval_reports(_request_with_container(container), context)

    assert detail["records"] == 1
    assert [result["sample_id"] for result in detail["results"]] == ["default"]
    assert detail["aggregate"]["answer_exact_match"] == 1.0
    assert listed[0]["records"] == 1


@pytest.mark.asyncio
async def test_generated_eval_report_preserves_kb_id_for_scoped_filtering(
    monkeypatch, tmp_path
) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    report = RagasRunner().run(
        [
            {
                "sample_id": "other",
                "kb_id": "other",
                "query": "q",
                "answer": "secret",
                "reference_answer": "secret",
            },
            {
                "sample_id": "default",
                "kb_id": "default",
                "query": "q",
                "answer": "visible",
                "reference_answer": "visible",
            },
        ]
    )
    (report_dir / "report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )
    monkeypatch.setenv("EVAL_REPORT_DIR", str(report_dir))
    container = SimpleNamespace(
        trace_store=TraceStore(),
        knowledge_base_catalog=FakeCatalog(),
    )

    detail = await get_eval_report_detail(
        _request_with_container(container),
        "report.json",
        RequestContext(auth_mode="api_key", allowed_kb_ids={"default"}),
    )

    assert [result["kb_id"] for result in report["results"]] == ["other", "default"]
    assert [result["sample_id"] for result in detail["results"]] == ["default"]


@pytest.mark.asyncio
async def test_scoped_report_filter_excludes_results_without_known_kb(
    monkeypatch, tmp_path
) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / "report.json").write_text(
        """
{
  "records": 2,
  "aggregate": {"answer_exact_match": 0.5},
  "results": [
    {"sample_id": "legacy", "answer_exact_match": 1.0},
    {"sample_id": "default", "kb_id": "default", "answer_exact_match": 0.0}
  ]
}
        """.strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("EVAL_REPORT_DIR", str(report_dir))
    container = SimpleNamespace(
        trace_store=TraceStore(),
        knowledge_base_catalog=FakeCatalog(),
    )

    detail = await get_eval_report_detail(
        _request_with_container(container),
        "report.json",
        RequestContext(auth_mode="api_key", allowed_kb_ids={"default"}),
    )

    assert [result["sample_id"] for result in detail["results"]] == ["default"]


@pytest.mark.asyncio
async def test_benchmark_report_detail_filters_results_by_allowed_kbs(
    monkeypatch, tmp_path
) -> None:
    report_dir = tmp_path / "reports"
    benchmark_dir = report_dir / "benchmarks"
    benchmark_dir.mkdir(parents=True)
    (benchmark_dir / "benchmark.json").write_text(
        """
{
  "records": 2,
  "aggregate": {"answer_exact_match": 0.5},
  "diagnosis_counts": {"generation_misalignment": 1},
  "results": [
    {
      "sample_id": "default",
      "kb_id": "default",
      "answer_exact_match": 1.0,
      "diagnosis": {"findings": [{"category": "ok"}]}
    },
    {
      "sample_id": "other",
      "kb_id": "other",
      "answer_exact_match": 0.0,
      "diagnosis": {"findings": [{"category": "secret"}]}
    }
  ]
}
        """.strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("EVAL_REPORT_DIR", str(report_dir))
    container = SimpleNamespace(
        trace_store=TraceStore(),
        knowledge_base_catalog=FakeCatalog(),
    )
    context = RequestContext(auth_mode="api_key", allowed_kb_ids={"default"})

    detail = await get_benchmark_report_detail(
        _request_with_container(container),
        "benchmark.json",
        context,
    )
    listed = await get_benchmark_reports(_request_with_container(container), context)

    assert detail["records"] == 1
    assert [result["sample_id"] for result in detail["results"]] == ["default"]
    assert detail["diagnosis_counts"] == {"ok": 1}
    assert listed[0]["records"] == 1


@pytest.mark.asyncio
async def test_diagnose_eval_uses_filtered_result_index(monkeypatch, tmp_path) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / "report.json").write_text(
        """
{
  "results": [
    {
      "sample_id": "other",
      "kb_id": "other",
      "answer_exact_match": 0.0,
      "reference_context_recall": 1.0,
      "answer": "secret",
      "reference_answer": "secret"
    },
    {
      "sample_id": "default",
      "kb_id": "default",
      "answer_exact_match": 0.0,
      "reference_context_recall": 1.0,
      "answer": "visible",
      "reference_answer": "expected"
    }
  ]
}
        """.strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("EVAL_REPORT_DIR", str(report_dir))
    container = SimpleNamespace(
        diagnosis_service=SimpleNamespace(
            diagnose_eval_result=lambda report, index: SimpleNamespace(
                target_type="eval_result",
                sample_id=report["results"][index]["sample_id"],
                findings=[],
                model_dump=lambda: {},
            )
        ),
        trace_store=TraceStore(),
        knowledge_base_catalog=FakeCatalog(),
    )

    diagnosis = await diagnose_eval(
        EvalDiagnosisRequest(report_path="report.json", result_index=0),
        _request_with_container(container),
        RequestContext(auth_mode="api_key", allowed_kb_ids={"default"}),
    )

    assert diagnosis.sample_id == "default"


@pytest.mark.asyncio
async def test_diagnose_eval_uses_result_session_for_trace_scope(
    monkeypatch, tmp_path
) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / "report.json").write_text(
        """
{
  "results": [
    {
      "sample_id": "default",
      "kb_id": "default",
      "session_id": "session-a",
      "trace_id": "trace-1",
      "answer_exact_match": 0.0,
      "reference_context_recall": 1.0,
      "answer": "visible",
      "reference_answer": "expected"
    }
  ]
}
        """.strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("EVAL_REPORT_DIR", str(report_dir))
    trace_store = TraceStore()
    trace_store.save_raw(
        {"trace_id": "trace-1", "query": "q", "kb_id": "default", "session_id": "session-a"}
    )
    container = SimpleNamespace(
        diagnosis_service=SimpleNamespace(
            diagnose_eval_result=lambda report, index: SimpleNamespace(
                target_type="eval_result",
                sample_id=report["results"][index]["sample_id"],
                findings=[],
                model_dump=lambda: {},
            )
        ),
        trace_store=trace_store,
        knowledge_base_catalog=FakeCatalog(),
    )

    diagnosis = await diagnose_eval(
        EvalDiagnosisRequest(report_path="report.json", result_index=0),
        _request_with_container(container),
        RequestContext(auth_mode="api_key", allowed_kb_ids={"default"}),
    )

    assert diagnosis.sample_id == "default"


@pytest.mark.asyncio
async def test_run_eval_rejects_dataset_kb_outside_context(monkeypatch, tmp_path) -> None:
    dataset_dir = tmp_path / "eval"
    dataset_dir.mkdir()
    dataset_path = dataset_dir / "sample.jsonl"
    dataset_path.write_text('{"query":"q","kb_id":"other"}\n', encoding="utf-8")
    monkeypatch.setenv("EVAL_DATASET_DIR", str(dataset_dir))

    container = SimpleNamespace(
        ragas_runner=SimpleNamespace(run=lambda records: {"results": records}),
        knowledge_base_catalog=FakeCatalog(),
    )

    with pytest.raises(HTTPException) as exc_info:
        await run_eval(
            EvalRunRequest(dataset_path="sample.jsonl"),
            _request_with_container(container),
            RequestContext(auth_mode="api_key", allowed_kb_ids={"default"}),
        )

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_run_eval_returns_503_when_eval_disabled() -> None:
    container = SimpleNamespace(ragas_runner=None)

    with pytest.raises(HTTPException) as exc_info:
        await run_eval(
            EvalRunRequest(dataset_path="data/eval/sample.jsonl"),
            _request_with_container(container),
        )

    assert exc_info.value.status_code == 503
    assert "RAG_EVAL_ENABLED=true" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_diagnose_trace_returns_503_when_diagnosis_disabled() -> None:
    container = SimpleNamespace(diagnosis_service=None, trace_store=SimpleNamespace(get=lambda trace_id: None))

    with pytest.raises(HTTPException) as exc_info:
        await diagnose_trace(
            TraceDiagnosisRequest(trace_id="trace-1"),
            _request_with_container(container),
        )

    assert exc_info.value.status_code == 503
    assert "RAG_DIAGNOSIS_ENABLED=true" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_diagnose_trace_accepts_matching_session_scope() -> None:
    trace_store = TraceStore()
    trace_store.save_raw(
        {"trace_id": "trace-1", "query": "q", "kb_id": "default", "session_id": "session-a"}
    )
    container = SimpleNamespace(
        diagnosis_service=SimpleNamespace(
            diagnose_trace=lambda trace: SimpleNamespace(
                target_type="trace",
                trace_id=trace.trace_id,
                findings=[],
                model_dump=lambda: {},
            )
        ),
        trace_store=trace_store,
        knowledge_base_catalog=FakeCatalog(),
    )

    diagnosis = await diagnose_trace(
        TraceDiagnosisRequest(trace_id="trace-1", session_id="session-a"),
        _request_with_container(container),
        RequestContext(auth_mode="api_key", allowed_kb_ids={"default"}),
    )

    assert diagnosis.trace_id == "trace-1"


@pytest.mark.asyncio
async def test_diagnose_auto_returns_503_when_diagnosis_disabled() -> None:
    container = SimpleNamespace(diagnosis_service=None, trace_store=SimpleNamespace(list=lambda: SimpleNamespace(items=[])))

    with pytest.raises(HTTPException) as exc_info:
        await diagnose_auto(
            AutoDiagnosisRequest(),
            _request_with_container(container),
        )

    assert exc_info.value.status_code == 503
    assert "RAG_DIAGNOSIS_ENABLED=true" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_retrieve_debug_rejects_inaccessible_kb() -> None:
    async def fake_debug(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("debug retrieval should not run")

    container = SimpleNamespace(
        retrieval_pipeline=SimpleNamespace(debug=fake_debug),
        knowledge_base_catalog=FakeCatalog(),
    )

    with pytest.raises(HTTPException) as exc_info:
        await retrieve_debug(
            SimpleNamespace(query="q", top_k=1, kb_id="other", session_id=None),
            _request_with_container(container),
            RequestContext(auth_mode="api_key", allowed_kb_ids={"default"}),
        )

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_trace_routes_filter_by_allowed_kbs() -> None:
    trace_store = TraceStore()
    trace_store.save_raw({"trace_id": "t-default", "query": "q", "kb_id": "default"})
    trace_store.save_raw({"trace_id": "t-other", "query": "q", "kb_id": "other"})
    container = SimpleNamespace(
        trace_store=trace_store,
        knowledge_base_catalog=FakeCatalog(),
    )
    context = RequestContext(auth_mode="api_key", allowed_kb_ids={"default"})

    listed = await list_traces(_request_with_container(container), context=context)
    assert [item.trace_id for item in listed.items] == ["t-default"]

    with pytest.raises(HTTPException) as exc_info:
        await get_trace("t-other", _request_with_container(container), context=context)

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_trace_list_scopes_before_paginating_past_large_unauthorized_slice() -> None:
    trace_store = TraceStore(max_records=10002)
    trace_store.save_raw({"trace_id": "t-default", "query": "q", "kb_id": "default"})
    for index in range(10001):
        trace_store.save_raw(
            {"trace_id": f"t-other-{index}", "query": "q", "kb_id": "other"}
        )
    container = SimpleNamespace(
        trace_store=trace_store,
        knowledge_base_catalog=FakeCatalog(),
    )

    listed = await list_traces(
        _request_with_container(container),
        context=RequestContext(auth_mode="api_key", allowed_kb_ids={"default"}),
    )

    assert listed.total == 1
    assert [item.trace_id for item in listed.items] == ["t-default"]


@pytest.mark.asyncio
async def test_storage_and_parsed_debug_filter_by_allowed_kbs(tmp_path) -> None:
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    (parsed_dir / "doc-default.json").write_text(
        '{"document":{"metadata":{"kb_id":"default"}},"chunks":[]}',
        encoding="utf-8",
    )
    (parsed_dir / "doc-other.json").write_text(
        '{"document":{"metadata":{"kb_id":"other"}},"chunks":[]}',
        encoding="utf-8",
    )
    container = SimpleNamespace(
        config=SimpleNamespace(parsed_dir=parsed_dir),
        repository=SimpleNamespace(stats=lambda: {"backend": "memory"}),
        knowledge_base_catalog=FakeCatalog(),
    )
    context = RequestContext(auth_mode="api_key", allowed_kb_ids={"default"})

    debug = await storage_debug(_request_with_container(container), context)
    assert debug["parsed_files"] == ["doc-default.json"]

    with pytest.raises(HTTPException) as exc_info:
        await parsed_document_debug(
            "doc-other",
            _request_with_container(container),
            context,
        )

    assert exc_info.value.status_code == 403
