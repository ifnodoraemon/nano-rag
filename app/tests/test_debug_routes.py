from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api.routes_debug import (
    diagnose_auto,
    diagnose_trace,
    get_benchmark_report_detail,
    get_eval_report_detail,
    run_eval,
)
from app.schemas.diagnosis import AutoDiagnosisRequest, TraceDiagnosisRequest
from app.schemas.eval import EvalRunRequest


def _request_with_container(container) -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(container=container))
    )


@pytest.mark.asyncio
async def test_eval_report_detail_rejects_path_outside_eval_dir() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await get_eval_report_detail("../secret.json")

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_benchmark_report_detail_rejects_path_outside_benchmark_dir() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await get_benchmark_report_detail("../not-benchmark.json")

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
async def test_diagnose_auto_returns_503_when_diagnosis_disabled() -> None:
    container = SimpleNamespace(diagnosis_service=None, trace_store=SimpleNamespace(list=lambda: SimpleNamespace(items=[])))

    with pytest.raises(HTTPException) as exc_info:
        await diagnose_auto(
            AutoDiagnosisRequest(),
            _request_with_container(container),
        )

    assert exc_info.value.status_code == 503
    assert "RAG_DIAGNOSIS_ENABLED=true" in str(exc_info.value.detail)
