import json
import re
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.api.auth import require_api_key
from app.eval.dataset import (
    ROOT,
    get_eval_report_dir,
    list_benchmark_reports,
    list_eval_datasets,
    list_eval_reports,
    load_json,
    load_jsonl_dataset,
    resolve_benchmark_report_path,
    resolve_eval_dataset_path,
    resolve_eval_report_path,
    save_json,
)
from app.eval.service import materialize_eval_records
from app.eval.replay import replay_trace
from app.schemas.chat import ChatRequest
from app.schemas.common import PaginatedResponse
from app.schemas.diagnosis import (
    AutoDiagnosisRequest,
    DiagnosisResponse,
    EvalDiagnosisRequest,
    TraceDiagnosisRequest,
)
from app.schemas.eval import EvalRunRequest, EvalRunResponse
from app.schemas.trace import RetrievalDebugResponse, TraceRecord, TraceSummary
from app.utils.text import safe_float

router = APIRouter()


def _require_eval_runner(container) -> object:
    runner = getattr(container, "ragas_runner", None)
    if runner is None:
        raise HTTPException(
            status_code=503,
            detail="evaluation is disabled. Set RAG_EVAL_ENABLED=true to enable eval and benchmark runs.",
        )
    return runner


def _require_diagnosis_service(container) -> object:
    service = getattr(container, "diagnosis_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="diagnosis is disabled. Set RAG_DIAGNOSIS_ENABLED=true to enable diagnosis endpoints.",
        )
    return service


@router.post(
    "/retrieve/debug",
    response_model=RetrievalDebugResponse,
    dependencies=[Depends(require_api_key)],
)
async def retrieve_debug(
    payload: ChatRequest, request: Request
) -> RetrievalDebugResponse:
    container = request.app.state.container
    return await container.retrieval_pipeline.debug(
        payload.query,
        payload.top_k,
        kb_id=payload.kb_id or "default",
        tenant_id=payload.tenant_id,
        session_id=payload.session_id,
    )


@router.get(
    "/traces",
    response_model=PaginatedResponse[TraceSummary],
    dependencies=[Depends(require_api_key)],
)
async def list_traces(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    kb_id: str | None = Query(default=None),
    tenant_id: str | None = Query(default=None),
) -> PaginatedResponse[TraceSummary]:
    container = request.app.state.container
    return container.trace_store.list(
        page=page, page_size=page_size, kb_id=kb_id, tenant_id=tenant_id
    )


@router.get(
    "/traces/{trace_id}",
    response_model=TraceRecord,
    dependencies=[Depends(require_api_key)],
)
async def get_trace(
    trace_id: str,
    request: Request,
    kb_id: str | None = Query(default=None),
    tenant_id: str | None = Query(default=None),
) -> TraceRecord:
    container = request.app.state.container
    record = container.trace_store.get(trace_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"trace not found: {trace_id}")
    if kb_id and record.kb_id and record.kb_id != kb_id:
        raise HTTPException(status_code=404, detail=f"trace not found: {trace_id}")
    if tenant_id and record.tenant_id and record.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail=f"trace not found: {trace_id}")
    return record


@router.get("/eval/datasets", dependencies=[Depends(require_api_key)])
async def get_eval_datasets() -> list[dict[str, object]]:
    return list_eval_datasets()


@router.get("/eval/reports", dependencies=[Depends(require_api_key)])
async def get_eval_reports() -> list[dict[str, object]]:
    return list_eval_reports()


@router.get("/benchmark/reports", dependencies=[Depends(require_api_key)])
async def get_benchmark_reports() -> list[dict[str, object]]:
    return list_benchmark_reports()


@router.get("/eval/reports/detail", dependencies=[Depends(require_api_key)])
async def get_eval_report_detail(path: str = Query(...)) -> dict:
    try:
        target = resolve_eval_report_path(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"eval report not found: {path}")
    return load_json(str(target))


@router.get("/benchmark/reports/detail", dependencies=[Depends(require_api_key)])
async def get_benchmark_report_detail(path: str = Query(...)) -> dict:
    try:
        target = resolve_benchmark_report_path(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not target.exists():
        raise HTTPException(
            status_code=404, detail=f"benchmark report not found: {path}"
        )
    return load_json(str(target))


@router.post(
    "/eval/run", response_model=EvalRunResponse, dependencies=[Depends(require_api_key)]
)
async def run_eval(payload: EvalRunRequest, request: Request) -> EvalRunResponse:
    container = request.app.state.container
    ragas_runner = _require_eval_runner(container)
    try:
        dataset_path = resolve_eval_dataset_path(payload.dataset_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    dataset = load_jsonl_dataset(str(dataset_path))
    evaluated_records = await materialize_eval_records(container, dataset)
    report = ragas_runner.run(evaluated_records)
    output_path = payload.output_path
    if output_path:
        try:
            target = resolve_eval_report_path(output_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        report_dir = get_eval_report_dir()
        report_dir.mkdir(parents=True, exist_ok=True)
        dataset_name = dataset_path.stem or "eval"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = report_dir / f"{dataset_name}_{timestamp}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    save_json(str(target), report)
    return EvalRunResponse(status="ok", output_path=str(target), report=report)


@router.post(
    "/diagnose/trace",
    response_model=DiagnosisResponse,
    dependencies=[Depends(require_api_key)],
)
async def diagnose_trace(
    payload: TraceDiagnosisRequest, request: Request
) -> DiagnosisResponse:
    container = request.app.state.container
    diagnosis_service = _require_diagnosis_service(container)
    trace = container.trace_store.get(payload.trace_id)
    if trace is None:
        raise HTTPException(
            status_code=404, detail=f"trace not found: {payload.trace_id}"
        )
    diagnosis = diagnosis_service.diagnose_trace(trace)
    if payload.include_ai:
        diagnosis = await diagnosis_service.add_ai_suggestion(
            diagnosis,
            {"trace": trace.model_dump()},
        )
    return diagnosis


@router.post(
    "/diagnose/eval",
    response_model=DiagnosisResponse,
    dependencies=[Depends(require_api_key)],
)
async def diagnose_eval(
    payload: EvalDiagnosisRequest, request: Request
) -> DiagnosisResponse:
    container = request.app.state.container
    diagnosis_service = _require_diagnosis_service(container)
    try:
        report_path = resolve_eval_report_path(payload.report_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not report_path.exists():
        raise HTTPException(
            status_code=404, detail=f"eval report not found: {payload.report_path}"
        )
    report = load_json(str(report_path))
    try:
        diagnosis = diagnosis_service.diagnose_eval_result(report, payload.result_index)
    except IndexError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if payload.include_ai:
        diagnosis = await diagnosis_service.add_ai_suggestion(
            diagnosis,
            {
                "report_path": payload.report_path,
                "result_index": payload.result_index,
                "result": report.get("results", [])[payload.result_index],
            },
        )
    return diagnosis


@router.post(
    "/diagnose/auto",
    response_model=DiagnosisResponse,
    dependencies=[Depends(require_api_key)],
)
async def diagnose_auto(
    payload: AutoDiagnosisRequest, request: Request
) -> DiagnosisResponse:
    container = request.app.state.container
    diagnosis_service = _require_diagnosis_service(container)

    reports = list_eval_reports()
    max_iterations = 10
    if reports:
        latest_report = reports[0]
        report = load_json(str(ROOT / latest_report["path"]))
        results = report.get("results", []) if isinstance(report, dict) else []
        for index, result in enumerate(results):
            if index >= max_iterations:
                break
            if (
                safe_float(result.get("answer_exact_match")) < 1.0
                or safe_float(result.get("reference_context_recall")) < 1.0
            ):
                diagnosis = diagnosis_service.diagnose_eval_result(report, index)
                if payload.include_ai:
                    diagnosis = await diagnosis_service.add_ai_suggestion(
                        diagnosis,
                        {
                            "report_path": latest_report["path"],
                            "result_index": index,
                            "result": result,
                        },
                    )
                return diagnosis

    traces = container.trace_store.list()
    if traces.items:
        latest_trace_id = traces.items[0].trace_id
        trace = container.trace_store.get(latest_trace_id)
        if trace is not None:
            diagnosis = diagnosis_service.diagnose_trace(trace)
            if payload.include_ai:
                diagnosis = await diagnosis_service.add_ai_suggestion(
                    diagnosis,
                    {"trace": trace.model_dump()},
                )
            return diagnosis

    raise HTTPException(
        status_code=404, detail="no eval bad cases or traces available for diagnosis"
    )


@router.post(
    "/replay/{trace_id}",
    dependencies=[Depends(require_api_key)],
)
async def replay_trace_endpoint(trace_id: str, request: Request) -> dict:
    container = request.app.state.container
    result = await replay_trace(container, trace_id)
    return {
        "trace_id": result.trace_id,
        "status": result.status,
        "diffs": [
            {"field": d.field, "original": d.original, "replayed": d.replayed}
            for d in result.diffs
        ],
        "original_context_count": len(result.original_contexts),
        "replayed_context_count": len(result.replayed_contexts),
        "original_answer_preview": (result.original_answer or "")[:200],
        "replayed_answer_preview": (result.replayed_answer or "")[:200],
    }


@router.get("/debug/storage", dependencies=[Depends(require_api_key)])
async def storage_debug(request: Request) -> dict[str, object]:
    container = request.app.state.container
    parsed_dir = container.config.parsed_dir
    parsed_files = (
        sorted(path.name for path in parsed_dir.glob("*.json"))
        if parsed_dir.exists()
        else []
    )
    return {
        "vectorstore": container.repository.stats(),
        "parsed_dir": str(parsed_dir),
        "parsed_files": parsed_files,
    }


@router.get("/debug/parsed/{doc_id}", dependencies=[Depends(require_api_key)])
async def parsed_document_debug(doc_id: str, request: Request) -> dict:
    if not re.match(r"^[a-zA-Z0-9_-]+$", doc_id):
        raise HTTPException(
            status_code=400,
            detail="doc_id must contain only alphanumeric characters, hyphens, and underscores",
        )
    container = request.app.state.container
    target = container.config.parsed_dir / f"{doc_id}.json"
    if not target.exists():
        raise HTTPException(
            status_code=404, detail=f"parsed artifact not found: {doc_id}"
        )
    return json.loads(target.read_text(encoding="utf-8"))
