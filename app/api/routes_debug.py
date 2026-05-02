import json
import re
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.api.auth import RequestContext, require_api_key
from app.eval.dataset import (
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


def _ensure_kb_access(
    container,
    kb_id: str,
    context: RequestContext | None = None,
) -> None:  # noqa: ANN001
    allowed_kb_ids = getattr(context, "allowed_kb_ids", None)
    if allowed_kb_ids is not None and kb_id not in allowed_kb_ids:
        raise HTTPException(status_code=403, detail="knowledge base is not accessible")
    catalog = getattr(container, "knowledge_base_catalog", None)
    if catalog is not None and not catalog.exists(kb_id):
        raise HTTPException(
            status_code=404, detail=f"knowledge base not found: {kb_id}"
        )


def _ensure_trace_scope(
    trace: TraceRecord,
    kb_id: str,
    session_id: str | None = None,
) -> None:
    trace_kb_id = trace.kb_id or "default"
    if trace_kb_id != kb_id:
        raise HTTPException(
            status_code=403, detail="trace does not belong to the requested kb_id"
        )
    if trace.session_id and trace.session_id != session_id:
        raise HTTPException(
            status_code=403, detail="trace does not belong to the requested session_id"
        )


def _allowed_kb_ids_for_context(
    container,
    context: RequestContext | None,
) -> set[str] | None:  # noqa: ANN001
    allowed_kb_ids = getattr(context, "allowed_kb_ids", None)
    if allowed_kb_ids is None:
        return None
    catalog = getattr(container, "knowledge_base_catalog", None)
    if catalog is None:
        return set(allowed_kb_ids)
    return {
        kb_id
        for kb_id in allowed_kb_ids
        if catalog.exists(kb_id)
    }


def _record_kb_allowed(
    record: TraceSummary | TraceRecord, allowed_kb_ids: set[str] | None
) -> bool:
    if allowed_kb_ids is None:
        return True
    return (record.kb_id or "default") in allowed_kb_ids


def _optional_query_value(value: object) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return None


def _int_query_value(value: object, default: int) -> int:
    if isinstance(value, int):
        return value
    return default


def _result_kb_id(result: object, container) -> str | None:  # noqa: ANN001
    if not isinstance(result, dict):
        return None
    explicit_kb_id = result.get("kb_id")
    if explicit_kb_id:
        return str(explicit_kb_id)
    trace_id = result.get("trace_id")
    if trace_id:
        trace_store = getattr(container, "trace_store", None)
        trace = trace_store.get(str(trace_id)) if trace_store is not None else None
        if trace is not None:
            return trace.kb_id or "default"
    return None


def _numeric_result_aggregate(results: list[dict]) -> dict[str, float]:
    numeric_values: dict[str, list[float]] = {}
    for result in results:
        for key, value in result.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            numeric_values.setdefault(key, []).append(float(value))
    return {
        key: round(sum(values) / len(values), 4)
        for key, values in numeric_values.items()
        if values
    }


def _diagnosis_counts(results: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        diagnosis = result.get("diagnosis")
        findings = diagnosis.get("findings", []) if isinstance(diagnosis, dict) else []
        for finding in findings:
            if not isinstance(finding, dict):
                continue
            category = finding.get("category")
            if category:
                counts[str(category)] = counts.get(str(category), 0) + 1
    return counts


def _filter_report_for_context(
    report: dict,
    container,
    context: RequestContext | None,
) -> dict:  # noqa: ANN001
    allowed_kb_ids = _allowed_kb_ids_for_context(container, context)
    if allowed_kb_ids is None:
        return report
    results = report.get("results", []) if isinstance(report, dict) else []
    filtered_results = [
        dict(result)
        for result in results
        if isinstance(result, dict)
        and _result_kb_id(result, container) in allowed_kb_ids
    ]
    filtered_report = dict(report)
    filtered_report["results"] = filtered_results
    filtered_report["records"] = len(filtered_results)
    filtered_report["aggregate"] = _numeric_result_aggregate(filtered_results)
    if "diagnosis_counts" in filtered_report:
        filtered_report["diagnosis_counts"] = _diagnosis_counts(filtered_results)
    return filtered_report


def _scope_report_summary(
    summary: dict[str, object],
    container,
    context: RequestContext,
    resolver,
) -> dict[str, object]:  # noqa: ANN001
    allowed_kb_ids = _allowed_kb_ids_for_context(container, context)
    if allowed_kb_ids is None:
        return summary
    try:
        report_path = resolver(str(summary["path"]))
    except (KeyError, ValueError):
        return {**summary, "records": 0, "aggregate": {}}
    report = _filter_report_for_context(load_json(str(report_path)), container, context)
    return {
        **summary,
        "records": int(report.get("records", 0)),
        "aggregate": report.get("aggregate", {}),
    }


async def _run_eval_report(
    ragas_runner: object, records: list[dict], use_ragas_lib: bool
) -> dict:
    if use_ragas_lib:
        run_async = getattr(ragas_runner, "run_async", None)
        if run_async is None:
            raise HTTPException(
                status_code=503,
                detail="RAGAS library evaluation is not available in this runner.",
            )
        return await run_async(records)
    return ragas_runner.run(records)


@router.post(
    "/retrieve/debug",
    response_model=RetrievalDebugResponse,
)
async def retrieve_debug(
    payload: ChatRequest,
    request: Request,
    context: RequestContext = Depends(require_api_key),
) -> RetrievalDebugResponse:
    container = request.app.state.container
    kb_id = payload.kb_id or "default"
    _ensure_kb_access(container, kb_id, context)
    return await container.retrieval_pipeline.debug(
        payload.query,
        payload.top_k,
        kb_id=kb_id,
        session_id=payload.session_id,
    )


@router.get(
    "/traces",
    response_model=PaginatedResponse[TraceSummary],
)
async def list_traces(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    kb_id: str | None = Query(default=None),
    context: RequestContext = Depends(require_api_key),
) -> PaginatedResponse[TraceSummary]:
    container = request.app.state.container
    page = _int_query_value(page, 1)
    page_size = _int_query_value(page_size, 20)
    kb_id = _optional_query_value(kb_id)
    if kb_id is not None:
        _ensure_kb_access(container, kb_id, context)
        return container.trace_store.list(
            page=page, page_size=page_size, kb_id=kb_id
        )

    allowed_kb_ids = _allowed_kb_ids_for_context(container, context)
    if allowed_kb_ids is None:
        return container.trace_store.list(
            page=page, page_size=page_size, kb_id=kb_id
        )
    # Keep pagination scoped to authorized records inside TraceStore so large
    # unauthorized slices cannot hide later authorized records.
    return container.trace_store.list(
        page=page, page_size=page_size, kb_ids=allowed_kb_ids
    )


@router.get(
    "/traces/{trace_id}",
    response_model=TraceRecord,
)
async def get_trace(
    trace_id: str,
    request: Request,
    kb_id: str | None = Query(default=None),
    session_id: str | None = Query(default=None),
    context: RequestContext = Depends(require_api_key),
) -> TraceRecord:
    container = request.app.state.container
    kb_id = _optional_query_value(kb_id)
    session_id = _optional_query_value(session_id)
    record = container.trace_store.get(trace_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"trace not found: {trace_id}")
    resolved_kb_id = kb_id or record.kb_id or "default"
    _ensure_kb_access(container, resolved_kb_id, context)
    _ensure_trace_scope(record, resolved_kb_id, session_id)
    return record


@router.get("/eval/datasets", dependencies=[Depends(require_api_key)])
async def get_eval_datasets() -> list[dict[str, object]]:
    return list_eval_datasets()


@router.get("/eval/reports")
async def get_eval_reports(
    request: Request,
    context: RequestContext = Depends(require_api_key),
) -> list[dict[str, object]]:
    container = request.app.state.container
    return [
        _scope_report_summary(report, container, context, resolve_eval_report_path)
        for report in list_eval_reports()
    ]


@router.get("/benchmark/reports")
async def get_benchmark_reports(
    request: Request,
    context: RequestContext = Depends(require_api_key),
) -> list[dict[str, object]]:
    container = request.app.state.container
    return [
        _scope_report_summary(report, container, context, resolve_benchmark_report_path)
        for report in list_benchmark_reports()
    ]


@router.get("/eval/reports/detail")
async def get_eval_report_detail(
    request: Request,
    path: str = Query(...),
    context: RequestContext = Depends(require_api_key),
) -> dict:
    try:
        target = resolve_eval_report_path(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"eval report not found: {path}")
    return _filter_report_for_context(
        load_json(str(target)), request.app.state.container, context
    )


@router.get("/benchmark/reports/detail")
async def get_benchmark_report_detail(
    request: Request,
    path: str = Query(...),
    context: RequestContext = Depends(require_api_key),
) -> dict:
    try:
        target = resolve_benchmark_report_path(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not target.exists():
        raise HTTPException(
            status_code=404, detail=f"benchmark report not found: {path}"
        )
    return _filter_report_for_context(
        load_json(str(target)), request.app.state.container, context
    )


@router.post(
    "/eval/run", response_model=EvalRunResponse
)
async def run_eval(
    payload: EvalRunRequest,
    request: Request,
    context: RequestContext = Depends(require_api_key),
) -> EvalRunResponse:
    container = request.app.state.container
    ragas_runner = _require_eval_runner(container)
    try:
        dataset_path = resolve_eval_dataset_path(payload.dataset_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    dataset = load_jsonl_dataset(str(dataset_path))
    for record in dataset:
        kb_id = str(record.get("kb_id", "default") or "default")
        _ensure_kb_access(container, kb_id, context)
    evaluated_records = await materialize_eval_records(container, dataset)
    report = await _run_eval_report(
        ragas_runner, evaluated_records, payload.use_ragas_lib
    )
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
)
async def diagnose_trace(
    payload: TraceDiagnosisRequest,
    request: Request,
    context: RequestContext = Depends(require_api_key),
) -> DiagnosisResponse:
    container = request.app.state.container
    diagnosis_service = _require_diagnosis_service(container)
    trace = container.trace_store.get(payload.trace_id)
    if trace is None:
        raise HTTPException(
            status_code=404, detail=f"trace not found: {payload.trace_id}"
        )
    kb_id = trace.kb_id or "default"
    _ensure_kb_access(container, kb_id, context)
    _ensure_trace_scope(trace, kb_id, payload.session_id)
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
)
async def diagnose_eval(
    payload: EvalDiagnosisRequest,
    request: Request,
    context: RequestContext = Depends(require_api_key),
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
    report = _filter_report_for_context(
        load_json(str(report_path)), container, context
    )
    try:
        result = report.get("results", [])[payload.result_index]
        trace_id = result.get("trace_id") if isinstance(result, dict) else None
        if trace_id:
            trace = container.trace_store.get(str(trace_id))
            if trace is None:
                raise HTTPException(
                    status_code=404, detail=f"trace not found: {trace_id}"
                )
            kb_id = trace.kb_id or "default"
            session_id = (
                str(result["session_id"])
                if isinstance(result, dict) and result.get("session_id")
                else None
            )
            _ensure_kb_access(container, kb_id, context)
            _ensure_trace_scope(trace, kb_id, session_id)
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
)
async def diagnose_auto(
    payload: AutoDiagnosisRequest,
    request: Request,
    context: RequestContext = Depends(require_api_key),
) -> DiagnosisResponse:
    container = request.app.state.container
    diagnosis_service = _require_diagnosis_service(container)

    allowed_kb_ids = _allowed_kb_ids_for_context(container, context)
    reports = list_eval_reports()
    max_iterations = 10
    if reports:
        latest_report = reports[0]
        try:
            latest_report_path = resolve_eval_report_path(str(latest_report["path"]))
        except ValueError:
            latest_report_path = None
        if latest_report_path is None:
            report = {}
        else:
            report = load_json(str(latest_report_path))
        report = _filter_report_for_context(report, container, context)
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

    if allowed_kb_ids is None:
        traces = container.trace_store.list(page=1, page_size=1)
    else:
        traces = container.trace_store.list(
            page=1, page_size=1, kb_ids=allowed_kb_ids
        )
    if traces.items:
        scoped_traces = traces.items
    else:
        scoped_traces = []
    if scoped_traces:
        latest_trace_id = scoped_traces[0].trace_id
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
)
async def replay_trace_endpoint(
    trace_id: str,
    request: Request,
    kb_id: str | None = Query(default=None),
    session_id: str | None = Query(default=None),
    context: RequestContext = Depends(require_api_key),
) -> dict:
    container = request.app.state.container
    kb_id = _optional_query_value(kb_id)
    session_id = _optional_query_value(session_id)
    trace = container.trace_store.get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"trace not found: {trace_id}")
    resolved_kb_id = kb_id or trace.kb_id or "default"
    _ensure_kb_access(container, resolved_kb_id, context)
    _ensure_trace_scope(trace, resolved_kb_id, session_id)
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


@router.get("/debug/storage")
async def storage_debug(
    request: Request,
    context: RequestContext = Depends(require_api_key),
) -> dict[str, object]:
    container = request.app.state.container
    parsed_dir = container.config.parsed_dir
    allowed_kb_ids = _allowed_kb_ids_for_context(container, context)
    parsed_files: list[str] = []
    if parsed_dir.exists():
        for path in sorted(parsed_dir.glob("*.json")):
            if allowed_kb_ids is None:
                parsed_files.append(path.name)
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            document = payload.get("document") if isinstance(payload, dict) else None
            metadata = document.get("metadata", {}) if isinstance(document, dict) else {}
            if metadata.get("kb_id", "default") in allowed_kb_ids:
                parsed_files.append(path.name)
    return {
        "vectorstore": container.repository.stats(),
        "parsed_dir": str(parsed_dir),
        "parsed_files": parsed_files,
    }


@router.get("/debug/parsed/{doc_id}")
async def parsed_document_debug(
    doc_id: str,
    request: Request,
    context: RequestContext = Depends(require_api_key),
) -> dict:
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
    payload = json.loads(target.read_text(encoding="utf-8"))
    document = payload.get("document") if isinstance(payload, dict) else None
    metadata = document.get("metadata", {}) if isinstance(document, dict) else {}
    kb_id = str(metadata.get("kb_id", "default"))
    _ensure_kb_access(container, kb_id, context)
    return payload
