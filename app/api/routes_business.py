from __future__ import annotations

import os
from time import time
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.api.auth import require_api_key
from app.schemas.benchmark import BenchmarkRunRequest, BenchmarkRunResponse
from app.schemas.business import (
    BusinessChatRequest,
    BusinessChatResponse,
    BusinessIngestRequest,
    BusinessIngestResponse,
    FeedbackRequest,
    FeedbackResponse,
)
from app.schemas.chat import ChatRequest
from app.schemas.trace import TraceRecord
from app.eval.dataset import (
    get_benchmark_report_dir,
    load_jsonl_dataset,
    resolve_benchmark_report_path,
    resolve_eval_dataset_path,
    save_json,
)
from app.eval.service import materialize_eval_records
from app.benchmark.service import build_benchmark_report

router = APIRouter(prefix="/v1/rag", tags=["rag"])


def _get_supported_kb_ids() -> set[str]:
    raw = os.getenv("RAG_SUPPORTED_KB_IDS", "default")
    return {item.strip() for item in raw.split(",") if item.strip()}


SUPPORTED_KB_IDS = _get_supported_kb_ids()


def _ensure_supported_kb_id(kb_id: str) -> None:
    if kb_id not in SUPPORTED_KB_IDS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"kb_id '{kb_id}' is not supported. "
                f"Supported kb_ids: {', '.join(sorted(SUPPORTED_KB_IDS))}. "
                "Set RAG_SUPPORTED_KB_IDS env var to configure additional knowledge bases."
            ),
        )


def _ensure_trace_scope(
    trace: TraceRecord,
    kb_id: str,
    tenant_id: str | None = None,
    session_id: str | None = None,
) -> None:
    trace_kb_id = trace.kb_id or "default"
    if trace_kb_id != kb_id:
        raise HTTPException(
            status_code=403, detail="trace does not belong to the requested kb_id"
        )
    if (trace.tenant_id or None) != (tenant_id or None):
        raise HTTPException(
            status_code=403, detail="trace does not belong to the requested tenant_id"
        )
    if trace.session_id and trace.session_id != session_id:
        raise HTTPException(
            status_code=403, detail="trace does not belong to the requested session_id"
        )


@router.post(
    "/chat",
    response_model=BusinessChatResponse,
    dependencies=[Depends(require_api_key)],
)
async def rag_chat(
    payload: BusinessChatRequest, request: Request
) -> BusinessChatResponse:
    _ensure_supported_kb_id(payload.kb_id)
    container = request.app.state.container
    response = await container.chat_pipeline.run(
        ChatRequest(
            query=payload.query,
            top_k=payload.top_k,
            kb_id=payload.kb_id,
            tenant_id=payload.tenant_id,
            session_id=payload.session_id,
            metadata_filters=payload.metadata_filters,
        )
    )
    return BusinessChatResponse(
        answer=response.answer,
        citations=response.citations,
        contexts=response.contexts,
        trace_id=response.trace_id,
        kb_id=payload.kb_id,
        tenant_id=payload.tenant_id,
        session_id=payload.session_id,
    )


@router.post(
    "/ingest",
    response_model=BusinessIngestResponse,
    dependencies=[Depends(require_api_key)],
)
async def rag_ingest(
    payload: BusinessIngestRequest, request: Request
) -> BusinessIngestResponse:
    _ensure_supported_kb_id(payload.kb_id)
    container = request.app.state.container
    response = await container.ingestion_pipeline.run(
        payload.path,
        kb_id=payload.kb_id,
        tenant_id=payload.tenant_id,
    )
    return BusinessIngestResponse(
        status="ok",
        kb_id=payload.kb_id,
        tenant_id=payload.tenant_id,
        documents=response.documents,
        chunks=response.chunks,
    )


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    dependencies=[Depends(require_api_key)],
)
async def rag_feedback(payload: FeedbackRequest, request: Request) -> FeedbackResponse:
    _ensure_supported_kb_id(payload.kb_id)
    container = request.app.state.container
    trace = container.trace_store.get(payload.trace_id)
    if trace is None:
        raise HTTPException(
            status_code=404, detail=f"trace not found: {payload.trace_id}"
        )
    _ensure_trace_scope(trace, payload.kb_id, payload.tenant_id, payload.session_id)
    feedback_id = f"fb-{uuid4().hex[:16]}"
    container.feedback_store.save(
        {
            "feedback_id": feedback_id,
            "trace_id": payload.trace_id,
            "rating": payload.rating,
            "kb_id": payload.kb_id,
            "tenant_id": payload.tenant_id,
            "session_id": payload.session_id,
            "comment": payload.comment,
            "tags": payload.tags,
            "created_at": time(),
        }
    )
    return FeedbackResponse(status="ok", feedback_id=feedback_id)


@router.get(
    "/traces/{trace_id}",
    response_model=TraceRecord,
    dependencies=[Depends(require_api_key)],
)
async def rag_trace(
    trace_id: str,
    request: Request,
    kb_id: str = Query(default="default"),
    tenant_id: str | None = Query(default=None),
    session_id: str | None = Query(default=None),
) -> TraceRecord:
    _ensure_supported_kb_id(kb_id)
    container = request.app.state.container
    record = container.trace_store.get(trace_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"trace not found: {trace_id}")
    _ensure_trace_scope(record, kb_id, tenant_id, session_id)
    return record


@router.post(
    "/benchmark/run",
    response_model=BenchmarkRunResponse,
    dependencies=[Depends(require_api_key)],
)
async def rag_benchmark(
    payload: BenchmarkRunRequest, request: Request
) -> BenchmarkRunResponse:
    container = request.app.state.container
    try:
        dataset_path = resolve_eval_dataset_path(payload.dataset_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    dataset = load_jsonl_dataset(str(dataset_path))
    evaluated_records = await materialize_eval_records(container, dataset)
    eval_report = container.ragas_runner.run(evaluated_records)
    benchmark_report = build_benchmark_report(
        dataset_path=str(dataset_path),
        eval_report=eval_report,
        trace_store=container.trace_store,
        diagnosis_service=container.diagnosis_service,
    )
    if payload.output_path:
        try:
            output_path = str(resolve_benchmark_report_path(payload.output_path))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        report_dir = get_benchmark_report_dir()
        report_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(report_dir / f"{uuid4().hex[:12]}_benchmark.json")
    save_json(output_path, benchmark_report)
    return BenchmarkRunResponse(
        status="ok", output_path=output_path, report=benchmark_report
    )
