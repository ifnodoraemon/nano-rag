from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from time import time
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile

from app.api.auth import require_api_key
from app.ingestion.loader import IngestPathError, SUPPORTED_EXTENSIONS
from app.ingestion.pipeline import ChunkConfigurationError
from app.schemas.benchmark import BenchmarkRunRequest, BenchmarkRunResponse
from app.schemas.business import (
    BusinessChatRequest,
    BusinessChatResponse,
    BusinessDocumentSummary,
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

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
MAX_FILES_PER_BATCH = int(os.getenv("MAX_FILES_PER_BATCH", "10"))
UPLOAD_CHUNK_BYTES = 1024 * 1024


def _build_upload_source_path(
    original_name: str, kb_id: str, tenant_id: str | None = None
) -> str:
    tenant_scope = tenant_id or "__shared__"
    return (Path("uploads") / kb_id / tenant_scope / original_name).as_posix()


def _get_supported_kb_ids() -> set[str]:
    raw = os.getenv("RAG_SUPPORTED_KB_IDS", "default")
    return {item.strip() for item in raw.split(",") if item.strip()}


def _ensure_supported_kb_id(kb_id: str) -> None:
    supported = _get_supported_kb_ids()
    if kb_id not in supported:
        raise HTTPException(
            status_code=400,
            detail=(
                f"kb_id '{kb_id}' is not supported. "
                f"Supported kb_ids: {', '.join(sorted(supported))}. "
                "Set RAG_SUPPORTED_KB_IDS env var to configure additional knowledge bases."
            ),
        )


def _require_eval_runner(container) -> object:
    runner = getattr(container, "ragas_runner", None)
    if runner is None:
        raise HTTPException(
            status_code=503,
            detail="benchmark is disabled because evaluation is off. Set RAG_EVAL_ENABLED=true.",
        )
    return runner


def _require_diagnosis_service(container) -> object:
    service = getattr(container, "diagnosis_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="benchmark is disabled because diagnosis is off. Set RAG_DIAGNOSIS_ENABLED=true.",
        )
    return service


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


def _list_scope_documents(
    parsed_dir: Path, kb_id: str, tenant_id: str | None = None
) -> list[BusinessDocumentSummary]:
    if not parsed_dir.exists():
        return []
    documents: list[BusinessDocumentSummary] = []
    for artifact in sorted(parsed_dir.glob("*.json")):
        try:
            payload = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        document = payload.get("document") if isinstance(payload, dict) else None
        chunks = payload.get("chunks") if isinstance(payload, dict) else None
        if not isinstance(document, dict):
            continue
        metadata = document.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        if metadata.get("kb_id", "default") != kb_id:
            continue
        if metadata.get("tenant_id") != tenant_id:
            continue
        doc_id = str(document.get("doc_id", "")).strip()
        source_path = str(document.get("source_path", "")).strip()
        if not doc_id or not source_path:
            continue
        title = str(document.get("title", "")).strip() or Path(source_path).name
        documents.append(
            BusinessDocumentSummary(
                doc_id=doc_id,
                title=title,
                source_path=source_path,
                kb_id=kb_id,
                tenant_id=tenant_id,
                chunk_count=len(chunks) if isinstance(chunks, list) else 0,
                updated_at=artifact.stat().st_mtime,
                doc_type=(
                    str(metadata.get("doc_type")).strip()
                    if metadata.get("doc_type") is not None
                    else None
                ),
                source_key=(
                    str(metadata.get("source_key")).strip()
                    if metadata.get("source_key") is not None
                    else None
                ),
            )
        )
    return sorted(documents, key=lambda item: (-item.updated_at, item.title.lower()))


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
    try:
        response = await container.ingestion_pipeline.run(
            payload.path,
            kb_id=payload.kb_id,
            tenant_id=payload.tenant_id,
        )
    except IngestPathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ChunkConfigurationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return BusinessIngestResponse(
        status="ok",
        kb_id=payload.kb_id,
        tenant_id=payload.tenant_id,
        documents=response.documents,
        chunks=response.chunks,
        source="path",
    )


@router.post(
    "/ingest/upload",
    response_model=BusinessIngestResponse,
    dependencies=[Depends(require_api_key)],
)
async def rag_ingest_upload(
    request: Request,
    files: list[UploadFile] = File(...),
    kb_id: str = Form(default="default"),
    tenant_id: str | None = Form(default=None),
) -> BusinessIngestResponse:
    _ensure_supported_kb_id(kb_id)
    if not files:
        raise HTTPException(status_code=400, detail="at least one file is required")
    if len(files) > MAX_FILES_PER_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f"max {MAX_FILES_PER_BATCH} files per request",
        )

    container = request.app.state.container
    upload_batch_dir = container.config.upload_dir / uuid4().hex[:12]
    upload_batch_dir.mkdir(parents=True, exist_ok=True)

    uploaded_files: list[str] = []
    source_path_overrides: dict[str, str] = {}
    seen_upload_names: set[str] = set()
    try:
        for upload in files:
            original_name = Path(upload.filename or "upload.txt").name
            if original_name in seen_upload_names:
                raise HTTPException(
                    status_code=400,
                    detail=f"duplicate upload filename '{original_name}' in the same request",
                )
            seen_upload_names.add(original_name)
            extension = Path(original_name).suffix.lower()
            if extension not in SUPPORTED_EXTENSIONS:
                allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
                raise HTTPException(
                    status_code=400,
                    detail=f"unsupported file type '{extension or 'unknown'}'. Supported types: {allowed}",
                )
            target = upload_batch_dir / f"{uuid4().hex[:8]}_{original_name}"
            total_bytes = 0
            with target.open("wb") as output:
                while True:
                    chunk = await upload.read(UPLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    total_bytes += len(chunk)
                    if total_bytes > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"file '{original_name}' exceeds max size ({MAX_UPLOAD_BYTES} bytes)",
                        )
                    output.write(chunk)
            uploaded_files.append(original_name)
            source_path_overrides[str(target.resolve())] = _build_upload_source_path(
                original_name, kb_id, tenant_id
            )

        response = await container.ingestion_pipeline.run(
            str(upload_batch_dir),
            kb_id=kb_id,
            tenant_id=tenant_id,
            source_path_overrides=source_path_overrides,
        )
        return BusinessIngestResponse(
            status="ok",
            kb_id=kb_id,
            tenant_id=tenant_id,
            documents=response.documents,
            chunks=response.chunks,
            source="upload",
            uploaded_files=uploaded_files,
        )
    finally:
        for upload in files:
            try:
                await upload.close()
            except Exception:
                pass
        shutil.rmtree(upload_batch_dir, ignore_errors=True)


@router.get(
    "/documents",
    response_model=list[BusinessDocumentSummary],
    dependencies=[Depends(require_api_key)],
)
async def rag_documents(
    request: Request,
    kb_id: str = Query(default="default"),
    tenant_id: str | None = Query(default=None),
) -> list[BusinessDocumentSummary]:
    _ensure_supported_kb_id(kb_id)
    container = request.app.state.container
    return _list_scope_documents(
        container.config.parsed_dir, kb_id=kb_id, tenant_id=tenant_id
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
    container.feedback_store.save_raw(
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
    ragas_runner = _require_eval_runner(container)
    diagnosis_service = _require_diagnosis_service(container)
    try:
        dataset_path = resolve_eval_dataset_path(payload.dataset_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    dataset = load_jsonl_dataset(str(dataset_path))
    evaluated_records = await materialize_eval_records(container, dataset)
    eval_report = ragas_runner.run(evaluated_records)
    benchmark_report = build_benchmark_report(
        dataset_path=str(dataset_path),
        eval_report=eval_report,
        trace_store=container.trace_store,
        diagnosis_service=diagnosis_service,
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
