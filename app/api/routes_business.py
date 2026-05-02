from __future__ import annotations

import json
import logging
import os
import re
import shutil
from pathlib import Path
from time import time
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile

from app.api.auth import RequestContext, require_api_key
from app.ingestion.loader import IngestPathError, SUPPORTED_EXTENSIONS, list_allowed_ingest_sources
from app.ingestion.pipeline import ChunkConfigurationError
from app.schemas.benchmark import BenchmarkRunRequest, BenchmarkRunResponse
from app.schemas.business import (
    BusinessChatRequest,
    BusinessChatResponse,
    BusinessDocumentSummary,
    BusinessIngestRequest,
    BusinessIngestResponse,
    BusinessRetrieveRequest,
    BusinessRetrieveResponse,
    FeedbackRequest,
    FeedbackResponse,
    IngestSourceSummary,
    KnowledgeBaseCreateRequest,
    KnowledgeBaseSummary,
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
logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
MAX_FILES_PER_BATCH = int(os.getenv("MAX_FILES_PER_BATCH", "10"))
UPLOAD_CHUNK_BYTES = 1024 * 1024
SAFE_PATH_COMPONENT_RE = re.compile(r"[^\w.-]+", re.UNICODE)


def _safe_path_component(value: str | None, default: str) -> str:
    raw = str(value or "").strip()
    safe = SAFE_PATH_COMPONENT_RE.sub("_", raw).strip("._-")
    if not safe or safe in {".", ".."}:
        return default
    return safe[:160]


def _safe_upload_filename(original_name: str) -> str:
    return _safe_path_component(Path(original_name or "upload.txt").name, "upload.txt")


def _build_upload_source_path(original_name: str, kb_id: str) -> str:
    return (
        Path("uploads")
        / _safe_path_component(kb_id, "default")
        / _safe_upload_filename(original_name)
    ).as_posix()


def _upload_storage_path(upload_dir: Path, source_path: str) -> Path:
    path = Path(source_path)
    if not path.parts or path.parts[0] != "uploads":
        raise ValueError(f"upload source path must start with uploads/: {source_path}")
    return upload_dir.joinpath(*path.parts[1:])


def _discard_path(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Failed to remove temporary upload file %s: %s", path, exc)


def _ensure_kb_access(container, kb_id: str, context: RequestContext | None = None) -> None:  # noqa: ANN001
    if context and context.allowed_kb_ids is not None and kb_id not in context.allowed_kb_ids:
        raise HTTPException(status_code=403, detail="knowledge base is not accessible")
    if not container.knowledge_base_catalog.exists(kb_id):
        raise HTTPException(
            status_code=404,
            detail=f"knowledge base not found: {kb_id}",
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


def _list_scope_documents(
    parsed_dir: Path, kb_id: str
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


def _list_knowledge_bases(
    container, context: RequestContext | None = None  # noqa: ANN001
) -> list[KnowledgeBaseSummary]:
    records = container.knowledge_base_catalog.list(
        allowed_kb_ids=context.allowed_kb_ids if context else None
    )
    summaries = {
        record.kb_id: KnowledgeBaseSummary(
            **record.model_dump(),
        )
        for record in records
    }

    parsed_dir = container.config.parsed_dir
    if parsed_dir.exists():
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
            kb_id = str(metadata.get("kb_id", "default"))
            if kb_id not in summaries:
                continue
            summary = summaries[kb_id]
            summary.document_count += 1
            summary.chunk_count += len(chunks) if isinstance(chunks, list) else 0
            updated_at = artifact.stat().st_mtime
            if summary.last_activity_at is None or updated_at > summary.last_activity_at:
                summary.last_activity_at = updated_at

    traces = container.trace_store.list(page=1, page_size=100)
    for trace in traces.items:
        kb_id = trace.kb_id or "default"
        if kb_id in summaries:
            summaries[kb_id].trace_count += 1

    return sorted(
        summaries.values(),
        key=lambda item: (
            -(item.last_activity_at or item.updated_at or 0),
            item.kb_id,
        ),
    )


@router.post(
    "/chat",
    response_model=BusinessChatResponse,
)
async def rag_chat(
    payload: BusinessChatRequest,
    request: Request,
    context: RequestContext = Depends(require_api_key),
) -> BusinessChatResponse:
    container = request.app.state.container
    _ensure_kb_access(container, payload.kb_id, context)
    response = await container.chat_pipeline.run(
        ChatRequest(
            query=payload.query,
            top_k=payload.top_k,
            kb_id=payload.kb_id,
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
        session_id=payload.session_id,
    )


@router.post(
    "/retrieve",
    response_model=BusinessRetrieveResponse,
)
async def rag_retrieve(
    payload: BusinessRetrieveRequest,
    request: Request,
    context: RequestContext = Depends(require_api_key),
) -> BusinessRetrieveResponse:
    container = request.app.state.container
    _ensure_kb_access(container, payload.kb_id, context)
    response = await container.retrieval_pipeline.debug(
        payload.query,
        payload.top_k,
        kb_id=payload.kb_id,
        session_id=payload.session_id,
        metadata_filters=payload.metadata_filters,
    )
    return BusinessRetrieveResponse(
        query=response.query,
        contexts=response.contexts,
        retrieved=response.retrieved,
        reranked=response.reranked,
        trace_id=response.trace_id,
        kb_id=payload.kb_id,
        session_id=payload.session_id,
    )


@router.post(
    "/ingest",
    response_model=BusinessIngestResponse,
)
async def rag_ingest(
    payload: BusinessIngestRequest,
    request: Request,
    context: RequestContext = Depends(require_api_key),
) -> BusinessIngestResponse:
    container = request.app.state.container
    _ensure_kb_access(container, payload.kb_id, context)
    try:
        response = await container.ingestion_pipeline.run(
            payload.path,
            kb_id=payload.kb_id,
        )
    except IngestPathError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ChunkConfigurationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return BusinessIngestResponse(
        status="ok",
        kb_id=payload.kb_id,
        documents=response.documents,
        chunks=response.chunks,
        source="path",
    )


@router.get("/knowledge-bases", response_model=list[KnowledgeBaseSummary])
async def rag_knowledge_bases(
    request: Request,
    context: RequestContext = Depends(require_api_key),
) -> list[KnowledgeBaseSummary]:
    return _list_knowledge_bases(request.app.state.container, context)


@router.post("/knowledge-bases", response_model=KnowledgeBaseSummary)
async def rag_create_knowledge_base(
    payload: KnowledgeBaseCreateRequest,
    request: Request,
    context: RequestContext = Depends(require_api_key),
) -> KnowledgeBaseSummary:
    if context.allowed_kb_ids is not None and payload.kb_id not in context.allowed_kb_ids:
        raise HTTPException(status_code=403, detail="knowledge base is not accessible")
    try:
        record = request.app.state.container.knowledge_base_catalog.create(
            kb_id=payload.kb_id,
            name=payload.name,
            description=payload.description,
            source=payload.source,
            external_ref=payload.external_ref,
            metadata=payload.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return KnowledgeBaseSummary(**record.model_dump())


@router.get(
    "/ingest/sources",
    response_model=list[IngestSourceSummary],
    dependencies=[Depends(require_api_key)],
)
async def rag_ingest_sources() -> list[IngestSourceSummary]:
    return [
        IngestSourceSummary.model_validate(source)
        for source in list_allowed_ingest_sources()
    ]


@router.post(
    "/ingest/upload",
    response_model=BusinessIngestResponse,
)
async def rag_ingest_upload(
    request: Request,
    files: list[UploadFile] = File(...),
    kb_id: str = Form(default="default"),
    context: RequestContext = Depends(require_api_key),
) -> BusinessIngestResponse:
    _ensure_kb_access(request.app.state.container, kb_id, context)
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
    durable_uploads: list[tuple[Path, Path]] = []
    staged_uploads: list[tuple[Path, Path]] = []
    finalized_uploads: list[tuple[Path, Path | None]] = []
    seen_upload_names: set[str] = set()
    seen_source_paths: set[str] = set()
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
            source_path = _build_upload_source_path(original_name, kb_id)
            if source_path in seen_source_paths:
                raise HTTPException(
                    status_code=400,
                    detail=f"duplicate upload source path '{source_path}' in the same request",
                )
            seen_source_paths.add(source_path)
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
            source_path_overrides[str(target.resolve())] = source_path
            durable_uploads.append(
                (target, _upload_storage_path(container.config.upload_dir, source_path))
            )

        for temporary_path, durable_path in durable_uploads:
            durable_path.parent.mkdir(parents=True, exist_ok=True)
            pending_path = durable_path.with_name(
                f".{durable_path.name}.{uuid4().hex[:8]}.tmp"
            )
            staged_uploads.append((pending_path, durable_path))
            shutil.copy2(temporary_path, pending_path)

        try:
            for pending_path, durable_path in staged_uploads:
                backup_path: Path | None = None
                if durable_path.exists():
                    backup_path = durable_path.with_name(
                        f".{durable_path.name}.{uuid4().hex[:8]}.bak"
                    )
                    durable_path.replace(backup_path)
                try:
                    pending_path.replace(durable_path)
                except Exception:
                    if backup_path is not None and backup_path.exists():
                        backup_path.replace(durable_path)
                    raise
                finalized_uploads.append((durable_path, backup_path))
        except Exception:
            for durable_path, backup_path in reversed(finalized_uploads):
                durable_path.unlink(missing_ok=True)
                if backup_path is not None and backup_path.exists():
                    backup_path.replace(durable_path)
            raise
        try:
            response = await container.ingestion_pipeline.run(
                str(upload_batch_dir),
                kb_id=kb_id,
                source_path_overrides=source_path_overrides,
            )
        except Exception:
            for durable_path, backup_path in reversed(finalized_uploads):
                durable_path.unlink(missing_ok=True)
                if backup_path is not None and backup_path.exists():
                    backup_path.replace(durable_path)
            raise
        for _, backup_path in finalized_uploads:
            if backup_path is not None:
                _discard_path(backup_path)
        return BusinessIngestResponse(
            status="ok",
            kb_id=kb_id,
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
        for pending_path, _ in staged_uploads:
            _discard_path(pending_path)
        for _, backup_path in finalized_uploads:
            if backup_path is not None:
                _discard_path(backup_path)
        shutil.rmtree(upload_batch_dir, ignore_errors=True)


@router.get(
    "/documents",
    response_model=list[BusinessDocumentSummary],
)
async def rag_documents(
    request: Request,
    kb_id: str = Query(default="default"),
    context: RequestContext = Depends(require_api_key),
) -> list[BusinessDocumentSummary]:
    container = request.app.state.container
    _ensure_kb_access(container, kb_id, context)
    return _list_scope_documents(container.config.parsed_dir, kb_id=kb_id)


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
)
async def rag_feedback(
    payload: FeedbackRequest,
    request: Request,
    context: RequestContext = Depends(require_api_key),
) -> FeedbackResponse:
    container = request.app.state.container
    _ensure_kb_access(container, payload.kb_id, context)
    trace = container.trace_store.get(payload.trace_id)
    if trace is None:
        raise HTTPException(
            status_code=404, detail=f"trace not found: {payload.trace_id}"
        )
    _ensure_trace_scope(trace, payload.kb_id, payload.session_id)
    feedback_id = f"fb-{uuid4().hex[:16]}"
    container.feedback_store.save_raw(
        {
            "feedback_id": feedback_id,
            "trace_id": payload.trace_id,
            "rating": payload.rating,
            "kb_id": payload.kb_id,
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
)
async def rag_trace(
    trace_id: str,
    request: Request,
    kb_id: str = Query(default="default"),
    session_id: str | None = Query(default=None),
    context: RequestContext = Depends(require_api_key),
) -> TraceRecord:
    container = request.app.state.container
    _ensure_kb_access(container, kb_id, context)
    record = container.trace_store.get(trace_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"trace not found: {trace_id}")
    _ensure_trace_scope(record, kb_id, session_id)
    return record


@router.post(
    "/benchmark/run",
    response_model=BenchmarkRunResponse,
)
async def rag_benchmark(
    payload: BenchmarkRunRequest,
    request: Request,
    context: RequestContext = Depends(require_api_key),
) -> BenchmarkRunResponse:
    container = request.app.state.container
    ragas_runner = _require_eval_runner(container)
    diagnosis_service = _require_diagnosis_service(container)
    try:
        dataset_path = resolve_eval_dataset_path(payload.dataset_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    dataset = load_jsonl_dataset(str(dataset_path))
    for record in dataset:
        kb_id = str(record.get("kb_id", "default") or "default")
        _ensure_kb_access(container, kb_id, context)
    evaluated_records = await materialize_eval_records(container, dataset)
    eval_report = await _run_eval_report(
        ragas_runner, evaluated_records, payload.use_ragas_lib
    )
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
