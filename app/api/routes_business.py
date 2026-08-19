from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from time import time
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

from app.api.auth import RequestContext, require_admin_key, require_api_key
from app.core.config import AppContainer
from app.diagnostics.service import DiagnosisService
from app.eval.deepeval_runner import DeepevalRunner
from app.ingestion.executor import submit_ingest_paths
from app.ingestion.loader import (
    SUPPORTED_EXTENSIONS,
    list_allowed_ingest_sources,
)
from app.schemas.benchmark import BenchmarkRunRequest, BenchmarkRunResponse
from app.schemas.business import (
    BusinessChatRequest,
    BusinessChatResponse,
    BusinessDocumentSummary,
    BusinessIngestJobResponse,
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
from app.schemas.structured import DocumentNode, StructuredDocument
from app.schemas.trace import TraceRecord
from app.retrieval.graph_index import GraphIndex
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


def _ensure_kb_access(container, kb_id: str, context: RequestContext | None = None) -> None:  # noqa: ANN001
    if context and context.allowed_kb_ids is not None and kb_id not in context.allowed_kb_ids:
        raise HTTPException(status_code=403, detail="knowledge base is not accessible")
    if not container.knowledge_base_catalog.exists(kb_id):
        raise HTTPException(
            status_code=404,
            detail=f"knowledge base not found: {kb_id}",
        )


def _require_eval_runner(container: AppContainer) -> DeepevalRunner:
    runner = getattr(container, "eval_runner", None)
    if runner is None:
        raise HTTPException(
            status_code=503,
            detail="benchmark is disabled because evaluation is off. Set RAG_EVAL_ENABLED=true.",
        )
    return runner


def _require_diagnosis_service(container: AppContainer) -> DiagnosisService:
    service = getattr(container, "diagnosis_service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="benchmark is disabled because diagnosis is off. Set RAG_DIAGNOSIS_ENABLED=true.",
        )
    return service


async def _run_eval_report(
    eval_runner: DeepevalRunner, records: list[dict], use_ragas_lib: bool
) -> dict:
    if use_ragas_lib:
        return await eval_runner.run_async(records)
    return eval_runner.run(records)


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


def _load_structured_document(parsed_dir: Path, doc_id: str) -> StructuredDocument | None:
    artifact = parsed_dir / f"{doc_id}.json"
    if not artifact.exists():
        return None
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    raw = payload.get("structured_document") if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        return None
    return StructuredDocument.model_validate(raw)


def _find_node(document: StructuredDocument, node_id: str) -> DocumentNode | None:
    return next((node for node in document.iter_nodes() if node.node_id == node_id), None)


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


@router.post("/chat/stream")
async def rag_chat_stream(
    payload: BusinessChatRequest,
    request: Request,
    context: RequestContext = Depends(require_api_key),
) -> StreamingResponse:
    """
    流式响应接口 (SSE)。
    支持通过 LangGraph stream_queue 进行打字机式流式输出。
    """
    container = request.app.state.container
    _ensure_kb_access(container, payload.kb_id, context)
    
    async def event_generator():
        import asyncio
        from time import perf_counter
        from app.schemas.chat import ChatResponse

        def _frame(payload: dict[str, object]) -> str:
            return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

        yield _frame({"status": "thinking", "message": "Retrieving and synthesizing..."})

        queue: "asyncio.Queue[object]" = asyncio.Queue()
        payload_copy = ChatRequest(
            query=payload.query,
            top_k=payload.top_k,
            kb_id=payload.kb_id,
            session_id=payload.session_id,
            metadata_filters=payload.metadata_filters,
        )

        state_input = {"payload": payload_copy, "started_at": perf_counter(), "stream_queue": queue}

        async def run_workflow():
            try:
                state = await container.chat_pipeline.workflow.ainvoke(state_input)
                await queue.put(state["response"])
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 - surfaced as a generic SSE error frame
                # Full detail stays server-side; only a safe message crosses the wire.
                logger.exception("chat stream workflow failed: %s", e)
                await queue.put(e)

        task = asyncio.create_task(run_workflow())

        try:
            while True:
                chunk = await queue.get()
                if isinstance(chunk, BaseException):
                    yield _frame({"status": "error", "message": "upstream generation error"})
                    break
                if isinstance(chunk, ChatResponse):
                    yield _frame(
                        {
                            "status": "success",
                            "answer": chunk.answer,
                            "trace_id": chunk.trace_id,
                            "citations": [c.model_dump() for c in chunk.citations],
                        }
                    )
                    break
                if isinstance(chunk, str):
                    yield _frame({"status": "generating", "chunk": chunk})
                # Fail loud on unexpected queue payloads instead of spinning
                # the consumer loop forever with a dead stream.
                logger.error("unexpected chat stream queue payload: %r", type(chunk))
                yield _frame({"status": "error", "message": "internal stream error"})
                break
        except asyncio.CancelledError:
            task.cancel()
            raise

    return StreamingResponse(event_generator(), media_type="text/event-stream")


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
    context: RequestContext = Depends(require_admin_key),
    background_tasks: BackgroundTasks = None,  # type: ignore
) -> BusinessIngestResponse:
    container = request.app.state.container
    _ensure_kb_access(container, payload.kb_id, context)
    job = container.ingest_job_store.create(
        kb_id=payload.kb_id,
        source="path",
        path=payload.path,
    )
    if background_tasks is None:
        background_tasks = BackgroundTasks()
    submit_ingest_paths(
        background_tasks=background_tasks,
        container=container,
        job_id=job.job_id,
        paths=[payload.path],
        kb_id=payload.kb_id,
    )
    return BusinessIngestResponse(
        status=job.status.value,
        stage=job.stage,
        job_id=job.job_id,
        kb_id=payload.kb_id,
        source="path",
    )


@router.get(
    "/ingest/jobs/{job_id}",
    response_model=BusinessIngestJobResponse,
)
async def rag_ingest_job(
    job_id: str,
    request: Request,
    context: RequestContext = Depends(require_api_key),
) -> BusinessIngestJobResponse:
    record = request.app.state.container.ingest_job_store.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"ingest job not found: {job_id}")
    _ensure_kb_access(request.app.state.container, record.kb_id, context)
    return BusinessIngestJobResponse(
        **record.model_dump(mode="json"),
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
    dependencies=[Depends(require_admin_key)],
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
    background_tasks: BackgroundTasks = None,  # type: ignore
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
    uploaded_files: list[str] = []
    source_paths_by_durable_path: dict[str, str] = {}
    durable_paths: list[str] = []
    staged_uploads: list[tuple[Path, Path, str, str]] = []
    temp_upload_paths: list[Path] = []
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
            source_path = _build_upload_source_path(original_name, kb_id)
            if source_path in seen_source_paths:
                raise HTTPException(
                    status_code=400,
                    detail=f"duplicate upload source path '{source_path}' in the same request",
                )
            seen_source_paths.add(source_path)
            durable_path = _upload_storage_path(container.config.upload_dir, source_path)
            durable_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = durable_path.with_name(
                f".{durable_path.name}.{uuid4().hex}.tmp"
            )
            temp_upload_paths.append(temp_path)
            total_bytes = 0
            with temp_path.open("wb") as output:
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
            staged_uploads.append((temp_path, durable_path, original_name, source_path))

        for temp_path, durable_path, original_name, source_path in staged_uploads:
            os.replace(temp_path, durable_path)
            uploaded_files.append(original_name)
            source_paths_by_durable_path[str(durable_path.resolve())] = source_path
            durable_paths.append(str(durable_path))

        durable_source_overrides = {
            str(Path(path).resolve()): source_paths_by_durable_path[str(Path(path).resolve())]
            for path in durable_paths
        }
        job = container.ingest_job_store.create(
            kb_id=kb_id,
            source="upload",
            path=",".join(durable_paths),
            uploaded_files=uploaded_files,
        )
        if background_tasks is None:
            background_tasks = BackgroundTasks()
        submit_ingest_paths(
            background_tasks=background_tasks,
            container=container,
            job_id=job.job_id,
            paths=durable_paths,
            kb_id=kb_id,
            source_path_overrides=durable_source_overrides,
        )
        return BusinessIngestResponse(
            status=job.status.value,
            stage=job.stage,
            job_id=job.job_id,
            kb_id=kb_id,
            source="upload",
            uploaded_files=uploaded_files,
        )
    finally:
        for temp_path in temp_upload_paths:
            temp_path.unlink(missing_ok=True)
        for upload in files:
            try:
                await upload.close()
            except Exception:
                pass


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


@router.get("/documents/{doc_id}/tree")
async def rag_document_tree(
    doc_id: str,
    request: Request,
    kb_id: str = Query(default="default"),
    context: RequestContext = Depends(require_api_key),
) -> dict:
    container = request.app.state.container
    _ensure_kb_access(container, kb_id, context)
    document = _load_structured_document(container.config.parsed_dir, doc_id)
    if document is None or document.kb_id != kb_id:
        raise HTTPException(status_code=404, detail=f"document tree not found: {doc_id}")
    return document.model_dump(mode="json")


@router.get("/nodes/{node_id}")
async def rag_node(
    node_id: str,
    request: Request,
    kb_id: str = Query(default="default"),
    context: RequestContext = Depends(require_api_key),
) -> dict:
    container = request.app.state.container
    _ensure_kb_access(container, kb_id, context)
    doc_id = node_id.split(":node:", 1)[0] if ":node:" in node_id else node_id.split(":", 1)[0]
    document = _load_structured_document(container.config.parsed_dir, doc_id)
    if document is None or document.kb_id != kb_id:
        raise HTTPException(status_code=404, detail=f"node not found: {node_id}")
    node = _find_node(document, node_id)
    if node is None:
        raise HTTPException(status_code=404, detail=f"node not found: {node_id}")
    return node.model_dump(mode="json")


@router.get("/tables/{table_id}")
async def rag_table(
    table_id: str,
    request: Request,
    kb_id: str = Query(default="default"),
    context: RequestContext = Depends(require_api_key),
) -> dict:
    container = request.app.state.container
    _ensure_kb_access(container, kb_id, context)
    doc_id = table_id.split(":node:", 1)[0] if ":node:" in table_id else table_id.split(":", 1)[0]
    document = _load_structured_document(container.config.parsed_dir, doc_id)
    if document is None or document.kb_id != kb_id:
        raise HTTPException(status_code=404, detail=f"table not found: {table_id}")
    node = _find_node(document, table_id)
    if node is None or node.table is None:
        raise HTTPException(status_code=404, detail=f"table not found: {table_id}")
    return {
        "table_id": table_id,
        "doc_id": document.doc_id,
        "source_path": document.source_path,
        "provenance": node.provenance.model_dump(mode="json"),
        "table": node.table.model_dump(mode="json"),
    }


@router.get("/graph/neighborhood")
async def rag_graph_neighborhood(
    request: Request,
    node_id: str = Query(...),
    kb_id: str = Query(default="default"),
    context: RequestContext = Depends(require_api_key),
) -> dict:
    container = request.app.state.container
    _ensure_kb_access(container, kb_id, context)
    neighborhood = GraphIndex(container.config.parsed_dir).neighborhood(
        node_id,
        kb_id=kb_id,
    )
    if neighborhood is None:
        raise HTTPException(status_code=404, detail=f"node not found: {node_id}")
    node, neighbors = neighborhood
    return {"node": node, "neighbors": neighbors}


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
    context: RequestContext = Depends(require_admin_key),
) -> BenchmarkRunResponse:
    container = request.app.state.container
    eval_runner = _require_eval_runner(container)
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
        eval_runner, evaluated_records, payload.use_ragas_lib
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
