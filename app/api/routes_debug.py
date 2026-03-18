from fastapi import APIRouter, HTTPException, Request

from app.schemas.chat import ChatRequest
from app.schemas.trace import RetrievalDebugResponse, TraceRecord, TraceSummary

router = APIRouter()


@router.post("/retrieve/debug", response_model=RetrievalDebugResponse)
async def retrieve_debug(payload: ChatRequest, request: Request) -> RetrievalDebugResponse:
    container = request.app.state.container
    return await container.retrieval_pipeline.debug(payload.query, payload.top_k)


@router.get("/traces", response_model=list[TraceSummary])
async def list_traces(request: Request) -> list[TraceSummary]:
    container = request.app.state.container
    return container.trace_store.list()


@router.get("/traces/{trace_id}", response_model=TraceRecord)
async def get_trace(trace_id: str, request: Request) -> TraceRecord:
    container = request.app.state.container
    record = container.trace_store.get(trace_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"trace not found: {trace_id}")
    return record
