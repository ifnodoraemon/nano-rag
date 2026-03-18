import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from app.eval.dataset import load_jsonl_dataset, save_json
from app.schemas.chat import ChatRequest
from app.schemas.eval import EvalRunRequest, EvalRunResponse
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


@router.post("/eval/run", response_model=EvalRunResponse)
async def run_eval(payload: EvalRunRequest, request: Request) -> EvalRunResponse:
    container = request.app.state.container
    dataset = load_jsonl_dataset(payload.dataset_path)
    report = container.ragas_runner.run(dataset)
    output_path = payload.output_path
    if output_path:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        save_json(str(target), report)
    return EvalRunResponse(status="ok", output_path=output_path, report=report)


@router.get("/debug/storage")
async def storage_debug(request: Request) -> dict[str, object]:
    container = request.app.state.container
    parsed_dir = container.config.parsed_dir
    parsed_files = sorted(path.name for path in parsed_dir.glob("*.json")) if parsed_dir.exists() else []
    return {
        "vectorstore": container.repository.stats(),
        "parsed_dir": str(parsed_dir),
        "parsed_files": parsed_files,
    }


@router.get("/debug/parsed/{doc_id}")
async def parsed_document_debug(doc_id: str, request: Request) -> dict:
    container = request.app.state.container
    target = container.config.parsed_dir / f"{doc_id}.json"
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"parsed artifact not found: {doc_id}")
    return json.loads(target.read_text(encoding="utf-8"))
