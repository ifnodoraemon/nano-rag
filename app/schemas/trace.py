from pydantic import BaseModel


class TraceSummary(BaseModel):
    trace_id: str
    latency_seconds: float | None = None
    query: str | None = None
    model_alias: str | None = None
    prompt_version: str | None = None


class TraceRecord(BaseModel):
    trace_id: str
    latency_seconds: float | None = None
    query: str | None = None
    retrieved_chunk_ids: list[str] = []
    reranked_chunk_ids: list[str] = []
    retrieved: list[dict] = []
    reranked: list[dict] = []
    contexts: list[dict] = []
    citations: list[dict] = []
    answer: str | None = None
    model_alias: str | None = None
    prompt_version: str | None = None


class RetrievalDebugResponse(BaseModel):
    query: str
    retrieved: list[dict]
    reranked: list[dict]
    contexts: list[dict]
    trace_id: str | None = None
