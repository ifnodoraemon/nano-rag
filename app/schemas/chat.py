from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    top_k: int | None = None
    kb_id: str | None = None
    tenant_id: str | None = None
    session_id: str | None = None
    sample_id: str | None = None


class Citation(BaseModel):
    chunk_id: str
    source: str
    score: float | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    contexts: list[dict]
    trace_id: str | None = None
