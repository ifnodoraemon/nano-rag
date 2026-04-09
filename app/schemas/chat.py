from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    top_k: int | None = None
    kb_id: str | None = None
    tenant_id: str | None = None
    session_id: str | None = None
    sample_id: str | None = None
    metadata_filters: dict[str, object] | None = None


class Citation(BaseModel):
    citation_label: str | None = None
    chunk_id: str
    source: str
    score: float | None = None
    evidence_role: str | None = None
    wiki_status: str | None = None
    span_text: str | None = None
    span_start: int | None = None
    span_end: int | None = None


class SupportingClaim(BaseModel):
    claim_type: str = "factual"
    text: str
    citation_labels: list[str] = []


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    contexts: list[dict]
    supporting_claims: list[SupportingClaim] = []
    trace_id: str | None = None
