from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8192)
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
    claim_type: Literal["factual", "conditional", "conflict", "insufficiency"] = "factual"
    text: str
    citation_labels: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    contexts: list[dict]
    supporting_claims: list[SupportingClaim] = []
    trace_id: str | None = None
