from typing import Literal

from pydantic import BaseModel, Field, field_validator


def normalize_optional_scope(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized or normalized.lower() in {"null", "none", "__shared__"}:
        return None
    return normalized


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8192)
    top_k: int | None = Field(default=None, ge=1, le=100)
    kb_id: str | None = None
    session_id: str | None = None
    sample_id: str | None = None
    metadata_filters: dict[str, object] | None = None

    @field_validator("session_id", mode="before")
    @classmethod
    def blank_scope_values_are_none(cls, value: object) -> str | None:
        return normalize_optional_scope(value)


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
