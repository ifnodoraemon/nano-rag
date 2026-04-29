from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.schemas.chat import Citation


class BusinessChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8192)
    kb_id: str = Field(default="default", max_length=256)
    tenant_id: str | None = Field(default=None, max_length=256)
    session_id: str | None = Field(default=None, max_length=256)
    top_k: int | None = Field(default=None, ge=1, le=100)
    metadata_filters: dict[str, object] | None = None


class BusinessChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    contexts: list[dict]
    trace_id: str | None = None
    kb_id: str = "default"
    tenant_id: str | None = None
    session_id: str | None = None


class BusinessIngestRequest(BaseModel):
    path: str
    kb_id: str = "default"
    tenant_id: str | None = None

    @field_validator("path")
    @classmethod
    def path_must_be_non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("path must not be empty")
        return v


class BusinessIngestResponse(BaseModel):
    status: str
    kb_id: str
    tenant_id: str | None = None
    documents: int
    chunks: int
    source: str = "path"
    uploaded_files: list[str] = Field(default_factory=list)


class BusinessDocumentSummary(BaseModel):
    doc_id: str
    title: str
    source_path: str
    kb_id: str = "default"
    tenant_id: str | None = None
    chunk_count: int = 0
    updated_at: float
    doc_type: str | None = None
    source_key: str | None = None


class WorkspaceSummary(BaseModel):
    workspace_id: str
    name: str
    kb_id: str
    tenant_id: str | None = None
    document_count: int = 0
    chunk_count: int = 0
    trace_count: int = 0
    updated_at: float | None = None


class IngestSourceSummary(BaseModel):
    path: str
    name: str
    extension: str
    size_bytes: int
    updated_at: float


class FeedbackRequest(BaseModel):
    trace_id: str = Field(..., max_length=256)
    rating: Literal["up", "down"]
    kb_id: str = Field(default="default", max_length=256)
    tenant_id: str | None = Field(default=None, max_length=256)
    session_id: str | None = Field(default=None, max_length=256)
    comment: str | None = Field(default=None, max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=20)


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: str


class FeedbackRecord(BaseModel):
    feedback_id: str
    trace_id: str
    rating: str
    kb_id: str = "default"
    tenant_id: str | None = None
    session_id: str | None = None
    comment: str | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: float
