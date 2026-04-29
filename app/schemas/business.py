from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.schemas.chat import Citation
from app.schemas.chat import normalize_optional_scope


class BusinessChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8192)
    kb_id: str = Field(default="default", max_length=256)
    session_id: str | None = Field(default=None, max_length=256)
    top_k: int | None = Field(default=None, ge=1, le=100)
    metadata_filters: dict[str, object] | None = None

    @field_validator("session_id", mode="before")
    @classmethod
    def blank_scope_values_are_none(cls, value: object) -> str | None:
        return normalize_optional_scope(value)


class BusinessChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    contexts: list[dict]
    trace_id: str | None = None
    kb_id: str = "default"
    session_id: str | None = None


class BusinessRetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8192)
    kb_id: str = Field(default="default", max_length=256)
    session_id: str | None = Field(default=None, max_length=256)
    top_k: int | None = Field(default=None, ge=1, le=100)
    metadata_filters: dict[str, object] | None = None

    @field_validator("session_id", mode="before")
    @classmethod
    def blank_scope_values_are_none(cls, value: object) -> str | None:
        return normalize_optional_scope(value)


class BusinessRetrieveResponse(BaseModel):
    query: str
    contexts: list[dict]
    retrieved: list[dict]
    reranked: list[dict]
    trace_id: str | None = None
    kb_id: str = "default"
    session_id: str | None = None


class BusinessIngestRequest(BaseModel):
    path: str
    kb_id: str = "default"

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
    documents: int
    chunks: int
    source: str = "path"
    uploaded_files: list[str] = Field(default_factory=list)


class BusinessDocumentSummary(BaseModel):
    doc_id: str
    title: str
    source_path: str
    kb_id: str = "default"
    chunk_count: int = 0
    updated_at: float
    doc_type: str | None = None
    source_key: str | None = None


class KnowledgeBaseSummary(BaseModel):
    kb_id: str
    name: str
    description: str | None = None
    source: str = "local"
    external_ref: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at: float
    updated_at: float
    document_count: int = 0
    chunk_count: int = 0
    trace_count: int = 0
    last_activity_at: float | None = None


class KnowledgeBaseCreateRequest(BaseModel):
    kb_id: str
    name: str
    description: str | None = None
    source: str = "local"
    external_ref: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


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
    session_id: str | None = Field(default=None, max_length=256)
    comment: str | None = Field(default=None, max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("session_id", mode="before")
    @classmethod
    def blank_scope_values_are_none(cls, value: object) -> str | None:
        return normalize_optional_scope(value)


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: str


class FeedbackRecord(BaseModel):
    feedback_id: str
    trace_id: str
    rating: str
    kb_id: str = "default"
    session_id: str | None = None
    comment: str | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: float
