from pydantic import BaseModel, Field

from app.schemas.chat import Citation


class BusinessChatRequest(BaseModel):
    query: str
    kb_id: str = "default"
    tenant_id: str | None = None
    session_id: str | None = None
    top_k: int | None = None
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


class BusinessIngestResponse(BaseModel):
    status: str
    kb_id: str
    tenant_id: str | None = None
    documents: int
    chunks: int
    source: str = "path"
    uploaded_files: list[str] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    trace_id: str
    rating: str
    kb_id: str = "default"
    tenant_id: str | None = None
    session_id: str | None = None
    comment: str | None = None
    tags: list[str] = Field(default_factory=list)


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
