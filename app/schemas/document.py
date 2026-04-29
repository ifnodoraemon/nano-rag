from pydantic import BaseModel, Field


class Document(BaseModel):
    doc_id: str
    source_path: str
    title: str
    content: str
    metadata: dict = Field(default_factory=dict)


class IngestRequest(BaseModel):
    path: str
    kb_id: str | None = None


class IngestResponse(BaseModel):
    documents: int
    chunks: int
