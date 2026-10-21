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
    # Documents removed because their content matched an incoming one (a file
    # that moved or was re-uploaded under a different path). Reported so a
    # supersede is never silent.
    superseded_doc_ids: list[str] = Field(default_factory=list)
