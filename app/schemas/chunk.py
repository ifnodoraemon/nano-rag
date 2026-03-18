from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    source_path: str
    title: str | None = None
    metadata: dict = Field(default_factory=dict)
