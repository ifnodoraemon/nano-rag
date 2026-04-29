from typing import Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    source_path: str
    title: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    modality: Literal["text", "image", "audio", "video"] = "text"
    media_uri: str | None = None
    mime_type: str | None = None
