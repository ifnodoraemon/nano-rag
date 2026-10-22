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
    # Re-parse even when the committed artifact already matches the file byte for
    # byte. Needed to rebuild deliberately (e.g. the wiki page was deleted by
    # hand), never for normal ingest.
    force: bool = False


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    # Documents removed because their content matched an incoming one (a file
    # that moved or was re-uploaded under a different path). Reported so a
    # supersede is never silent.
    superseded_doc_ids: list[str] = Field(default_factory=list)
    # Files whose content and parse configuration already matched the committed
    # artifact, so nothing was parsed, written or re-indexed. Reported so the
    # skip is explicit: "0 documents" then means "nothing matched", not "nothing
    # done".
    unchanged_doc_ids: list[str] = Field(default_factory=list)
