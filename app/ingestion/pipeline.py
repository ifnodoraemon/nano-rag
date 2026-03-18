from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from app.ingestion.chunker import build_chunks
from app.ingestion.loader import discover_files
from app.ingestion.normalizer import normalize_text
from app.ingestion.parser_docling import parse_document
from app.model_client.embeddings import EmbeddingClient
from app.schemas.document import Document, IngestResponse
from app.vectorstore.repository import VectorRepository

if TYPE_CHECKING:
    from app.core.config import AppConfig


class IngestionPipeline:
    def __init__(self, config: AppConfig, repository: VectorRepository, embedding_client: EmbeddingClient) -> None:
        self.config = config
        self.repository = repository
        self.embedding_client = embedding_client

    async def run(self, path: str) -> IngestResponse:
        files = discover_files(path)
        chunk_count = 0
        doc_count = 0

        for file_path in files:
            doc_id = str(uuid4())
            text = normalize_text(parse_document(file_path))
            document = Document(
                doc_id=doc_id,
                source_path=str(file_path),
                title=Path(file_path).stem,
                content=text,
                metadata={},
            )
            chunk_size = self.config.settings["chunk"]["size"]
            overlap = self.config.settings["chunk"]["overlap"]
            chunks = build_chunks(doc_id, str(file_path), document.title, text, chunk_size, overlap)
            embeddings = await self.embedding_client.embed_texts([chunk.text for chunk in chunks])
            self.repository.upsert(document, chunks, embeddings)
            doc_count += 1
            chunk_count += len(chunks)

        return IngestResponse(documents=doc_count, chunks=chunk_count)
