from __future__ import annotations

import json
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
    from app.core.tracing import TracingManager


class IngestionPipeline:
    def __init__(
        self,
        config: AppConfig,
        repository: VectorRepository,
        embedding_client: EmbeddingClient,
        tracing_manager: TracingManager,
    ) -> None:
        self.config = config
        self.repository = repository
        self.embedding_client = embedding_client
        self.tracing_manager = tracing_manager

    async def run(self, path: str) -> IngestResponse:
        with self.tracing_manager.span("ingestion.run", {"ingestion.path": path}):
            files = discover_files(path)
            chunk_count = 0
            doc_count = 0
            self.config.parsed_dir.mkdir(parents=True, exist_ok=True)

            for file_path in files:
                with self.tracing_manager.span("ingestion.file", {"ingestion.file_path": str(file_path)}):
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
                    self._write_parsed_artifact(document, chunks)
                    embeddings = await self.embedding_client.embed_texts([chunk.text for chunk in chunks])
                    self.repository.upsert(document, chunks, embeddings)
                    doc_count += 1
                    chunk_count += len(chunks)

            return IngestResponse(documents=doc_count, chunks=chunk_count)

    def _write_parsed_artifact(self, document: Document, chunks) -> None:
        artifact = {
            "document": document.model_dump(),
            "chunks": [chunk.model_dump() for chunk in chunks],
        }
        target = self.config.parsed_dir / f"{document.doc_id}.json"
        target.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
