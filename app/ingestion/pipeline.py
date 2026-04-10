from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from app.ingestion.chunker import build_chunks
from app.ingestion.loader import IngestPathError, discover_files
from app.ingestion.metadata import extract_document_metadata
from app.ingestion.normalizer import normalize_text
from app.ingestion.parser_docling import parse_document
from app.ingestion.semantic_chunker import SemanticChunker
from app.core.exceptions import ParsingError
from app.model_client.embeddings import EmbeddingClient
from app.model_client.document_parser import DocumentParserClient
from app.retrieval.hybrid_retriever import HybridRetriever
from app.schemas.document import Document, IngestResponse
from app.vectorstore.repository import VectorRepository
from app.wiki.compiler import WikiCompiler
from app.wiki.search import WikiSearcher

if TYPE_CHECKING:
    from app.core.config import AppConfig
    from app.core.tracing import TracingManager


class ChunkConfigurationError(RuntimeError):
    pass


class IngestionPipeline:
    def __init__(
        self,
        config: AppConfig,
        repository: VectorRepository,
        embedding_client: EmbeddingClient,
        tracing_manager: TracingManager,
        semantic_chunker: SemanticChunker | None = None,
        document_parser: DocumentParserClient | None = None,
        hybrid_retriever: HybridRetriever | None = None,
        wiki_compiler: WikiCompiler | None = None,
        wiki_searcher: WikiSearcher | None = None,
    ) -> None:
        self.config = config
        self.repository = repository
        self.embedding_client = embedding_client
        self.tracing_manager = tracing_manager
        self.semantic_chunker = semantic_chunker
        self.document_parser = document_parser
        self.hybrid_retriever = hybrid_retriever
        self.wiki_compiler = wiki_compiler
        self.wiki_searcher = wiki_searcher
        self._use_semantic_chunker = semantic_chunker is not None and os.getenv(
            "RAG_SEMANTIC_CHUNKER_ENABLED", "false"
        ).lower() in ("true", "1", "yes")

    async def run(
        self, path: str, kb_id: str = "default", tenant_id: str | None = None
    ) -> IngestResponse:
        with self.tracing_manager.span(
            "ingestion.run",
            {
                "ingestion.path": path,
                "ingestion.kb_id": kb_id,
                "ingestion.tenant_id": tenant_id or "",
            },
        ):
            files = discover_files(path)
            chunk_count = 0
            doc_count = 0
            wiki_updated = False
            self.config.parsed_dir.mkdir(parents=True, exist_ok=True)

            for file_path in files:
                with self.tracing_manager.span(
                    "ingestion.file", {"ingestion.file_path": str(file_path)}
                ):
                    source_path = self._normalize_source_path(file_path)
                    doc_id = self._stable_doc_id(source_path, kb_id, tenant_id)
                    text = normalize_text(
                        await parse_document(file_path, self.document_parser)
                    )
                    if not text:
                        raise ParsingError(
                            f"Document parsing returned empty content for {source_path}. "
                            "If this is a scanned or image-heavy file, enable a multimodal document parser model."
                        )
                    document_metadata = extract_document_metadata(
                        source_path=source_path,
                        title=Path(file_path).stem,
                        text=text,
                        kb_id=kb_id,
                        tenant_id=tenant_id,
                    )
                    document = Document(
                        doc_id=doc_id,
                        source_path=source_path,
                        title=Path(file_path).stem,
                        content=text,
                        metadata=document_metadata,
                    )
                    chunk_size = self.config.settings["chunk"]["size"]
                    overlap = self.config.settings["chunk"]["overlap"]
                    try:
                        if self._use_semantic_chunker and self.semantic_chunker:
                            chunks = self.semantic_chunker.chunk(
                                text,
                                doc_id,
                                source_path,
                                document.title,
                                metadata=document.metadata,
                            )
                        else:
                            chunks = build_chunks(
                                doc_id,
                                source_path,
                                document.title,
                                text,
                                chunk_size,
                                overlap,
                                document.metadata,
                            )
                    except ValueError as exc:
                        raise ChunkConfigurationError(
                            f"Invalid chunk configuration for {source_path}: {exc}. "
                            f"chunk_size={chunk_size}, overlap={overlap}. "
                            "Ensure chunk_size > overlap in configs/settings.yaml"
                        ) from exc
                    if not chunks:
                        raise ParsingError(
                            f"Document parsing produced no chunks for {source_path}. "
                            "The extracted content may be empty or structurally invalid."
                        )
                    chunks = [
                        chunk.model_copy(
                            update={
                                "metadata": {
                                    **chunk.metadata,
                                    "kb_id": kb_id,
                                    "tenant_id": tenant_id,
                                }
                            }
                        )
                        for chunk in chunks
                    ]
                    if self.hybrid_retriever:
                        self.hybrid_retriever.remove_by_source(
                            source_path, kb_id=kb_id, tenant_id=tenant_id
                        )
                    self.repository.delete_by_source(
                        source_path, kb_id=kb_id, tenant_id=tenant_id
                    )
                    self._cleanup_parsed_artifacts(
                        source_path, doc_id, kb_id, tenant_id
                    )
                    self._write_parsed_artifact(document, chunks)
                    if chunks:
                        embeddings = await self.embedding_client.embed_texts(
                            [chunk.text for chunk in chunks]
                        )
                        await asyncio.to_thread(
                            self.repository.upsert, document, chunks, embeddings
                        )
                        if self.hybrid_retriever:
                            self.hybrid_retriever.index_chunks(chunks)
                    if self.wiki_compiler:
                        self.wiki_compiler.upsert_document(document, chunks)
                        wiki_updated = True
                    doc_count += 1
                    chunk_count += len(chunks)

            if wiki_updated and self.wiki_searcher:
                self.wiki_searcher.refresh()
            return IngestResponse(documents=doc_count, chunks=chunk_count)

    def _normalize_source_path(self, file_path: Path) -> str:
        resolved = file_path.resolve()
        project_root = self.config.config_dir.parent.resolve()
        try:
            return str(resolved.relative_to(project_root))
        except ValueError:
            return str(resolved)

    def _stable_doc_id(
        self, source_path: str, kb_id: str, tenant_id: str | None = None
    ) -> str:
        identity = "|".join([kb_id, tenant_id or "", source_path])
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return f"doc-{digest[:24]}"

    def _cleanup_parsed_artifacts(
        self,
        source_path: str,
        active_doc_id: str,
        kb_id: str,
        tenant_id: str | None = None,
    ) -> None:
        if not self.config.parsed_dir.exists():
            return
        for artifact in self.config.parsed_dir.glob("*.json"):
            if artifact.stem == active_doc_id:
                continue
            try:
                payload = json.loads(artifact.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            document = payload.get("document") if isinstance(payload, dict) else None
            metadata = (
                document.get("metadata", {}) if isinstance(document, dict) else {}
            )
            if (
                isinstance(document, dict)
                and document.get("source_path") == source_path
                and metadata.get("kb_id", "default") == kb_id
                and metadata.get("tenant_id") == tenant_id
            ):
                artifact.unlink(missing_ok=True)

    def _write_parsed_artifact(self, document: Document, chunks) -> None:
        artifact = {
            "document": document.model_dump(),
            "chunks": [chunk.model_dump() for chunk in chunks],
        }
        target = self.config.parsed_dir / f"{document.doc_id}.json"
        target.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
        )
