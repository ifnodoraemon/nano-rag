from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mimetypes

from app.ingestion.chunker import build_chunks
from app.ingestion.loader import discover_files
from app.ingestion.metadata import extract_document_metadata
from app.ingestion.normalizer import normalize_text
from app.ingestion.parser_docling import parse_document
from app.ingestion.semantic_chunker import SemanticChunker
from app.core.exceptions import ModelGatewayError, ParsingError
from app.model_client.embeddings import EmbeddingClient
from app.model_client.document_parser import DocumentParserClient
from app.model_client.multimodal_embedding import (
    EmbedItem,
    ImageItem,
    TextItem,
)
from app.retrieval.hybrid_retriever import HybridRetriever
from app.schemas.chunk import Chunk
from app.schemas.document import Document, IngestResponse

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
IMAGE_MIME_FALLBACK = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}

logger = logging.getLogger(__name__)
from app.vectorstore.repository import VectorRepository
from app.wiki.compiler import WikiCompiler
from app.wiki.search import WikiSearcher

if TYPE_CHECKING:
    from app.core.config import AppConfig
    from app.core.tracing import TracingManager


class ChunkConfigurationError(RuntimeError):
    pass


@dataclass
class ParsedArtifactSnapshot:
    document: Document
    chunks: list[Chunk]


@dataclass
class PreparedDocument:
    source_path: str
    doc_id: str
    document: Document
    chunks: list[Chunk]
    embeddings: list[list[float]]


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
        self,
        path: str,
        kb_id: str = "default",
        tenant_id: str | None = None,
        source_path_overrides: dict[str, str] | None = None,
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
            prepared_documents: list[PreparedDocument] = []

            for file_path in files:
                with self.tracing_manager.span(
                    "ingestion.file", {"ingestion.file_path": str(file_path)}
                ):
                    prepared = await self._prepare_document(
                        file_path,
                        kb_id=kb_id,
                        tenant_id=tenant_id,
                        source_path_overrides=source_path_overrides,
                    )
                    prepared_documents.append(prepared)
                    doc_count += 1
                    chunk_count += len(prepared.chunks)

            self.config.parsed_dir.mkdir(parents=True, exist_ok=True)
            applied_snapshots: list[
                tuple[PreparedDocument, ParsedArtifactSnapshot | None]
            ] = []
            try:
                for prepared in prepared_documents:
                    applied_snapshots.append(
                        (
                            prepared,
                            self._load_parsed_artifact(
                                prepared.source_path, prepared.doc_id, kb_id, tenant_id
                            ),
                        )
                    )
                    self._delete_committed_state(
                        prepared.source_path, prepared.doc_id, kb_id, tenant_id
                    )
                    self.repository.upsert(
                        prepared.document,
                        prepared.chunks,
                        prepared.embeddings,
                    )
                    if self.hybrid_retriever:
                        self.hybrid_retriever.index_chunks(prepared.chunks)
                    self._write_parsed_artifact(prepared.document, prepared.chunks)
                    if self.wiki_compiler:
                        self.wiki_compiler.upsert_document(
                            prepared.document, prepared.chunks
                        )
                        wiki_updated = True
            except Exception:
                await self._rollback_applied_documents(
                    applied_snapshots, kb_id=kb_id, tenant_id=tenant_id
                )
                raise

            if wiki_updated and self.wiki_searcher:
                self.wiki_searcher.refresh()
            return IngestResponse(documents=doc_count, chunks=chunk_count)

    async def _prepare_document(
        self,
        file_path: Path,
        kb_id: str,
        tenant_id: str | None = None,
        source_path_overrides: dict[str, str] | None = None,
    ) -> PreparedDocument:
        source_path = self._resolve_source_path(file_path, source_path_overrides)
        doc_id = self._stable_doc_id(source_path, kb_id, tenant_id)
        if file_path.suffix.lower() in IMAGE_SUFFIXES:
            return await self._prepare_image_document(
                file_path=file_path,
                source_path=source_path,
                doc_id=doc_id,
                kb_id=kb_id,
                tenant_id=tenant_id,
            )
        text = normalize_text(await parse_document(file_path, self.document_parser))
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
        embed_inputs = self._chunks_to_embed_inputs(chunks, file_path=file_path)
        embeddings = await self._embed_items(embed_inputs)
        if len(embeddings) != len(chunks):
            raise ModelGatewayError(
                "embedding service returned an inconsistent number of vectors"
            )
        return PreparedDocument(
            source_path=source_path,
            doc_id=doc_id,
            document=document,
            chunks=chunks,
            embeddings=embeddings,
        )

    async def _embed_items(
        self, items: list[list[EmbedItem]]
    ) -> list[list[float]]:
        embed_items_fn = getattr(self.embedding_client, "embed_items", None)
        if embed_items_fn is not None:
            return await embed_items_fn(items)
        # Backward-compat fallback for legacy / test fake clients without
        # multimodal support: degrade to text-only by stringifying items.
        flat_texts: list[str] = []
        for batch in items:
            parts: list[str] = []
            for item in batch:
                if isinstance(item, TextItem):
                    parts.append(item.text)
                elif isinstance(item, ImageItem):
                    parts.append(f"<image:{item.mime_type}:{len(item.data)} bytes>")
                else:
                    parts.append(str(item))
            flat_texts.append("\n".join(parts))
        return await self.embedding_client.embed_texts(flat_texts)

    async def _prepare_image_document(
        self,
        file_path: Path,
        source_path: str,
        doc_id: str,
        kb_id: str,
        tenant_id: str | None,
    ) -> PreparedDocument:
        suffix = file_path.suffix.lower()
        mime_type = IMAGE_MIME_FALLBACK.get(suffix) or (
            mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        )
        title = file_path.stem
        document_metadata = {
            "kb_id": kb_id,
            "tenant_id": tenant_id,
            "doc_type": "image",
            "modality": "image",
            "mime_type": mime_type,
            "source_key": title.lower(),
            "headings": [],
            "section_count": 0,
        }
        document = Document(
            doc_id=doc_id,
            source_path=source_path,
            title=title,
            content="",
            metadata=document_metadata,
        )
        chunk = Chunk(
            chunk_id=f"{doc_id}:0",
            doc_id=doc_id,
            chunk_index=0,
            text="",
            source_path=source_path,
            title=title,
            metadata={
                "kb_id": kb_id,
                "tenant_id": tenant_id,
                "modality": "image",
                "mime_type": mime_type,
                "media_uri": source_path,
            },
            modality="image",
            media_uri=source_path,
            mime_type=mime_type,
        )
        try:
            image_bytes = file_path.read_bytes()
        except OSError as exc:
            raise ParsingError(
                f"failed to read image bytes for {source_path}: {exc}"
            ) from exc
        embeddings = await self._embed_items(
            [[ImageItem(data=image_bytes, mime_type=mime_type)]]
        )
        if len(embeddings) != 1:
            raise ModelGatewayError(
                "embedding service returned an inconsistent number of vectors"
            )
        return PreparedDocument(
            source_path=source_path,
            doc_id=doc_id,
            document=document,
            chunks=[chunk],
            embeddings=embeddings,
        )

    def _chunks_to_embed_inputs(
        self, chunks: list[Chunk], file_path: Path | None = None
    ) -> list[list[EmbedItem]]:
        inputs: list[list[EmbedItem]] = []
        for chunk in chunks:
            if chunk.modality == "image":
                source = (
                    file_path
                    if file_path is not None and file_path.suffix.lower() in IMAGE_SUFFIXES
                    else None
                )
                if source is None and chunk.media_uri:
                    candidate = Path(chunk.media_uri)
                    source = candidate if candidate.exists() else None
                if source is None:
                    raise ModelGatewayError(
                        f"cannot re-embed image chunk {chunk.chunk_id}: bytes not "
                        "available; the upload directory may have been pruned. "
                        "Re-ingest the original file."
                    )
                mime = chunk.mime_type or IMAGE_MIME_FALLBACK.get(
                    source.suffix.lower(), "application/octet-stream"
                )
                inputs.append(
                    [ImageItem(data=source.read_bytes(), mime_type=mime)]
                )
            else:
                inputs.append([TextItem(chunk.text)])
        return inputs

    async def _rollback_applied_documents(
        self,
        applied_snapshots: list[tuple[PreparedDocument, ParsedArtifactSnapshot | None]],
        kb_id: str,
        tenant_id: str | None = None,
    ) -> None:
        wiki_needs_refresh = False
        for prepared, snapshot in reversed(applied_snapshots):
            try:
                if self.wiki_compiler:
                    wiki_needs_refresh = True
                self._delete_committed_state(
                    prepared.source_path, prepared.doc_id, kb_id, tenant_id
                )
                if snapshot is None:
                    continue
                embeddings = await self._embed_items(
                    self._chunks_to_embed_inputs(snapshot.chunks)
                )
                if len(embeddings) != len(snapshot.chunks):
                    raise ModelGatewayError(
                        "embedding service returned an inconsistent number of vectors"
                    )
                self.repository.upsert(
                    snapshot.document,
                    snapshot.chunks,
                    embeddings,
                )
                if self.hybrid_retriever:
                    self.hybrid_retriever.index_chunks(snapshot.chunks)
                self._write_parsed_artifact(snapshot.document, snapshot.chunks)
                if self.wiki_compiler:
                    self.wiki_compiler.upsert_document(
                        snapshot.document, snapshot.chunks
                    )
                    wiki_needs_refresh = True
            except Exception:
                logger.warning(
                    "rollback failed for document %s — manual repair may be needed",
                    prepared.doc_id,
                    exc_info=True,
                )
                continue
        if wiki_needs_refresh and self.wiki_searcher:
            self.wiki_searcher.refresh()

    def _resolve_source_path(
        self, file_path: Path, source_path_overrides: dict[str, str] | None = None
    ) -> str:
        resolved = file_path.resolve()
        if source_path_overrides:
            override = source_path_overrides.get(str(resolved))
            if override:
                return override
        return self._normalize_source_path(resolved)

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
        remove_active_doc: bool = False,
    ) -> None:
        if not self.config.parsed_dir.exists():
            return
        for artifact in self.config.parsed_dir.glob("*.json"):
            if artifact.stem == active_doc_id and not remove_active_doc:
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

    def _delete_committed_state(
        self,
        source_path: str,
        doc_id: str,
        kb_id: str,
        tenant_id: str | None = None,
    ) -> None:
        if self.hybrid_retriever:
            self.hybrid_retriever.remove_by_source(
                source_path, kb_id=kb_id, tenant_id=tenant_id
            )
        self.repository.delete_by_source(source_path, kb_id=kb_id, tenant_id=tenant_id)
        self._cleanup_parsed_artifacts(
            source_path,
            doc_id,
            kb_id,
            tenant_id,
            remove_active_doc=True,
        )
        if self.wiki_compiler:
            self.wiki_compiler.remove_document(doc_id)

    def _load_parsed_artifact(
        self,
        source_path: str,
        doc_id: str,
        kb_id: str,
        tenant_id: str | None = None,
    ) -> ParsedArtifactSnapshot | None:
        if not self.config.parsed_dir.exists():
            return None
        primary_artifact = self.config.parsed_dir / f"{doc_id}.json"
        if primary_artifact.exists():
            snapshot = self._read_parsed_artifact(primary_artifact)
            if snapshot and self._snapshot_matches_scope(
                snapshot, source_path, kb_id, tenant_id
            ):
                return snapshot
        for artifact in sorted(self.config.parsed_dir.glob("*.json")):
            snapshot = self._read_parsed_artifact(artifact)
            if snapshot and self._snapshot_matches_scope(
                snapshot, source_path, kb_id, tenant_id
            ):
                return snapshot
        return None

    def _snapshot_matches_scope(
        self,
        snapshot: ParsedArtifactSnapshot,
        source_path: str,
        kb_id: str,
        tenant_id: str | None = None,
    ) -> bool:
        return (
            snapshot.document.source_path == source_path
            and snapshot.document.metadata.get("kb_id", "default") == kb_id
            and snapshot.document.metadata.get("tenant_id") == tenant_id
        )

    def _read_parsed_artifact(self, artifact: Path) -> ParsedArtifactSnapshot | None:
        try:
            payload = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        document_payload = payload.get("document") if isinstance(payload, dict) else None
        chunks_payload = payload.get("chunks") if isinstance(payload, dict) else None
        if not isinstance(document_payload, dict) or not isinstance(chunks_payload, list):
            return None
        try:
            return ParsedArtifactSnapshot(
                document=Document.model_validate(document_payload),
                chunks=[Chunk.model_validate(chunk) for chunk in chunks_payload],
            )
        except Exception:
            return None

    def _write_parsed_artifact(self, document: Document, chunks) -> None:
        artifact = {
            "document": document.model_dump(),
            "chunks": [chunk.model_dump() for chunk in chunks],
        }
        target = self.config.parsed_dir / f"{document.doc_id}.json"
        target.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
        )
