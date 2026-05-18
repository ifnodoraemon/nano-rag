from __future__ import annotations

import hashlib
import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from app.ingestion.loader import discover_files
from app.ingestion.metadata import extract_document_metadata
from app.ingestion.graph_extractor import GraphExtractor
from app.vectorstore.repository import VectorRepository
from app.wiki.compiler import WikiCompiler
from app.wiki.search import WikiSearcher
from app.ingestion.structured_parser import StructuredDocumentParser
from app.core.exceptions import ModelGatewayError, ParsingError
from app.model_client.embeddings import EmbeddingClient
from app.model_client.document_parser import DocumentParserClient
from app.model_client.generation import GenerationClient
from app.model_client.multimodal_embedding import (
    AudioItem,
    EmbedItem,
    FileItem,
    ImageItem,
    TextItem,
    VideoItem,
)
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.multivector import (
    attach_chunk_multivectors,
    multivector_provider_from_config,
    multivector_store_from_config,
)
from app.retrieval.graph_store import GraphStore
from app.schemas.chunk import Chunk
from app.schemas.document import Document, IngestResponse
from app.schemas.structured import KnowledgeGraph, NodeType, StructuredDocument

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".mpeg", ".mpg"}
MEDIA_SUFFIXES = IMAGE_SUFFIXES | AUDIO_SUFFIXES | VIDEO_SUFFIXES
DOCUMENT_ATTACHMENT_SUFFIXES = {".pdf", ".docx", ".pptx", ".xlsx"}
OOXML_MEDIA_PREFIXES = {
    ".docx": "word/media/",
    ".pptx": "ppt/media/",
    ".xlsx": "xl/media/",
}
logger = logging.getLogger(__name__)

MEDIA_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpeg",
}
DOCUMENT_ATTACHMENT_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}
DEFAULT_TEXT_CHUNK_MAX_CHARS = 6000
DEFAULT_TEXT_CHUNK_OVERLAP_CHARS = 300
DEFAULT_CODE_CHUNK_MAX_CHARS = 3000
DEFAULT_CODE_CHUNK_OVERLAP_CHARS = 400
DEFAULT_LOG_CHUNK_MAX_CHARS = 4000
DEFAULT_LOG_CHUNK_OVERLAP_CHARS = 200


def _modality_for_suffix(suffix: str) -> str:
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in AUDIO_SUFFIXES:
        return "audio"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    return "text"


def _mime_type_for_suffix(suffix: str) -> str:
    mime_type = MEDIA_MIME_TYPES.get(suffix)
    if not mime_type:
        raise ParsingError(f"mime type is not configured for media suffix {suffix}")
    return mime_type


if TYPE_CHECKING:
    from app.core.config import AppConfig
    from app.core.tracing import TracingManager


@dataclass
class PreparedDocument:
    source_path: str
    doc_id: str
    document: Document
    chunks: list[Chunk]
    embeddings: list[list[float]]
    structured_document: StructuredDocument | None = None


class IngestionPipeline:
    def __init__(
        self,
        config: AppConfig,
        repository: VectorRepository,
        embedding_client: EmbeddingClient,
        generation_client: GenerationClient,
        tracing_manager: TracingManager,
        document_parser: DocumentParserClient | None = None,
        hybrid_retriever: HybridRetriever | None = None,
        graph_store: GraphStore | None = None,
        wiki_compiler: WikiCompiler | None = None,
        wiki_searcher: WikiSearcher | None = None,
    ) -> None:
        self.config = config
        self.repository = repository
        self.embedding_client = embedding_client
        self.generation_client = generation_client
        self.tracing_manager = tracing_manager
        self.document_parser = document_parser
        self.hybrid_retriever = hybrid_retriever
        self.graph_store = graph_store
        self.wiki_compiler = wiki_compiler
        self.wiki_searcher = wiki_searcher
        self.structured_parser = StructuredDocumentParser(document_parser)
        self.graph_extractor = GraphExtractor(generation_client)
        self.multivector_provider = multivector_provider_from_config(config)
        self.multivector_store = multivector_store_from_config(config)

    async def run(
        self,
        path: str,
        kb_id: str = "default",
        source_path_overrides: dict[str, str] | None = None,
    ) -> IngestResponse:
        with self.tracing_manager.span(
            "ingestion.run",
            {
                "ingestion.path": path,
                "ingestion.kb_id": kb_id,
            },
        ):
            files = discover_files(path)
            chunk_count = 0
            doc_count = 0
            wiki_updated = False
            prepared_documents: list[PreparedDocument] = []

            try:
                for file_path in files:
                    with self.tracing_manager.span(
                        "ingestion.file", {"ingestion.file_path": str(file_path)}
                    ):
                        prepared = await self._prepare_document(
                            file_path,
                            kb_id=kb_id,
                            source_path_overrides=source_path_overrides,
                        )
                        prepared_documents.append(prepared)
                        doc_count += 1
                        chunk_count += len(prepared.chunks)
            except Exception:
                self._delete_prepared_multivector_refs(prepared_documents)
                raise

            self.config.parsed_dir.mkdir(parents=True, exist_ok=True)
            for prepared in prepared_documents:
                artifact_tmp = self._stage_parsed_artifact(
                    prepared.document,
                    prepared.chunks,
                    prepared.structured_document,
                )
                old_chunk_ids = self._committed_chunk_ids(
                    prepared.source_path,
                    kb_id=kb_id,
                    active_doc_id=prepared.doc_id,
                )
                old_multivector_refs = self._committed_multivector_refs(
                    prepared.source_path,
                    kb_id=kb_id,
                    active_doc_id=prepared.doc_id,
                )
                new_chunk_ids = {chunk.chunk_id for chunk in prepared.chunks}
                new_multivector_refs = _chunk_multivector_refs(prepared.chunks)
                upsert_started = False
                try:
                    self.repository.upsert(
                        prepared.document,
                        prepared.chunks,
                        prepared.embeddings,
                    )
                    upsert_started = True
                    self._commit_parsed_artifact(artifact_tmp, prepared.document.doc_id)
                    try:
                        self._delete_stale_committed_state(
                            prepared.source_path,
                            kb_id=kb_id,
                            doc_id=prepared.doc_id,
                            keep_chunk_ids=new_chunk_ids,
                        )
                    except Exception as exc:
                        logger.warning(
                            "stale vector cleanup failed for %s; new index remains committed: %s",
                            prepared.source_path,
                            exc,
                        )
                    self.multivector_store.delete_refs(
                        old_multivector_refs - new_multivector_refs
                    )
                    if self.hybrid_retriever:
                        try:
                            self.hybrid_retriever.remove_by_source(
                                prepared.source_path,
                                kb_id=kb_id,
                                keep_chunk_ids=new_chunk_ids,
                            )
                            self.hybrid_retriever.index_chunks(prepared.chunks)
                        except Exception as exc:
                            logger.warning(
                                "hybrid index refresh failed for %s; vector index remains committed: %s",
                                prepared.source_path,
                                exc,
                            )
                    if self.graph_store and prepared.structured_document:
                        try:
                            await asyncio.to_thread(
                                self.graph_store.upsert_document,
                                prepared.structured_document,
                            )
                        except Exception as exc:
                            logger.warning(
                                "graph store update failed for %s; vector index remains committed: %s",
                                prepared.source_path,
                                exc,
                            )
                    if self.wiki_compiler:
                        try:
                            self.wiki_compiler.upsert_document(prepared.document, prepared.chunks)
                            wiki_updated = True
                        except Exception as exc:
                            logger.warning(
                                "wiki compiler update failed for %s; vector index remains committed: %s",
                                prepared.source_path,
                                exc,
                            )
                except Exception:
                    artifact_tmp.unlink(missing_ok=True)
                    self.multivector_store.delete_refs(
                        new_multivector_refs - old_multivector_refs
                    )
                    if upsert_started:
                        self._rollback_uncommitted_vector_state(
                            prepared.source_path,
                            kb_id=kb_id,
                            keep_chunk_ids=old_chunk_ids,
                        )
                    raise

            if wiki_updated and self.wiki_searcher:
                self.wiki_searcher.refresh()
            return IngestResponse(documents=doc_count, chunks=chunk_count)

    async def _prepare_document(
        self,
        file_path: Path,
        kb_id: str,
        source_path_overrides: dict[str, str] | None = None,
    ) -> PreparedDocument:
        source_path = self._resolve_source_path(file_path, source_path_overrides)
        doc_id = self._stable_doc_id(source_path, kb_id)
        suffix = file_path.suffix.lower()
        if suffix in MEDIA_SUFFIXES:
            return await self._prepare_media_document(
                file_path=file_path,
                source_path=source_path,
                doc_id=doc_id,
                kb_id=kb_id,
            )
        structured_document = await self.structured_parser.parse(
            file_path,
            doc_id=doc_id,
            kb_id=kb_id,
            source_path=source_path,
        )
        try:
            structured_document.graph = await self.graph_extractor.extract(structured_document)
            structured_document.metadata["graph_extraction"] = {"status": "ok"}
        except Exception as exc:
            structured_document.graph = KnowledgeGraph()
            structured_document.metadata["graph_extraction"] = {
                "status": "failed",
                "error_type": exc.__class__.__name__,
                "error": str(exc)[:500],
            }
        text = self._structured_document_text(structured_document)
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
        )
        document_metadata.update(
            self._source_version_metadata(
                file_path,
                structured_document=structured_document,
                content=text,
            )
        )
        structured_document.metadata.update(
            {
                "source_content_hash": document_metadata["source_content_hash"],
                "source_size_bytes": document_metadata["source_size_bytes"],
                "parser": structured_document.metadata.get("parser"),
            }
        )
        document = Document(
            doc_id=doc_id,
            source_path=source_path,
            title=Path(file_path).stem,
            content=text,
            metadata=document_metadata,
        )
        chunks = self._structured_document_to_chunks(
            structured_document,
            source_path=source_path,
            title=document.title,
            metadata=document.metadata,
        )
        if not chunks:
            raise ParsingError(
                f"Document parsing produced no indexable structured nodes for {source_path}. "
                "The extracted content may be empty or structurally invalid."
            )
        chunks = [
            chunk.model_copy(
                update={
                    "metadata": {
                        **chunk.metadata,
                        "kb_id": kb_id,
                    }
                }
            )
            for chunk in chunks
        ]
        chunks.extend(
            self._document_attachment_chunks(
                file_path=file_path,
                source_path=source_path,
                doc_id=doc_id,
                title=document.title,
                metadata=document.metadata,
                start_index=len(chunks),
            )
        )
        embed_inputs = self._chunks_to_embed_inputs(chunks, file_path=file_path)
        embeddings = await self._embed_items(embed_inputs)
        if len(embeddings) != len(chunks):
            raise ModelGatewayError(
                "embedding service returned an inconsistent number of vectors"
            )
        chunks = await self._attach_multivectors(chunks)
        return PreparedDocument(
            source_path=source_path,
            doc_id=doc_id,
            document=document,
            chunks=chunks,
            embeddings=embeddings,
            structured_document=structured_document,
        )

    def _source_version_metadata(
        self,
        file_path: Path,
        *,
        structured_document: StructuredDocument,
        content: str,
    ) -> dict[str, object]:
        try:
            stat = file_path.stat()
            size_bytes = stat.st_size
            modified_at = stat.st_mtime
        except OSError:
            size_bytes = len(content.encode("utf-8"))
            modified_at = None
        source_bytes_hash = _file_sha256(file_path)
        return {
            "source_file_name": file_path.name,
            "source_suffix": file_path.suffix.lower(),
            "source_size_bytes": size_bytes,
            "source_modified_at": modified_at,
            "source_content_hash": source_bytes_hash
            or hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "parser": structured_document.metadata.get("parser"),
            "index_schema_version": "2026-05-rag-v1",
        }

    def _structured_document_text(self, structured_document: StructuredDocument) -> str:
        parts: list[str] = []
        for node in structured_document.iter_nodes():
            if node.node_type == NodeType.ROOT:
                continue
            if node.title and node.node_type == NodeType.SECTION:
                parts.append(f"# {node.title}")
            elif node.text.strip():
                parts.append(node.text.strip())
        return "\n\n".join(parts).strip()

    def _structured_document_to_chunks(
        self,
        structured_document: StructuredDocument,
        *,
        source_path: str,
        title: str,
        metadata: dict,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for index, node in enumerate(structured_document.iter_leaves()):
            if node.node_type == NodeType.SECTION:
                continue
            text = (node.table.narrative if node.table and node.table.narrative else node.text).strip()
            if not text and node.node_type != NodeType.IMAGE:
                continue
            provenance = node.provenance.model_dump()
            table_payload = node.table.model_dump() if node.table else None
            chunk_metadata = {
                **metadata,
                "kb_id": structured_document.kb_id,
                "node_id": node.node_id,
                "node_type": node.node_type.value,
                "clause_id": node.metadata.get("clause_id"),
                "clause_title": node.metadata.get("clause_title"),
                "clause_type": node.metadata.get("clause_type"),
                "definition_term": node.metadata.get("definition_term"),
                "hierarchy_path": node.provenance.hierarchy_path,
                "page_number": node.provenance.page_number,
                "bounding_box": (
                    node.provenance.bounding_box.model_dump()
                    if node.provenance.bounding_box
                    else None
                ),
                "source_ref": node.provenance.source_ref,
                "content_hash": node.metadata.get("content_hash"),
                "parser_confidence": node.metadata.get("parser_confidence"),
                "provenance": provenance,
            }
            chunk_strategy = _chunk_strategy_for_node(node.node_type.value, chunk_metadata)
            chunk_metadata["chunk_strategy"] = chunk_strategy
            chunk_metadata.update(
                _discourse_metadata(
                    text,
                    node_type=node.node_type.value,
                    has_table=bool(node.table),
                )
            )
            if table_payload:
                chunk_metadata["table"] = table_payload
                chunk_metadata["chunk_kind"] = "table_summary"
                chunk_metadata["chunk_strategy"] = "table_summary"
                chunk_strategy = "table_summary"
            text_parts = _split_text_for_index(
                text,
                max_chars=_text_chunk_max_chars(chunk_strategy),
                overlap_chars=_text_chunk_overlap_chars(chunk_strategy),
            )
            for part_index, part_text in enumerate(text_parts):
                chunk_id = (
                    node.node_id
                    if len(text_parts) == 1
                    else f"{node.node_id}:part:{part_index}"
                )
                part_metadata = {
                    **chunk_metadata,
                    "source_node_id": node.node_id,
                    "text_part_index": part_index,
                    "text_part_count": len(text_parts),
                }
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=structured_document.doc_id,
                        chunk_index=len(chunks),
                        text=part_text,
                        source_path=source_path,
                        title=" / ".join(node.provenance.hierarchy_path) or title,
                        metadata=part_metadata,
                        modality="text",
                    )
                )
            if node.table:
                chunks.extend(
                    self._table_row_chunks(
                        node,
                        table=node.table,
                        source_path=source_path,
                        title=title,
                        metadata=chunk_metadata,
                        start_index=len(chunks),
                    )
                )
        return chunks

    def _table_row_chunks(
        self,
        node,
        *,
        table,
        source_path: str,
        title: str,
        metadata: dict,
        start_index: int,
    ) -> list[Chunk]:
        rows: dict[int, dict[int, str]] = {}
        header_rows: set[int] = set()
        for cell in table.cells:
            rows.setdefault(cell.row, {})[cell.col] = cell.text
            if cell.is_header:
                header_rows.add(cell.row)
        if not rows:
            return []
        header_index = min(header_rows) if header_rows else min(rows)
        headers = rows.get(header_index, {})
        row_chunks: list[Chunk] = []
        body_row_indices = [index for index in sorted(rows) if index not in header_rows]
        for offset, row_index in enumerate(body_row_indices):
            row = rows[row_index]
            if not any(str(value).strip() for value in row.values()):
                continue
            row_pairs = []
            for col in sorted(set(headers) | set(row)):
                header = str(headers.get(col) or f"col_{col + 1}").strip()
                value = str(row.get(col) or "").strip()
                if value:
                    row_pairs.append(f"{header}={value}")
            if not row_pairs:
                continue
            row_text = "Table row: " + "; ".join(row_pairs)
            row_metadata = {
                **metadata,
                "chunk_kind": "table_row",
                "chunk_strategy": "table_row",
                "table_node_id": node.node_id,
                "table_row_index": row_index,
                "table_headers": [headers.get(col, f"col_{col + 1}") for col in sorted(headers)],
                "table_row": row,
            }
            row_metadata.update(
                {
                    "claim_role": "evidence",
                    "certainty": "strong",
                    "discourse_units": [
                        {
                            "role": "evidence",
                            "text": row_text[:240],
                        }
                    ],
                }
            )
            row_chunks.append(
                Chunk(
                    chunk_id=f"{node.node_id}:row:{row_index}",
                    doc_id=node.doc_id,
                    chunk_index=start_index + offset,
                    text=row_text,
                    source_path=source_path,
                    title=" / ".join(node.provenance.hierarchy_path) or title,
                    metadata=row_metadata,
                    modality="text",
                )
            )
        return row_chunks

    def _document_attachment_chunks(
        self,
        *,
        file_path: Path,
        source_path: str,
        doc_id: str,
        title: str,
        metadata: dict,
        start_index: int,
    ) -> list[Chunk]:
        suffix = file_path.suffix.lower()
        if suffix not in DOCUMENT_ATTACHMENT_SUFFIXES or not _document_attachment_index_enabled():
            return []
        if suffix == ".pdf":
            page_chunks = self._pdf_page_attachment_chunks(
                file_path=file_path,
                source_path=source_path,
                doc_id=doc_id,
                title=title,
                metadata=metadata,
                start_index=start_index,
            )
            image_chunks = self._pdf_rendered_page_image_chunks(
                file_path=file_path,
                source_path=source_path,
                doc_id=doc_id,
                title=title,
                metadata=metadata,
                start_index=start_index + len(page_chunks),
            )
            return [*page_chunks, *image_chunks]
        mime_type = DOCUMENT_ATTACHMENT_MIME_TYPES[suffix]
        attachment_chunks = [
            Chunk(
                chunk_id=f"{doc_id}:attachment:0",
                doc_id=doc_id,
                chunk_index=start_index,
                text="",
                source_path=source_path,
                title=title,
                metadata={
                    **metadata,
                    "chunk_kind": "document_attachment",
                    "chunk_strategy": "document_attachment",
                    "source_modality": "document",
                    "media_uri": self._media_uri_for_source(file_path, source_path),
                    "mime_type": mime_type,
                    "attachment_scope": "document",
                    "claim_role": "evidence",
                    "certainty": "weak",
                },
                modality="document",
                media_uri=self._media_uri_for_source(file_path, source_path),
                mime_type=mime_type,
            )
        ]
        embedded_images = self._ooxml_embedded_image_chunks(
            file_path=file_path,
            source_path=source_path,
            doc_id=doc_id,
            title=title,
            metadata=metadata,
            start_index=start_index + len(attachment_chunks),
        )
        return [*attachment_chunks, *embedded_images]

    def _pdf_page_attachment_chunks(
        self,
        *,
        file_path: Path,
        source_path: str,
        doc_id: str,
        title: str,
        metadata: dict,
        start_index: int,
    ) -> list[Chunk]:
        try:
            from pypdf import PdfReader, PdfWriter
        except ImportError:
            logger.warning("pypdf unavailable; skipping PDF page attachment chunks")
            return []
        try:
            reader = PdfReader(str(file_path))
        except Exception as exc:
            logger.warning("failed to inspect PDF pages for %s: %s", source_path, exc)
            return []
        page_count = len(reader.pages)
        if page_count <= 0:
            return []
        attachment_dir = self.config.parsed_dir / "attachments" / doc_id
        attachment_dir.mkdir(parents=True, exist_ok=True)
        chunks: list[Chunk] = []
        for page_index, page in enumerate(reader.pages, start=1):
            if page_index > _pdf_attachment_max_pages():
                break
            page_path = attachment_dir / f"page-{page_index}.pdf"
            try:
                writer = PdfWriter()
                writer.add_page(page)
                with page_path.open("wb") as handle:
                    writer.write(handle)
            except Exception as exc:
                logger.warning(
                    "failed to write PDF page attachment for %s page %d: %s",
                    source_path,
                    page_index,
                    exc,
                )
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}:page:{page_index}",
                    doc_id=doc_id,
                    chunk_index=start_index + len(chunks),
                    text="",
                    source_path=source_path,
                    title=f"{title} / page {page_index}",
                    metadata={
                        **metadata,
                        "chunk_kind": "document_page",
                        "chunk_strategy": "page_attachment",
                        "source_modality": "document",
                        "media_uri": str(page_path),
                        "mime_type": "application/pdf",
                        "attachment_scope": "page",
                        "page_number": page_index,
                        "page_count": page_count,
                        "claim_role": "evidence",
                        "certainty": "weak",
                    },
                    modality="document",
                    media_uri=str(page_path),
                    mime_type="application/pdf",
                )
            )
        return chunks

    def _pdf_rendered_page_image_chunks(
        self,
        *,
        file_path: Path,
        source_path: str,
        doc_id: str,
        title: str,
        metadata: dict,
        start_index: int,
    ) -> list[Chunk]:
        rendered = _render_pdf_page_images(
            file_path,
            self.config.parsed_dir / "attachments" / doc_id / "rendered",
        )
        chunks: list[Chunk] = []
        for page_number, image_path in rendered:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}:page-image:{page_number}",
                    doc_id=doc_id,
                    chunk_index=start_index + len(chunks),
                    text="",
                    source_path=source_path,
                    title=f"{title} / page image {page_number}",
                    metadata={
                        **metadata,
                        "chunk_kind": "rendered_page_image",
                        "chunk_strategy": "rendered_page_image",
                        "source_modality": "document",
                        "media_uri": str(image_path),
                        "mime_type": "image/png",
                        "attachment_scope": "page_image",
                        "page_number": page_number,
                        "claim_role": "evidence",
                        "certainty": "weak",
                    },
                    modality="image",
                    media_uri=str(image_path),
                    mime_type="image/png",
                )
            )
        return chunks

    def _ooxml_embedded_image_chunks(
        self,
        *,
        file_path: Path,
        source_path: str,
        doc_id: str,
        title: str,
        metadata: dict,
        start_index: int,
    ) -> list[Chunk]:
        suffix = file_path.suffix.lower()
        media_prefix = OOXML_MEDIA_PREFIXES.get(suffix)
        if not media_prefix or not _embedded_image_index_enabled():
            return []
        output_dir = self.config.parsed_dir / "attachments" / doc_id / "embedded"
        output_dir.mkdir(parents=True, exist_ok=True)
        chunks: list[Chunk] = []
        try:
            with zipfile.ZipFile(file_path) as archive:
                media_names = [
                    name
                    for name in archive.namelist()
                    if name.startswith(media_prefix)
                    and Path(name).suffix.lower() in IMAGE_SUFFIXES
                ]
                for name in sorted(media_names, key=_natural_key)[: _embedded_image_max_count()]:
                    suffix = Path(name).suffix.lower()
                    image_path = output_dir / f"image-{len(chunks) + 1}{suffix}"
                    image_path.write_bytes(archive.read(name))
                    chunks.append(
                        Chunk(
                            chunk_id=f"{doc_id}:embedded-image:{len(chunks) + 1}",
                            doc_id=doc_id,
                            chunk_index=start_index + len(chunks),
                            text="",
                            source_path=source_path,
                            title=f"{title} / embedded image {len(chunks) + 1}",
                            metadata={
                                **metadata,
                                "chunk_kind": "embedded_image",
                                "chunk_strategy": "embedded_image",
                                "source_modality": "document",
                                "media_uri": str(image_path),
                                "mime_type": MEDIA_MIME_TYPES[suffix],
                                "attachment_scope": "embedded_image",
                                "embedded_source_path": name,
                                "claim_role": "evidence",
                                "certainty": "weak",
                            },
                            modality="image",
                            media_uri=str(image_path),
                            mime_type=MEDIA_MIME_TYPES[suffix],
                        )
                    )
        except (OSError, zipfile.BadZipFile, KeyError) as exc:
            logger.warning("failed to extract embedded images from %s: %s", source_path, exc)
        return chunks

    async def _embed_items(
        self, items: list[list[EmbedItem]]
    ) -> list[list[float]]:
        embed_items_fn = getattr(self.embedding_client, "embed_items", None)
        if embed_items_fn is None:
            raise ModelGatewayError(
                "embedding client must implement embed_items; text-only compatibility path is disabled"
            )
        return await embed_items_fn(items)

    async def _prepare_media_document(
        self,
        file_path: Path,
        source_path: str,
        doc_id: str,
        kb_id: str,
    ) -> PreparedDocument:
        suffix = file_path.suffix.lower()
        modality = _modality_for_suffix(suffix)
        mime_type = _mime_type_for_suffix(suffix)
        title = file_path.stem
        try:
            stat = file_path.stat()
            source_size_bytes = stat.st_size
            source_modified_at = stat.st_mtime
        except OSError:
            source_size_bytes = 0
            source_modified_at = None
        document_metadata = {
            "kb_id": kb_id,
            "doc_type": modality,
            "source_modality": modality,
            "mime_type": mime_type,
            "media_uri": self._media_uri_for_source(file_path, source_path),
            "source_key": title.lower(),
            "source_file_name": file_path.name,
            "source_suffix": suffix,
            "source_size_bytes": source_size_bytes,
            "source_modified_at": source_modified_at,
            "source_content_hash": _file_sha256(file_path),
            "index_schema_version": "2026-05-rag-v1",
            "media_text_extraction": "not_applicable" if modality != "image" else "not_configured",
            "headings": [],
            "section_count": 0,
        }
        media_chunk = Chunk(
            chunk_id=f"{doc_id}:0",
            doc_id=doc_id,
            chunk_index=0,
            text="",
            source_path=source_path,
            title=title,
            metadata={
                **document_metadata,
                "chunk_kind": "media_object",
                "chunk_strategy": "media_object",
                "modality": modality,
                "source_node_id": f"{doc_id}:media",
                "claim_role": "evidence",
                "certainty": "weak",
            },
            modality=modality,
            media_uri=str(document_metadata["media_uri"]),
            mime_type=mime_type,
        )
        try:
            media_bytes = file_path.read_bytes()
        except OSError as exc:
            raise ParsingError(
                f"failed to read {modality} bytes for {source_path}: {exc}"
            ) from exc
        item: EmbedItem
        if modality == "image":
            item = ImageItem(data=media_bytes, mime_type=mime_type)
        elif modality == "audio":
            item = AudioItem(data=media_bytes, mime_type=mime_type)
        elif modality == "video":
            item = VideoItem(data=media_bytes, mime_type=mime_type)
        else:  # pragma: no cover — guarded by MEDIA_SUFFIXES
            raise ParsingError(f"unsupported media modality for {source_path}")

        chunks = [media_chunk]
        structured_document: StructuredDocument | None = None
        text = ""
        if modality == "image":
            structured_document = await self._try_parse_image_text(
                file_path=file_path,
                doc_id=doc_id,
                kb_id=kb_id,
                source_path=source_path,
            )
            if structured_document is not None:
                text = self._structured_document_text(structured_document)
                if text:
                    document_metadata["media_text_extraction"] = "ok"
                    structured_document.metadata.update(
                        {
                            "media_text_extraction": "ok",
                            "source_content_hash": document_metadata["source_content_hash"],
                            "source_size_bytes": document_metadata["source_size_bytes"],
                        }
                    )
                    text_chunks = self._structured_document_to_chunks(
                        structured_document,
                        source_path=source_path,
                        title=title,
                        metadata=document_metadata,
                    )
                    chunks.extend(
                        chunk.model_copy(
                            update={
                                "chunk_index": index,
                                "metadata": {
                                    **chunk.metadata,
                                    "kb_id": kb_id,
                                    "source_modality": modality,
                                    "media_uri": document_metadata["media_uri"],
                                    "media_text_extraction": "ok",
                                },
                            }
                        )
                        for index, chunk in enumerate(text_chunks, start=1)
                    )
                else:
                    document_metadata["media_text_extraction"] = "empty"

        chunks[0] = chunks[0].model_copy(
            update={
                "metadata": {
                    **chunks[0].metadata,
                    "media_text_extraction": document_metadata["media_text_extraction"],
                }
            }
        )
        document = Document(
            doc_id=doc_id,
            source_path=source_path,
            title=title,
            content=text,
            metadata=document_metadata,
        )
        embed_inputs = [[item], *self._chunks_to_embed_inputs(chunks[1:], file_path=file_path)]
        embeddings = await self._embed_items(embed_inputs)
        if len(embeddings) != len(chunks):
            raise ModelGatewayError(
                "embedding service returned an inconsistent number of vectors"
            )
        chunks = await self._attach_multivectors(chunks)
        return PreparedDocument(
            source_path=source_path,
            doc_id=doc_id,
            document=document,
            chunks=chunks,
            embeddings=embeddings,
            structured_document=structured_document,
        )

    async def _attach_multivectors(self, chunks: list[Chunk]) -> list[Chunk]:
        if self.multivector_provider is None:
            return chunks
        enriched: list[Chunk] = []
        for chunk in chunks:
            enriched.append(
                await asyncio.to_thread(
                    attach_chunk_multivectors,
                    chunk,
                    provider=self.multivector_provider,
                    store=self.multivector_store,
                )
            )
        return enriched

    async def _try_parse_image_text(
        self,
        *,
        file_path: Path,
        doc_id: str,
        kb_id: str,
        source_path: str,
    ) -> StructuredDocument | None:
        try:
            return await self.structured_parser.parse(
                file_path,
                doc_id=doc_id,
                kb_id=kb_id,
                source_path=source_path,
            )
        except Exception as exc:
            logger.warning(
                "image text extraction skipped for %s; keeping visual media chunk only: %s",
                source_path,
                exc,
            )
            return None

    def _chunks_to_embed_inputs(
        self,
        chunks: list[Chunk],
        file_path: Path | None = None,
    ) -> list[list[EmbedItem]]:
        inputs: list[list[EmbedItem]] = []
        for chunk in chunks:
            if chunk.modality in ("image", "audio", "video", "document"):
                source = None
                if chunk.media_uri:
                    candidate = Path(chunk.media_uri)
                    source = candidate if candidate.exists() else None
                if source is None and file_path is not None:
                    allowed_suffixes = (
                        DOCUMENT_ATTACHMENT_SUFFIXES
                        if chunk.modality == "document"
                        else MEDIA_SUFFIXES
                    )
                    source = file_path if file_path.suffix.lower() in allowed_suffixes else None
                if source is None:
                    raise ModelGatewayError(
                        f"cannot re-embed {chunk.modality} chunk {chunk.chunk_id}: "
                        "bytes not available; the upload directory may have been "
                        "pruned. Re-ingest the original file."
                    )
                mime = (
                    chunk.mime_type
                    or MEDIA_MIME_TYPES.get(source.suffix.lower())
                    or DOCUMENT_ATTACHMENT_MIME_TYPES.get(source.suffix.lower())
                    or "application/octet-stream"
                )
                payload = source.read_bytes()
                if chunk.modality == "image":
                    inputs.append([ImageItem(data=payload, mime_type=mime)])
                elif chunk.modality == "audio":
                    inputs.append([AudioItem(data=payload, mime_type=mime)])
                elif chunk.modality == "video":
                    inputs.append([VideoItem(data=payload, mime_type=mime)])
                else:
                    inputs.append([FileItem(path=source, mime_type=mime)])
            else:
                inputs.append([TextItem(chunk.text)])
        return inputs

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

    def _media_uri_for_source(self, file_path: Path, source_path: str) -> str:
        source = Path(source_path)
        if source.parts and source.parts[0] == "uploads":
            return str(self.config.upload_dir.joinpath(*source.parts[1:]))
        return str(file_path.resolve())

    def _stable_doc_id(
        self, source_path: str, kb_id: str
    ) -> str:
        identity = "|".join([kb_id, source_path])
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return f"doc-{digest[:24]}"

    def _cleanup_parsed_artifacts(
        self,
        source_path: str,
        active_doc_id: str,
        kb_id: str,
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
            ):
                artifact.unlink(missing_ok=True)

    def _delete_stale_committed_state(
        self,
        source_path: str,
        doc_id: str,
        kb_id: str,
        keep_chunk_ids: set[str],
    ) -> None:
        self.repository.delete_by_source(
            source_path,
            kb_id=kb_id,
            keep_chunk_ids=keep_chunk_ids,
        )
        self._cleanup_parsed_artifacts(
            source_path,
            doc_id,
            kb_id,
            remove_active_doc=False,
        )

    def _committed_chunk_ids(
        self,
        source_path: str,
        kb_id: str,
        active_doc_id: str,
    ) -> set[str]:
        artifact = self.config.parsed_dir / f"{active_doc_id}.json"
        if not artifact.exists():
            return set()
        try:
            payload = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return set()
        document = payload.get("document") if isinstance(payload, dict) else None
        metadata = document.get("metadata", {}) if isinstance(document, dict) else {}
        if not (
            isinstance(document, dict)
            and document.get("source_path") == source_path
            and metadata.get("kb_id", "default") == kb_id
        ):
            return set()
        raw_chunks = payload.get("chunks", []) if isinstance(payload, dict) else []
        if not isinstance(raw_chunks, list):
            return set()
        return {
            str(chunk.get("chunk_id"))
            for chunk in raw_chunks
            if isinstance(chunk, dict) and chunk.get("chunk_id")
        }

    def _committed_multivector_refs(
        self,
        source_path: str,
        kb_id: str,
        active_doc_id: str,
    ) -> set[str]:
        artifact = self.config.parsed_dir / f"{active_doc_id}.json"
        if not artifact.exists():
            return set()
        try:
            payload = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return set()
        document = payload.get("document") if isinstance(payload, dict) else None
        metadata = document.get("metadata", {}) if isinstance(document, dict) else {}
        if not (
            isinstance(document, dict)
            and document.get("source_path") == source_path
            and metadata.get("kb_id", "default") == kb_id
        ):
            return set()
        raw_chunks = payload.get("chunks", []) if isinstance(payload, dict) else []
        if not isinstance(raw_chunks, list):
            return set()
        refs: set[str] = set()
        for chunk in raw_chunks:
            if not isinstance(chunk, dict):
                continue
            metadata = chunk.get("metadata")
            if not isinstance(metadata, dict):
                continue
            ref = metadata.get("multi_vector_ref")
            if isinstance(ref, str) and ref:
                refs.add(ref)
        return refs

    def _delete_prepared_multivector_refs(
        self, prepared_documents: list[PreparedDocument]
    ) -> None:
        refs: set[str] = set()
        for prepared in prepared_documents:
            refs.update(_chunk_multivector_refs(prepared.chunks))
        self.multivector_store.delete_refs(refs)

    def _rollback_uncommitted_vector_state(
        self,
        source_path: str,
        kb_id: str,
        keep_chunk_ids: set[str],
    ) -> None:
        try:
            self.repository.delete_by_source(
                source_path,
                kb_id=kb_id,
                keep_chunk_ids=keep_chunk_ids,
            )
            if self.hybrid_retriever:
                self.hybrid_retriever.remove_by_source(
                    source_path,
                    kb_id=kb_id,
                    keep_chunk_ids=keep_chunk_ids,
                )
        except Exception as exc:
            logger.warning(
                "failed to roll back uncommitted vector state for %s: %s",
                source_path,
                exc,
            )


    def _stage_parsed_artifact(
        self,
        document: Document,
        chunks,
        structured_document: StructuredDocument | None = None,
    ) -> Path:
        artifact = {
            "document": document.model_dump(),
            "chunks": [chunk.model_dump() for chunk in chunks],
            "structured_document": (
                structured_document.model_dump() if structured_document else None
            ),
        }
        target = self.config.parsed_dir / f"{document.doc_id}.json"
        tmp_path = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        tmp_path.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return tmp_path

    def _commit_parsed_artifact(self, staged_path: Path, doc_id: str) -> None:
        target = self.config.parsed_dir / f"{doc_id}.json"
        os.replace(str(staged_path), str(target))


def _text_chunk_max_chars(strategy: str = "structural_leaf") -> int:
    env_name, default = _chunk_size_setting(strategy, "max")
    raw = os.getenv(env_name, str(default))
    try:
        return max(1000, int(raw))
    except ValueError:
        return default


def _text_chunk_overlap_chars(strategy: str = "structural_leaf") -> int:
    env_name, default = _chunk_size_setting(strategy, "overlap")
    raw = os.getenv(env_name, str(default))
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def _chunk_size_setting(strategy: str, kind: str) -> tuple[str, int]:
    if strategy == "code_window":
        return (
            "RAG_CODE_CHUNK_MAX_CHARS" if kind == "max" else "RAG_CODE_CHUNK_OVERLAP_CHARS",
            DEFAULT_CODE_CHUNK_MAX_CHARS if kind == "max" else DEFAULT_CODE_CHUNK_OVERLAP_CHARS,
        )
    if strategy == "log_window":
        return (
            "RAG_LOG_CHUNK_MAX_CHARS" if kind == "max" else "RAG_LOG_CHUNK_OVERLAP_CHARS",
            DEFAULT_LOG_CHUNK_MAX_CHARS if kind == "max" else DEFAULT_LOG_CHUNK_OVERLAP_CHARS,
        )
    return (
        "RAG_TEXT_CHUNK_MAX_CHARS" if kind == "max" else "RAG_TEXT_CHUNK_OVERLAP_CHARS",
        DEFAULT_TEXT_CHUNK_MAX_CHARS if kind == "max" else DEFAULT_TEXT_CHUNK_OVERLAP_CHARS,
    )


def _chunk_strategy_for_node(node_type: str, metadata: dict[str, object]) -> str:
    if metadata.get("doc_type") == "code":
        return "code_window"
    if metadata.get("doc_type") == "log":
        return "log_window"
    if node_type == "definition":
        return "definition_leaf"
    if node_type == "clause":
        return "clause_leaf"
    if node_type == "table":
        return "table_summary"
    return "structural_leaf"


def _document_attachment_index_enabled() -> bool:
    raw = os.getenv("RAG_DOCUMENT_ATTACHMENT_INDEX_ENABLED", "true")
    return raw.lower() in {"true", "1", "yes"}


def _pdf_attachment_max_pages() -> int:
    raw = os.getenv("RAG_PDF_ATTACHMENT_MAX_PAGES", "50")
    try:
        return max(1, int(raw))
    except ValueError:
        return 50


def _rendered_pdf_image_index_enabled() -> bool:
    raw = os.getenv("RAG_RENDERED_PAGE_IMAGE_INDEX_ENABLED", "true")
    return raw.lower() in {"true", "1", "yes"}


def _embedded_image_index_enabled() -> bool:
    raw = os.getenv("RAG_EMBEDDED_IMAGE_INDEX_ENABLED", "true")
    return raw.lower() in {"true", "1", "yes"}


def _embedded_image_max_count() -> int:
    raw = os.getenv("RAG_EMBEDDED_IMAGE_MAX_COUNT", "100")
    try:
        return max(1, int(raw))
    except ValueError:
        return 100


def _render_pdf_page_images(pdf_path: Path, output_dir: Path) -> list[tuple[int, Path]]:
    if not _rendered_pdf_image_index_enabled():
        return []
    renderer = shutil.which("pdftoppm")
    if not renderer:
        logger.info("pdftoppm not available; skipping rendered PDF page image chunks")
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    max_pages = _pdf_attachment_max_pages()
    prefix = output_dir / "page"
    cmd = [
        renderer,
        "-png",
        "-r",
        os.getenv("RAG_RENDERED_PAGE_IMAGE_DPI", "144"),
        "-f",
        "1",
        "-l",
        str(max_pages),
        str(pdf_path),
        str(prefix),
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.warning("failed to render PDF pages for %s: %s", pdf_path, exc)
        return []
    rendered: list[tuple[int, Path]] = []
    for path in sorted(output_dir.glob("page-*.png"), key=lambda item: _natural_key(item.name)):
        stem_number = path.stem.rsplit("-", 1)[-1]
        try:
            page_number = int(stem_number)
        except ValueError:
            continue
        rendered.append((page_number, path))
    return rendered


def _natural_key(value: object) -> list[object]:
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", str(value))
    ]


def _split_text_for_index(
    text: str,
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    value = text.strip()
    if len(value) <= max_chars:
        return [value] if value else []
    overlap = min(overlap_chars, max_chars // 4)
    parts: list[str] = []
    start = 0
    while start < len(value):
        end = min(len(value), start + max_chars)
        if end < len(value):
            split_at = max(
                value.rfind("\n\n", start, end),
                value.rfind("\n", start, end),
                value.rfind("。", start, end),
                value.rfind(".", start, end),
            )
            if split_at > start + max_chars // 2:
                end = split_at + 1
        part = value[start:end].strip()
        if part:
            parts.append(part)
        if end >= len(value):
            break
        start = max(end - overlap, start + 1)
    return parts


def _discourse_metadata(
    text: str,
    *,
    node_type: str,
    has_table: bool,
) -> dict[str, object]:
    role = _discourse_role(node_type=node_type, has_table=has_table)
    metadata: dict[str, object] = {
        "claim_role": role,
        "certainty": "strong" if has_table or node_type in {"definition", "clause"} else "weak",
        "discourse_units": _discourse_units(text, role),
    }
    return metadata


def _discourse_role(*, node_type: str, has_table: bool) -> str:
    if has_table or node_type in {"table", "definition", "clause"}:
        return "evidence"
    return "conclusion"


def _discourse_units(text: str, role: str) -> list[dict[str, str]]:
    normalized = " ".join(text.split())
    if not normalized:
        return []
    return [{"role": role, "text": normalized[:240]}]


def _file_sha256(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _chunk_multivector_refs(chunks: list[Chunk]) -> set[str]:
    refs: set[str] = set()
    for chunk in chunks:
        ref = (chunk.metadata or {}).get("multi_vector_ref")
        if isinstance(ref, str) and ref:
            refs.add(ref)
    return refs
