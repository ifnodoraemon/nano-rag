from __future__ import annotations

import hashlib
import asyncio
import io
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

from app.ingestion.dedup import find_duplicates_for_content, remove_document_everywhere
from app.ingestion.loader import discover_files
from app.ingestion.parse_cache import ParseCache, build_fingerprint, content_hash
from app.ingestion.metadata import build_source_key, extract_document_metadata
from app.ingestion.graph_extractor import GraphExtractor
from app.wiki.compiler import WikiCompiler
from app.wiki.search import WikiSearcher
from app.ingestion.structured_parser import StructuredDocumentParser
from app.core.exceptions import ConfigurationError, ParsingError, StoreError
from app.model_client.document_parser import DocumentParserClient
from app.model_client.generation import GenerationClient

from app.retrieval.graph_store import GraphStore
from app.schemas.chunk import Chunk
from app.schemas.document import Document, IngestResponse
from app.schemas.structured import NodeType, StructuredDocument

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
DEFAULT_TEXT_CHUNK_MAX_CHARS = 1500
DEFAULT_TEXT_CHUNK_OVERLAP_CHARS = 150
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
    structured_document: StructuredDocument | None = None


class IngestionPipeline:
    def __init__(
        self,
        config: AppConfig,
        generation_client: GenerationClient,
        tracing_manager: TracingManager,
        document_parser: DocumentParserClient | None = None,
        graph_store: GraphStore | None = None,
        wiki_compiler: WikiCompiler | None = None,
        wiki_searcher: WikiSearcher | None = None,
    ) -> None:
        self.config = config
        self.generation_client = generation_client
        self.tracing_manager = tracing_manager
        self.document_parser = document_parser
        self.graph_store = graph_store
        self.wiki_compiler = wiki_compiler
        self.wiki_searcher = wiki_searcher
        self.structured_parser = StructuredDocumentParser(document_parser)
        self.graph_extractor = GraphExtractor(generation_client)
        chunk_settings = config.settings.get("chunk", {}) or {}
        self.parse_cache = ParseCache(
            config.parsed_dir,
            build_fingerprint(
                parser_provider=getattr(document_parser, "provider", None),
                parser_model=getattr(document_parser, "alias", None),
                chunk_size=chunk_settings.get("size"),
                chunk_overlap=chunk_settings.get("overlap"),
            ),
            enabled=_parse_cache_enabled(),
        )


    async def run(
        self,
        path: str,
        kb_id: str = "default",
        source_path_overrides: dict[str, str] | None = None,
        force: bool = False,
    ) -> IngestResponse:
        with self.tracing_manager.span(
            "ingestion.run",
            {
                "ingestion.path": path,
                "ingestion.kb_id": kb_id,
            },
        ):
            # Recursive filesystem walk: keep it off the event loop.
            files = await asyncio.to_thread(discover_files, path)
            prepared_documents: list[PreparedDocument] = []
            unchanged_doc_ids: list[str] = []

            # Preparation is pure I/O (two model calls per document) and touches
            # no shared state, so it runs concurrently up to a bound; commits
            # below stay strictly sequential. Result order follows `files`, so
            # the commit sequence - and therefore rollback - is deterministic.
            for doc_id, prepared in await self._prepare_documents(
                files,
                kb_id=kb_id,
                source_path_overrides=source_path_overrides,
                force=force,
            ):
                if prepared is None:
                    # Reported, never silent: the caller must be able to tell
                    # "nothing needed doing" from "nothing was found".
                    unchanged_doc_ids.append(doc_id)
                    continue
                prepared_documents.append(prepared)

            doc_count = len(prepared_documents)
            chunk_count = sum(len(prepared.chunks) for prepared in prepared_documents)

            self.config.parsed_dir.mkdir(parents=True, exist_ok=True)
            # Artifacts, graph rows and wiki pages live in three stores with no
            # shared transaction. Committing artifacts first and failing later
            # (graph upsert, wiki write) left discovery serving the previous
            # version while deep-read read the new artifact. Track what this
            # batch committed and undo it on failure so the three stay aligned.
            committed: list[str] = []
            try:
                for prepared in prepared_documents:
                    doc_id = prepared.document.doc_id
                    artifact_tmp = await asyncio.to_thread(
                        self._stage_parsed_artifact,
                        prepared.document,
                        prepared.chunks,
                        prepared.structured_document,
                    )
                    await asyncio.to_thread(
                        self._commit_parsed_artifact, artifact_tmp, doc_id
                    )
                    committed.append(doc_id)
                    if self.graph_store and prepared.structured_document:
                        await asyncio.to_thread(
                            self.graph_store.upsert_document,
                            prepared.structured_document,
                        )

                # One lock acquisition + one version-ledger/index write for the
                # whole batch instead of per-file full rewrites.
                if self.wiki_compiler and prepared_documents:
                    await asyncio.to_thread(
                        self.wiki_compiler.upsert_documents,
                        [
                            (prepared.document, prepared.chunks)
                            for prepared in prepared_documents
                        ],
                    )
                if self.graph_store and any(
                    prepared.structured_document for prepared in prepared_documents
                ):
                    # Once-per-job maintenance: reclaim entity rows whose node
                    # links were rewritten by this batch (kept out of the
                    # per-document hot path).
                    await asyncio.to_thread(self.graph_store.collect_orphan_entities)
            except BaseException:
                await self._rollback_commit(prepared_documents, committed)
                raise
            # Only once the new copies are durably stored: deleting the old
            # document before that would lose the content entirely if the
            # commit failed. A failure here leaves a visible duplicate instead.
            superseded = await self._supersede_same_content(prepared_documents)
            if self.wiki_compiler and self.wiki_searcher and prepared_documents:
                # Incremental re-index (not a full rebuild) and off the loop:
                # refresh() re-scanned and re-scored the entire corpus after
                # every ingest, blocking every concurrent request.
                await asyncio.to_thread(self.wiki_searcher.sync)
            if superseded:
                logger.info(
                    "ingest superseded %d same-content document(s): %s",
                    len(superseded),
                    ", ".join(superseded),
                )
            if unchanged_doc_ids:
                logger.info(
                    "ingest skipped %d unchanged document(s): %s",
                    len(unchanged_doc_ids),
                    ", ".join(unchanged_doc_ids[:10]),
                )
            return IngestResponse(
                documents=doc_count,
                chunks=chunk_count,
                superseded_doc_ids=superseded,
                unchanged_doc_ids=unchanged_doc_ids,
            )

    async def _supersede_same_content(
        self, prepared_documents: list[PreparedDocument]
    ) -> list[str]:
        """Remove existing documents whose content equals an incoming one.

        Runs after the batch is committed and the wiki written, so a failure
        here leaves a duplicate (visible, harmless) rather than deleting the
        old copy while the new one is not yet stored.
        """
        superseded: list[str] = []
        for prepared in prepared_documents:
            kb_id = str(prepared.document.metadata.get("kb_id") or "default")
            duplicates = await asyncio.to_thread(
                find_duplicates_for_content,
                self.config.parsed_dir,
                kb_id=kb_id,
                content=prepared.document.content,
                exclude_doc_id=prepared.document.doc_id,
            )
            for duplicate in duplicates:
                touched = await asyncio.to_thread(
                    remove_document_everywhere,
                    doc_id=duplicate.doc_id,
                    kb_id=duplicate.kb_id,
                    parsed_dir=self.config.parsed_dir,
                    graph_store=self.graph_store,
                    wiki_compiler=self.wiki_compiler,
                )
                superseded.append(duplicate.doc_id)
                logger.info(
                    "superseded duplicate %s (content match with %s, stores: %s)",
                    duplicate.doc_id,
                    prepared.document.doc_id,
                    ",".join(touched) or "none",
                )
        return superseded

    async def _rollback_commit(
        self, prepared_documents: list[PreparedDocument], committed: list[str]
    ) -> None:
        """Undo this batch's partial writes so the stores stay consistent.

        Best-effort by design: a rollback failure must not mask the original
        error, but it is logged loudly because the stores are then divergent.
        """
        committed_ids = set(committed)
        for prepared in prepared_documents:
            doc_id = prepared.document.doc_id
            if doc_id in committed_ids:
                try:
                    await asyncio.to_thread(
                        (self.config.parsed_dir / f"{doc_id}.json").unlink,
                        missing_ok=True,
                    )
                except OSError:
                    logger.exception("rollback: could not remove artifact for %s", doc_id)
            if self.graph_store and prepared.structured_document:
                try:
                    await asyncio.to_thread(
                        self.graph_store.delete_document,
                        doc_id=doc_id,
                        kb_id=prepared.document.metadata.get("kb_id", "default"),
                    )
                except Exception:  # noqa: BLE001 - reported, never re-raised
                    logger.exception("rollback: could not delete graph rows for %s", doc_id)
            if self.wiki_compiler:
                try:
                    await asyncio.to_thread(self.wiki_compiler.remove_document, doc_id)
                except Exception:  # noqa: BLE001 - reported, never re-raised
                    logger.exception("rollback: could not remove wiki page for %s", doc_id)
        if self.wiki_compiler and self.wiki_searcher:
            try:
                await asyncio.to_thread(self.wiki_searcher.refresh)
            except Exception:  # noqa: BLE001
                logger.exception("rollback: wiki searcher refresh failed")

    async def _prepare_documents(
        self,
        files: list[Path],
        *,
        kb_id: str,
        source_path_overrides: dict[str, str] | None,
        force: bool,
    ) -> list[tuple[str, PreparedDocument | None]]:
        """Prepare every file, concurrently up to ``RAG_INGEST_CONCURRENCY``.

        Each document costs two model calls (parse + graph extraction) and writes
        nothing, so concurrency here cannot corrupt a store - all writes happen
        later in the sequential commit. ``TaskGroup`` cancels the siblings when one
        document fails instead of leaving them running against a doomed batch.
        """
        concurrency = _ingest_concurrency()

        async def prepare(file_path: Path) -> tuple[str, PreparedDocument | None]:
            doc_id = self._stable_doc_id(
                self._resolve_source_path(file_path, source_path_overrides), kb_id
            )
            with self.tracing_manager.span(
                "ingestion.file", {"ingestion.file_path": str(file_path)}
            ):
                prepared = await self._prepare_document(
                    file_path,
                    kb_id=kb_id,
                    source_path_overrides=source_path_overrides,
                    force=force,
                )
            return doc_id, prepared

        if concurrency <= 1 or len(files) <= 1:
            return [await prepare(file_path) for file_path in files]

        semaphore = asyncio.Semaphore(concurrency)

        async def bounded(file_path: Path) -> tuple[str, PreparedDocument | None]:
            async with semaphore:
                return await prepare(file_path)

        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(bounded(file_path)) for file_path in files]
        return [task.result() for task in tasks]

    async def _prepare_document(
        self,
        file_path: Path,
        kb_id: str,
        source_path_overrides: dict[str, str] | None = None,
        *,
        force: bool = False,
    ) -> PreparedDocument | None:
        source_path = self._resolve_source_path(file_path, source_path_overrides)
        doc_id = self._stable_doc_id(source_path, kb_id)
        suffix = file_path.suffix.lower()
        if not force and await asyncio.to_thread(
            self._artifact_matches_source, doc_id, file_path
        ):
            # Byte-identical to what is already committed under this path, and the
            # parse/chunk configuration has not changed: re-parsing would spend a
            # model call (and a graph-extraction call) to reproduce the artifact
            # that is already on disk.
            return None
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
            cache=self.parse_cache,
        )
        structured_document.graph = await self.graph_extractor.extract(structured_document)
        structured_document.metadata["graph_extraction"] = {"status": "ok"}
        text = await asyncio.to_thread(self._structured_document_text, structured_document)
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
            await asyncio.to_thread(
                self._source_version_metadata,
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
        chunks = await asyncio.to_thread(
            self._structured_document_to_chunks,
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
            await asyncio.to_thread(
                self._document_attachment_chunks,
                file_path=file_path,
                source_path=source_path,
                doc_id=doc_id,
                title=document.title,
                metadata=document.metadata,
                start_index=len(chunks),
            )
        )
        return PreparedDocument(
            source_path=source_path,
            doc_id=doc_id,
            document=document,
            chunks=chunks,
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
        except OSError as exc:
            raise ParsingError(
                f"cannot stat source file {file_path} during ingest: {exc}"
            ) from exc
        return {
            "source_file_name": file_path.name,
            "source_suffix": file_path.suffix.lower(),
            "source_size_bytes": size_bytes,
            "source_modified_at": modified_at,
            "source_content_hash": _file_sha256(file_path),
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
        attachment_root = self.config.parsed_dir / "attachments"
        final_dir = attachment_root / doc_id
        # Build the whole attachment tree in a staging directory and swap it in
        # at the end: an in-place rebuild deleted the previous version's files
        # before the new ones existed, so a rendering failure left the still
        # committed artifact pointing at missing media.
        staging_dir = attachment_root / f"{doc_id}.{uuid4().hex}.tmp"
        try:
            if suffix == ".pdf":
                page_chunks = self._pdf_page_attachment_chunks(
                    file_path=file_path,
                    source_path=source_path,
                    doc_id=doc_id,
                    title=title,
                    metadata=metadata,
                    start_index=start_index,
                    attachment_dir=staging_dir,
                )
                image_chunks = self._pdf_rendered_page_image_chunks(
                    file_path=file_path,
                    source_path=source_path,
                    doc_id=doc_id,
                    title=title,
                    metadata=metadata,
                    start_index=start_index + len(page_chunks),
                    attachment_dir=staging_dir,
                )
                built = [*page_chunks, *image_chunks]
            else:
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
                built = [
                    *attachment_chunks,
                    *self._ooxml_embedded_image_chunks(
                        file_path=file_path,
                        source_path=source_path,
                        doc_id=doc_id,
                        title=title,
                        metadata=metadata,
                        start_index=start_index + len(attachment_chunks),
                        attachment_dir=staging_dir,
                    ),
                ]
            return self._commit_attachment_tree(staging_dir, final_dir, built)
        except BaseException:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

    def _commit_attachment_tree(
        self, staging_dir: Path, final_dir: Path, chunks: list[Chunk]
    ) -> list[Chunk]:
        """Atomically replace a document's attachment tree and repoint chunks.

        Attachments are built under a staging path, so the media_uri values the
        builders embedded must be rewritten once the tree lands in its final
        location.
        """
        if not chunks and not staging_dir.exists():
            # Nothing was produced: leave the (already committed) tree alone.
            return []
        if final_dir.exists():
            shutil.rmtree(final_dir)
        if staging_dir.exists():
            staging_dir.parent.mkdir(parents=True, exist_ok=True)
            os.replace(str(staging_dir), str(final_dir))
        staging_prefix = str(staging_dir)
        final_prefix = str(final_dir)
        committed: list[Chunk] = []
        for chunk in chunks:
            media_uri = chunk.media_uri
            metadata = dict(chunk.metadata or {})
            metadata_uri = metadata.get("media_uri")
            if isinstance(media_uri, str):
                media_uri = media_uri.replace(staging_prefix, final_prefix)
            if isinstance(metadata_uri, str):
                metadata["media_uri"] = metadata_uri.replace(staging_prefix, final_prefix)
            committed.append(
                chunk.model_copy(update={"media_uri": media_uri, "metadata": metadata})
            )
        return committed

    def _pdf_page_attachment_chunks(
        self,
        *,
        file_path: Path,
        source_path: str,
        doc_id: str,
        title: str,
        metadata: dict,
        start_index: int,
        attachment_dir: Path,
    ) -> list[Chunk]:
        try:
            from pypdf import PdfReader, PdfWriter
        except ImportError as exc:  # pragma: no cover - pypdf is in requirements
            raise ParsingError(
                "pypdf is required for PDF page attachments but is not installed"
            ) from exc
        try:
            reader = PdfReader(str(file_path))
        except Exception as exc:
            raise ParsingError(
                f"cannot open PDF {source_path} for page attachments: {exc}"
            ) from exc
        page_count = len(reader.pages)
        if page_count <= 0:
            return []
        attachment_dir.mkdir(parents=True, exist_ok=True)
        chunks: list[Chunk] = []
        for page_index, page in enumerate(reader.pages, start=1):
            if page_index > _pdf_attachment_max_pages():
                break
            page_path = attachment_dir / f"page-{page_index}.pdf"
            try:
                writer = PdfWriter()
                writer.add_page(page)
                _write_bytes_atomic(page_path, _pdf_writer_bytes(writer))
            except Exception as exc:
                raise ParsingError(
                    f"failed to write PDF page attachment for {source_path} "
                    f"page {page_index}: {exc}"
                ) from exc
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
        attachment_dir: Path,
    ) -> list[Chunk]:
        rendered = _render_pdf_page_images(file_path, attachment_dir / "rendered")
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
        attachment_dir: Path,
    ) -> list[Chunk]:
        suffix = file_path.suffix.lower()
        media_prefix = OOXML_MEDIA_PREFIXES.get(suffix)
        if not media_prefix or not _embedded_image_index_enabled():
            return []
        output_dir = attachment_dir / "embedded"
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
                    _write_bytes_atomic(image_path, archive.read(name))
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
            raise ParsingError(
                f"failed to extract embedded images from {source_path}: {exc}"
            ) from exc
        return chunks

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
        except OSError as exc:
            raise ParsingError(
                f"cannot stat media file {file_path} during ingest: {exc}"
            ) from exc
        document_metadata = {
            "kb_id": kb_id,
            "doc_type": modality,
            "source_modality": modality,
            "mime_type": mime_type,
            "media_uri": self._media_uri_for_source(file_path, source_path),
            # Same derivation as text documents: title alone collided across
            # directories, making unrelated files "versions" of each other.
            "source_key": build_source_key(source_path, title),
            "source_file_name": file_path.name,
            "source_suffix": suffix,
            "source_size_bytes": source_size_bytes,
            "source_modified_at": source_modified_at,
            "source_content_hash": await asyncio.to_thread(_file_sha256, file_path),
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
            modality=_modality_literal(modality),
            media_uri=str(document_metadata["media_uri"]),
            mime_type=mime_type,
        )
        chunks = [media_chunk]
        structured_document: StructuredDocument | None = None
        text = ""
        if modality == "image":
            # Images have no local text to extract: the multimodal parser is the
            # only path, so it is required and its failure is terminal.
            structured_document = await self.structured_parser.parse(
                file_path,
                doc_id=doc_id,
                kb_id=kb_id,
                source_path=source_path,
            )
            if structured_document is not None:
                text = await asyncio.to_thread(
                    self._structured_document_text, structured_document
                )
                if text:
                    document_metadata["media_text_extraction"] = "ok"
                    structured_document.metadata.update(
                        {
                            "media_text_extraction": "ok",
                            "source_content_hash": document_metadata["source_content_hash"],
                            "source_size_bytes": document_metadata["source_size_bytes"],
                        }
                    )
                    text_chunks = await asyncio.to_thread(
                        self._structured_document_to_chunks,
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

        if not text.strip():
            # Audio/video (and images with no configured parser) used to be
            # ingested as an empty document that the job reported as success.
            # An empty document is undiscoverable: no BM25 text, no graph, no
            # citations. Fail loudly instead of storing a silent hole.
            raise ParsingError(
                f"{file_path.name} ({modality}) produced no text during ingest "
                f"(media_text_extraction={document_metadata['media_text_extraction']}). "
                "Configure a multimodal document parser that supports this media "
                "type, or ingest a text-bearing representation of it."
            )

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
        return PreparedDocument(
            source_path=source_path,
            doc_id=doc_id,
            document=document,
            chunks=chunks,
            structured_document=structured_document,
        )



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

    def _artifact_matches_source(self, doc_id: str, file_path: Path) -> bool:
        """True when the committed artifact already reflects these exact bytes.

        Deliberately strict: the artifact must exist, be readable, record the same
        content hash, the same ingest fingerprint and a non-empty chunk list. Any
        doubt falls through to a real re-ingest, so this can only ever save work -
        never serve a stale or empty document.
        """
        artifact_path = self.config.parsed_dir / f"{doc_id}.json"
        if not artifact_path.exists():
            return False
        try:
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A corrupt artifact is a real problem, but not one this check should
            # report: fall through and let the ingest rewrite it.
            return False
        if not isinstance(artifact, dict):
            return False
        if artifact.get("ingest_fingerprint") != self.parse_cache.fingerprint:
            return False
        chunks = artifact.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            return False
        document = artifact.get("document")
        if not isinstance(document, dict):
            return False
        metadata = document.get("metadata")
        if not isinstance(metadata, dict):
            return False
        recorded = metadata.get("source_content_hash")
        if not isinstance(recorded, str) or not recorded:
            return False
        try:
            current = content_hash(file_path)
        except StoreError:
            return False
        return recorded == current

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
            # What produced this artifact. A later ingest compares it to decide
            # whether re-parsing could even change anything.
            "ingest_fingerprint": self.parse_cache.fingerprint,
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
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{env_name} must be an integer, got {raw!r}") from exc
    if value < 1000:
        # Silently clamping up hid a misconfiguration behind "it works".
        raise ConfigurationError(
            f"{env_name} must be >= 1000 (got {value}); small windows shred "
            "documents into meaningless chunks."
        )
    return value


def _text_chunk_overlap_chars(strategy: str = "structural_leaf") -> int:
    env_name, default = _chunk_size_setting(strategy, "overlap")
    raw = os.getenv(env_name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{env_name} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ConfigurationError(f"{env_name} must be >= 0, got {value}")
    return value


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


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Write via a temp file + rename.

    The target path is referenced by the committed parsed artifact, so a crash
    mid-write must not leave a truncated file behind.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        tmp_path.write_bytes(payload)
        os.replace(str(tmp_path), str(path))
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _pdf_writer_bytes(writer) -> bytes:  # noqa: ANN001
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


_TRUE_TOKENS = {"true", "1", "yes", "on"}
_FALSE_TOKENS = {"false", "0", "no", "off"}


def _modality_literal(value: str):  # noqa: ANN202
    """Narrow a suffix-derived modality to the schema's Literal.

    Chunk.modality is a Literal; passing an unmapped suffix would fail
    validation deep inside the ingest job instead of at the mapping.
    """
    from typing import cast, get_args

    from app.schemas.chunk import Chunk as _Chunk

    allowed = set(get_args(_Chunk.model_fields["modality"].annotation))
    if value not in allowed:
        raise ParsingError(
            f"unsupported media modality {value!r}; expected one of {sorted(allowed)}"
        )
    return cast(str, value)


def _strict_bool_env(name: str, default: str) -> bool:
    """Boolean env parsing that refuses to guess.

    A typo ("ture", "enable", "") used to read as *disabled*, silently
    dropping an ingest stage. Malformed configuration must fail loudly.
    """
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default.strip().lower() in _TRUE_TOKENS
    token = raw.strip().lower()
    if token in _TRUE_TOKENS:
        return True
    if token in _FALSE_TOKENS:
        return False
    raise ConfigurationError(
        f"{name} must be a boolean (true/false), got {raw!r}"
    )


def _ingest_concurrency() -> int:
    """How many documents to prepare at once (each is two gateway round trips)."""
    raw = os.getenv("RAG_INGEST_CONCURRENCY", "4")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"RAG_INGEST_CONCURRENCY must be an integer, got {raw!r}"
        ) from exc
    if value < 1:
        raise ConfigurationError(f"RAG_INGEST_CONCURRENCY must be >= 1, got {value}")
    return value


def _parse_cache_enabled() -> bool:
    return _strict_bool_env("RAG_PARSE_CACHE_ENABLED", "true")


def _document_attachment_index_enabled() -> bool:
    return _strict_bool_env("RAG_DOCUMENT_ATTACHMENT_INDEX_ENABLED", "true")


def _pdf_attachment_max_pages() -> int:
    raw = os.getenv("RAG_PDF_ATTACHMENT_MAX_PAGES", "50")
    try:
        return max(1, int(raw))
    except ValueError as exc:
        raise ConfigurationError(
            f"RAG_PDF_ATTACHMENT_MAX_PAGES must be an integer, got {raw!r}"
        ) from exc


def _rendered_pdf_image_index_enabled() -> bool:
    return _strict_bool_env("RAG_RENDERED_PAGE_IMAGE_INDEX_ENABLED", "true")


def _embedded_image_index_enabled() -> bool:
    return _strict_bool_env("RAG_EMBEDDED_IMAGE_INDEX_ENABLED", "true")


def _embedded_image_max_count() -> int:
    raw = os.getenv("RAG_EMBEDDED_IMAGE_MAX_COUNT", "100")
    try:
        return max(1, int(raw))
    except ValueError as exc:
        raise ConfigurationError(
            f"RAG_EMBEDDED_IMAGE_MAX_COUNT must be an integer, got {raw!r}"
        ) from exc


def _pdf_render_timeout_seconds() -> float:
    raw = os.getenv("RAG_PDF_RENDER_TIMEOUT_SECONDS", "300")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"RAG_PDF_RENDER_TIMEOUT_SECONDS must be a number, got {raw!r}"
        ) from exc
    if value <= 0:
        raise ConfigurationError(
            f"RAG_PDF_RENDER_TIMEOUT_SECONDS must be > 0, got {value}"
        )
    return value


def _render_pdf_page_images(pdf_path: Path, output_dir: Path) -> list[tuple[int, Path]]:
    if not _rendered_pdf_image_index_enabled():
        return []
    renderer = shutil.which("pdftoppm")
    if not renderer:
        raise ConfigurationError(
            "RAG_RENDERED_PAGE_IMAGE_INDEX_ENABLED is true but pdftoppm is not "
            "installed; install poppler-utils or disable rendered page image indexing"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    max_pages = _pdf_attachment_max_pages()
    prefix = output_dir / "page"
    cmd = [
        renderer,
        "-png",
        "-r",
        _rendered_pdf_image_dpi(),
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
            # An unbounded renderer (corrupt PDF, poppler deadlock, enormous
            # page) used to block this ingest thread forever: the job stayed
            # RUNNING with nothing to reclaim it.
            timeout=_pdf_render_timeout_seconds(),
        )
    except subprocess.TimeoutExpired as exc:
        raise ParsingError(
            f"PDF page rendering timed out after "
            f"{_pdf_render_timeout_seconds():.0f}s for {pdf_path}; the file may "
            "be corrupt or too large to render."
        ) from exc
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ParsingError(
            f"failed to render PDF pages for {pdf_path}: "
            f"{exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) else exc}"
        ) from exc
    rendered: list[tuple[int, Path]] = []
    for path in sorted(output_dir.glob("page-*.png"), key=lambda item: _natural_key(item.name)):
        stem_number = path.stem.rsplit("-", 1)[-1]
        try:
            page_number = int(stem_number)
        except ValueError:
            continue
        rendered.append((page_number, path))
    if not rendered:
        raise ParsingError(
            f"pdftoppm produced no page images for {pdf_path}"
        )
    return rendered


def _rendered_pdf_image_dpi() -> str:
    raw = os.getenv("RAG_RENDERED_PAGE_IMAGE_DPI", "144")
    try:
        return str(max(36, int(raw)))
    except ValueError as exc:
        raise ConfigurationError(
            f"RAG_RENDERED_PAGE_IMAGE_DPI must be an integer, got {raw!r}"
        ) from exc


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as exc:
        raise ParsingError(f"cannot read source file {path} for hashing: {exc}") from exc
    return digest.hexdigest()



