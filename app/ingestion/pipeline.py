from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from app.ingestion.loader import discover_files
from app.ingestion.metadata import extract_document_metadata
from app.ingestion.structured_parser import StructuredDocumentParser
from app.core.exceptions import ModelGatewayError, ParsingError
from app.model_client.embeddings import EmbeddingClient
from app.model_client.document_parser import DocumentParserClient
from app.model_client.multimodal_embedding import (
    AudioItem,
    EmbedItem,
    ImageItem,
    TextItem,
    VideoItem,
)
from app.retrieval.hybrid_retriever import HybridRetriever
from app.schemas.chunk import Chunk
from app.schemas.document import Document, IngestResponse
from app.schemas.structured import NodeType, StructuredDocument

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".mpeg", ".mpg"}
MEDIA_SUFFIXES = IMAGE_SUFFIXES | AUDIO_SUFFIXES | VIDEO_SUFFIXES

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

from app.vectorstore.repository import VectorRepository
from app.wiki.compiler import WikiCompiler
from app.wiki.search import WikiSearcher

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
        tracing_manager: TracingManager,
        document_parser: DocumentParserClient | None = None,
        hybrid_retriever: HybridRetriever | None = None,
        wiki_compiler: WikiCompiler | None = None,
        wiki_searcher: WikiSearcher | None = None,
    ) -> None:
        self.config = config
        self.repository = repository
        self.embedding_client = embedding_client
        self.tracing_manager = tracing_manager
        self.document_parser = document_parser
        self.hybrid_retriever = hybrid_retriever
        self.wiki_compiler = wiki_compiler
        self.wiki_searcher = wiki_searcher
        self.structured_parser = StructuredDocumentParser(document_parser)

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

            self.config.parsed_dir.mkdir(parents=True, exist_ok=True)
            for prepared in prepared_documents:
                self._delete_committed_state(prepared.source_path, prepared.doc_id, kb_id)
                self.repository.upsert(
                    prepared.document,
                    prepared.chunks,
                    prepared.embeddings,
                )
                if self.hybrid_retriever:
                    self.hybrid_retriever.index_chunks(prepared.chunks)
                self._write_parsed_artifact(
                    prepared.document,
                    prepared.chunks,
                    prepared.structured_document,
                )
                if self.wiki_compiler:
                    self.wiki_compiler.upsert_document(prepared.document, prepared.chunks)
                    wiki_updated = True

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
            structured_document=structured_document,
        )

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
            if table_payload:
                chunk_metadata["table"] = table_payload
            chunks.append(
                Chunk(
                    chunk_id=node.node_id,
                    doc_id=structured_document.doc_id,
                    chunk_index=index,
                    text=text,
                    source_path=source_path,
                    title=" / ".join(node.provenance.hierarchy_path) or title,
                    metadata=chunk_metadata,
                    modality="text",
                )
            )
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
        document_metadata = {
            "kb_id": kb_id,
            "doc_type": modality,
            "modality": modality,
            "mime_type": mime_type,
            "media_uri": self._media_uri_for_source(file_path, source_path),
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
                "modality": modality,
                "mime_type": mime_type,
                "media_uri": document_metadata["media_uri"],
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
        embeddings = await self._embed_items([[item]])
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
        self,
        chunks: list[Chunk],
        file_path: Path | None = None,
    ) -> list[list[EmbedItem]]:
        inputs: list[list[EmbedItem]] = []
        for chunk in chunks:
            if chunk.modality in ("image", "audio", "video"):
                source = (
                    file_path
                    if file_path is not None
                    and file_path.suffix.lower() in MEDIA_SUFFIXES
                    else None
                )
                if source is None and chunk.media_uri:
                    candidate = Path(chunk.media_uri)
                    source = candidate if candidate.exists() else None
                if source is None:
                    raise ModelGatewayError(
                        f"cannot re-embed {chunk.modality} chunk {chunk.chunk_id}: "
                        "bytes not available; the upload directory may have been "
                        "pruned. Re-ingest the original file."
                    )
                mime = chunk.mime_type or _mime_type_for_suffix(source.suffix.lower())
                payload = source.read_bytes()
                if chunk.modality == "image":
                    inputs.append([ImageItem(data=payload, mime_type=mime)])
                elif chunk.modality == "audio":
                    inputs.append([AudioItem(data=payload, mime_type=mime)])
                else:  # video
                    inputs.append([VideoItem(data=payload, mime_type=mime)])
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

    def _delete_committed_state(
        self,
        source_path: str,
        doc_id: str,
        kb_id: str,
    ) -> None:
        if self.hybrid_retriever:
            self.hybrid_retriever.remove_by_source(
                source_path, kb_id=kb_id
            )
        self.repository.delete_by_source(source_path, kb_id=kb_id)
        self._cleanup_parsed_artifacts(
            source_path,
            doc_id,
            kb_id,
            remove_active_doc=True,
        )
        if self.wiki_compiler:
            self.wiki_compiler.remove_document(doc_id)


    def _write_parsed_artifact(
        self,
        document: Document,
        chunks,
        structured_document: StructuredDocument | None = None,
    ) -> None:
        artifact = {
            "document": document.model_dump(),
            "chunks": [chunk.model_dump() for chunk in chunks],
            "structured_document": (
                structured_document.model_dump() if structured_document else None
            ),
        }
        target = self.config.parsed_dir / f"{document.doc_id}.json"
        target.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
        )
