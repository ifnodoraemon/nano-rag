from __future__ import annotations

import math
import re
import threading
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.retrieval.filters import match_metadata_filters
from app.schemas.chunk import Chunk
from app.schemas.document import Document
from app.vectorstore.collections import CHUNKS_COLLECTION
from app.vectorstore.milvus_client import create_milvus_client

if TYPE_CHECKING:
    from app.core.config import AppConfig


def _escape_milvus_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\")
    escaped = escaped.replace('"', '\\"')
    escaped = escaped.replace("'", "\\'")
    escaped = re.sub(r"[\x00-\x1f]", lambda m: f"\\x{ord(m.group()):02x}", escaped)
    return escaped


def _tenant_matches(actual: object, wanted: str | None) -> bool:
    actual_normalized = actual if actual not in ("", "null") else None
    wanted_normalized = wanted if wanted not in ("", "null") else None
    return actual_normalized == wanted_normalized


@dataclass
class SearchHit:
    chunk: Chunk
    score: float


class VectorRepository(ABC):
    @abstractmethod
    def upsert(
        self, document: Document, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete_by_source(
        self, source_path: str, kb_id: str, tenant_id: str | None = None
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        vector: list[float],
        top_k: int,
        kb_id: str = "default",
        tenant_id: str | None = None,
        metadata_filters: dict[str, object] | None = None,
    ) -> list[SearchHit]:
        raise NotImplementedError

    @abstractmethod
    def stats(self) -> dict[str, object]:
        raise NotImplementedError


class InMemoryVectorRepository(VectorRepository):
    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.entries: list[tuple[Chunk, list[float]]] = []
        self._lock = threading.Lock()

    def upsert(
        self, document: Document, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        with self._lock:
            self.documents[document.doc_id] = document
            self.entries.extend(zip(chunks, embeddings, strict=True))

    def delete_by_source(
        self, source_path: str, kb_id: str, tenant_id: str | None = None
    ) -> None:
        with self._lock:
            removed_doc_ids = {
                doc_id
                for doc_id, document in self.documents.items()
                if document.source_path == source_path
                and document.metadata.get("kb_id", "default") == kb_id
                and _tenant_matches(document.metadata.get("tenant_id"), tenant_id)
            }
            if removed_doc_ids:
                self.documents = {
                    doc_id: document
                    for doc_id, document in self.documents.items()
                    if doc_id not in removed_doc_ids
                }
            self.entries = [
                (chunk, embedding)
                for chunk, embedding in self.entries
                if not (
                    chunk.source_path == source_path
                    and chunk.metadata.get("kb_id", "default") == kb_id
                    and _tenant_matches(chunk.metadata.get("tenant_id"), tenant_id)
                )
            ]

    def search(
        self,
        vector: list[float],
        top_k: int,
        kb_id: str = "default",
        tenant_id: str | None = None,
        metadata_filters: dict[str, object] | None = None,
    ) -> list[SearchHit]:
        with self._lock:
            entries_snapshot = list(self.entries)
        scored = [
            SearchHit(chunk=chunk, score=_cosine_similarity(vector, embedding))
            for chunk, embedding in entries_snapshot
            if chunk.metadata.get("kb_id", "default") == kb_id
            and _tenant_matches(chunk.metadata.get("tenant_id"), tenant_id)
            and match_metadata_filters(chunk.metadata, metadata_filters)
        ]
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def stats(self) -> dict[str, object]:
        with self._lock:
            return {
                "backend": "memory",
                "documents": len(self.documents),
                "chunks": len(self.entries),
            }


class MilvusVectorRepository(VectorRepository):
    def __init__(self, dimension: int = 1536) -> None:
        self.client = create_milvus_client()
        self.dimension = dimension
        self._ensure_collection()

    @classmethod
    def from_config(cls, config: AppConfig) -> "MilvusVectorRepository":
        dimension = int(config.models.get("embedding", {}).get("dimension", 1536))
        return cls(dimension=dimension)

    def _ensure_collection(self) -> None:
        try:
            from pymilvus import DataType
        except ImportError as exc:
            raise RuntimeError(
                "pymilvus is required when VECTORSTORE_BACKEND=milvus. "
                "Install pymilvus before starting the service."
            ) from exc
        if self.client.has_collection(CHUNKS_COLLECTION):
            collection = self.client.describe_collection(
                collection_name=CHUNKS_COLLECTION
            )
            fields = collection.get("fields", [])
            vector_field = next(
                (
                    field
                    for field in fields
                    if field.get("name") == "vector"
                ),
                None,
            )
            current_dim = int(
                (vector_field or {}).get("params", {}).get("dim", self.dimension)
            )
            if current_dim == self.dimension:
                field_names = {str(field.get("name")) for field in fields}
                missing_hybrid_fields = {TEXT_FIELD, SPARSE_FIELD} - field_names
                if not missing_hybrid_fields:
                    return
                missing = ", ".join(sorted(missing_hybrid_fields))
                raise RuntimeError(
                    f"Milvus collection '{CHUNKS_COLLECTION}' is missing native hybrid field(s): {missing}. "
                    "Create a new collection or run a migration that adds a VARCHAR text field with analyzer, "
                    "a SPARSE_FLOAT_VECTOR field, and a BM25 function."
                )
            raise RuntimeError(
                f"Milvus collection '{CHUNKS_COLLECTION}' dimension mismatch: existing={current_dim}, "
                f"configured={self.dimension}. Refusing to drop the collection automatically; "
                "migrate or recreate it explicitly."
            )
        from pymilvus import Function, FunctionType

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field(
            field_name="chunk_id",
            datatype=DataType.VARCHAR,
            is_primary=True,
            max_length=256,
        )
        schema.add_field(
            field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dimension
        )
        schema.add_field(
            field_name=TEXT_FIELD,
            datatype=DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
        )
        schema.add_field(field_name=SPARSE_FIELD, datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(
            Function(
                name="text_bm25_emb",
                input_field_names=[TEXT_FIELD],
                output_field_names=[SPARSE_FIELD],
                function_type=FunctionType.BM25,
            )
        )
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector", index_type="AUTOINDEX", metric_type="COSINE"
        )
        index_params.add_index(
            field_name=SPARSE_FIELD,
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )
        self.client.create_collection(
            collection_name=CHUNKS_COLLECTION,
            schema=schema,
            index_params=index_params,
        )

    def upsert(
        self, document: Document, chunks: list[Chunk], embeddings: list[list[float]]
    ) -> None:
        rows: list[dict[str, Any]] = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            rows.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "vector": embedding,
                    "doc_id": document.doc_id,
                    "source": chunk.source_path,
                    "title": chunk.title or "",
                    "chunk_index": chunk.chunk_index,
                    TEXT_FIELD: chunk.text,
                    "metadata_json": chunk.metadata,
                    "kb_id": document.metadata.get("kb_id", "default"),
                    "tenant_id": document.metadata.get("tenant_id") or "",
                    "modality": chunk.modality,
                    "media_uri": chunk.media_uri or "",
                    "mime_type": chunk.mime_type or "",
                }
            )
        self.client.upsert(collection_name=CHUNKS_COLLECTION, data=rows)

    def delete_by_source(
        self, source_path: str, kb_id: str, tenant_id: str | None = None
    ) -> None:
        escaped = _escape_milvus_string(source_path)
        escaped_kb_id = _escape_milvus_string(kb_id)
        base_filter = f'source == "{escaped}" and kb_id == "{escaped_kb_id}"'
        tenant = _escape_milvus_string(tenant_id or "")
        base_filter += f' and tenant_id == "{tenant}"'
        self.client.delete(
            collection_name=CHUNKS_COLLECTION,
            filter=base_filter,
        )

    def search(
        self,
        vector: list[float],
        top_k: int,
        kb_id: str = "default",
        tenant_id: str | None = None,
        metadata_filters: dict[str, object] | None = None,
    ) -> list[SearchHit]:
        escaped_kb_id = _escape_milvus_string(kb_id)
        base_filter = f'kb_id == "{escaped_kb_id}"'
        tenant = _escape_milvus_string(tenant_id or "")
        base_filter += f' and tenant_id == "{tenant}"'
        results = self.client.search(
            collection_name=CHUNKS_COLLECTION,
            data=[vector],
            anns_field="vector",
            filter=base_filter,
            limit=max(top_k * 4, 20),
            output_fields=[
                "chunk_id",
                "doc_id",
                "source",
                "title",
                "chunk_index",
                "text",
                "metadata_json",
                "modality",
                "media_uri",
                "mime_type",
            ],
        )
        hits: list[SearchHit] = []
        for item in results[0]:
            entity = item["entity"]
            metadata = entity.get("metadata_json") or {}
            if not match_metadata_filters(metadata, metadata_filters):
                continue
            modality = str(entity.get("modality") or metadata.get("modality") or "text")
            if modality not in ("text", "image", "audio", "video"):
                modality = "text"
            media_uri = entity.get("media_uri") or metadata.get("media_uri") or None
            mime_type = entity.get("mime_type") or metadata.get("mime_type") or None
            hits.append(
                SearchHit(
                    chunk=Chunk(
                        chunk_id=entity["chunk_id"],
                        doc_id=entity["doc_id"],
                        chunk_index=entity["chunk_index"],
                        text=entity["text"],
                        source_path=entity["source"],
                        title=entity.get("title"),
                        metadata=metadata,
                        modality=modality,
                        media_uri=media_uri or None,
                        mime_type=mime_type or None,
                    ),
                    score=float(item["distance"]),
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    def native_hybrid_search(
        self,
        vector: list[float],
        query: str,
        top_k: int,
        kb_id: str = "default",
        tenant_id: str | None = None,
        metadata_filters: dict[str, object] | None = None,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
    ) -> list[SearchHit]:
        from pymilvus import AnnSearchRequest, WeightedRanker

        escaped_kb_id = _escape_milvus_string(kb_id)
        base_filter = f'kb_id == "{escaped_kb_id}"'
        tenant = _escape_milvus_string(tenant_id or "")
        base_filter += f' and tenant_id == "{tenant}"'
        request_limit = max(top_k * 4, 20)
        dense_req = AnnSearchRequest(
            data=[vector],
            anns_field="vector",
            param={"metric_type": "COSINE"},
            limit=request_limit,
            expr=base_filter,
        )
        sparse_req = AnnSearchRequest(
            data=[query],
            anns_field=SPARSE_FIELD,
            param={"metric_type": "BM25"},
            limit=request_limit,
            expr=base_filter,
        )
        results = self.client.hybrid_search(
            collection_name=CHUNKS_COLLECTION,
            reqs=[dense_req, sparse_req],
            ranker=WeightedRanker(dense_weight, sparse_weight),
            limit=request_limit,
            output_fields=[
                "chunk_id",
                "doc_id",
                "source",
                "title",
                "chunk_index",
                TEXT_FIELD,
                "metadata_json",
                "modality",
                "media_uri",
                "mime_type",
            ],
        )
        return self._build_hits(results, top_k, metadata_filters)

    def _build_hits(
        self,
        results: list[list[dict[str, Any]]],
        top_k: int,
        metadata_filters: dict[str, object] | None = None,
    ) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for item in results[0]:
            entity = item["entity"]
            metadata = entity.get("metadata_json") or {}
            if not match_metadata_filters(metadata, metadata_filters):
                continue
            modality = str(entity.get("modality") or metadata.get("modality") or "text")
            if modality not in ("text", "image", "audio", "video"):
                modality = "text"
            media_uri = entity.get("media_uri") or metadata.get("media_uri") or None
            mime_type = entity.get("mime_type") or metadata.get("mime_type") or None
            hits.append(
                SearchHit(
                    chunk=Chunk(
                        chunk_id=entity["chunk_id"],
                        doc_id=entity["doc_id"],
                        chunk_index=entity["chunk_index"],
                        text=entity[TEXT_FIELD],
                        source_path=entity["source"],
                        title=entity.get("title"),
                        metadata=metadata,
                        modality=modality,
                        media_uri=media_uri or None,
                        mime_type=mime_type or None,
                    ),
                    score=float(item["distance"]),
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    def stats(self) -> dict[str, object]:
        collection = self.client.describe_collection(collection_name=CHUNKS_COLLECTION)
        return {
            "backend": "milvus",
            "collection_name": CHUNKS_COLLECTION,
            "dimension": self.dimension,
            "collection": _json_safe(collection),
        }

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass


def _cosine_similarity(lhs: list[float], rhs: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(lhs, rhs, strict=True))
    lhs_norm = math.sqrt(sum(value * value for value in lhs))
    rhs_norm = math.sqrt(sum(value * value for value in rhs))
    if lhs_norm == 0 or rhs_norm == 0:
        return 0.0
    return numerator / (lhs_norm * rhs_norm)


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "__iter__"):
        try:
            return [_json_safe(item) for item in value]  # type: ignore[operator]
        except TypeError:
            pass
    return str(value)


TEXT_FIELD = "text"
SPARSE_FIELD = "sparse"
