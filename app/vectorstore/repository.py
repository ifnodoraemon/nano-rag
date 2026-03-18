from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.schemas.chunk import Chunk
from app.schemas.document import Document
from app.vectorstore.collections import CHUNKS_COLLECTION
from app.vectorstore.milvus_client import create_milvus_client

if TYPE_CHECKING:
    from app.core.config import AppConfig


@dataclass
class SearchHit:
    chunk: Chunk
    score: float


class VectorRepository(ABC):
    @abstractmethod
    def upsert(self, document: Document, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, vector: list[float], top_k: int) -> list[SearchHit]:
        raise NotImplementedError


class InMemoryVectorRepository(VectorRepository):
    def __init__(self) -> None:
        self.documents: dict[str, Document] = {}
        self.entries: list[tuple[Chunk, list[float]]] = []

    def upsert(self, document: Document, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self.documents[document.doc_id] = document
        self.entries.extend(zip(chunks, embeddings, strict=True))

    def search(self, vector: list[float], top_k: int) -> list[SearchHit]:
        scored = [
            SearchHit(chunk=chunk, score=_cosine_similarity(vector, embedding))
            for chunk, embedding in self.entries
        ]
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]


class MilvusVectorRepository(VectorRepository):
    def __init__(self, dimension: int = 1024) -> None:
        self.client = create_milvus_client()
        self.dimension = dimension
        self._ensure_collection()

    @classmethod
    def from_config(cls, config: AppConfig) -> "MilvusVectorRepository":
        dimension = int(config.models.get("embedding", {}).get("dimension", 1024))
        return cls(dimension=dimension)

    def _ensure_collection(self) -> None:
        if self.client.has_collection(CHUNKS_COLLECTION):
            return
        self.client.create_collection(
            collection_name=CHUNKS_COLLECTION,
            dimension=self.dimension,
            auto_id=False,
            primary_field_name="chunk_id",
            id_type="string",
        )

    def upsert(self, document: Document, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
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
                    "text": chunk.text,
                    "metadata_json": chunk.metadata,
                }
            )
        self.client.upsert(collection_name=CHUNKS_COLLECTION, data=rows)

    def search(self, vector: list[float], top_k: int) -> list[SearchHit]:
        results = self.client.search(
            collection_name=CHUNKS_COLLECTION,
            data=[vector],
            limit=top_k,
            output_fields=["chunk_id", "doc_id", "source", "title", "chunk_index", "text", "metadata_json"],
        )
        hits: list[SearchHit] = []
        for item in results[0]:
            entity = item["entity"]
            hits.append(
                SearchHit(
                    chunk=Chunk(
                        chunk_id=entity["chunk_id"],
                        doc_id=entity["doc_id"],
                        chunk_index=entity["chunk_index"],
                        text=entity["text"],
                        source_path=entity["source"],
                        title=entity.get("title"),
                        metadata=entity.get("metadata_json") or {},
                    ),
                    score=float(item["distance"]),
                )
            )
        return hits


def _cosine_similarity(lhs: list[float], rhs: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(lhs, rhs, strict=True))
    lhs_norm = math.sqrt(sum(value * value for value in lhs))
    rhs_norm = math.sqrt(sum(value * value for value in rhs))
    if lhs_norm == 0 or rhs_norm == 0:
        return 0.0
    return numerator / (lhs_norm * rhs_norm)
