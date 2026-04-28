from __future__ import annotations

import asyncio
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from app.retrieval.bm25 import BM25Index
from app.retrieval.filters import match_metadata_filters
from app.retrieval.hybrid_fusion import (
    HybridSearchConfig,
    reciprocal_rank_fusion,
)
from app.schemas.chunk import Chunk
from app.vectorstore.repository import SearchHit, VectorRepository

if TYPE_CHECKING:
    from app.model_client.embeddings import EmbeddingClient


@dataclass
class HybridSearchResult:
    chunk: Chunk
    vector_score: float
    bm25_score: float
    hybrid_score: float


@dataclass
class HybridRetriever:
    repository: VectorRepository
    embedding_client: EmbeddingClient
    bm25_index: BM25Index = field(default_factory=lambda: BM25Index())
    hybrid_config: HybridSearchConfig = field(
        default_factory=HybridSearchConfig.from_env
    )
    _chunk_cache: dict[str, Chunk] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _bm25_enabled: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._bm25_enabled = os.getenv(
            "RAG_HYBRID_SEARCH_ENABLED", "false"
        ).lower() in ("true", "1", "yes")

    @property
    def _native_hybrid_available(self) -> bool:
        return callable(getattr(self.repository, "native_hybrid_search", None))

    @property
    def enabled(self) -> bool:
        return self._bm25_enabled

    def index_chunk(self, chunk: Chunk) -> None:
        if not self._bm25_enabled or self._native_hybrid_available:
            return
        with self._lock:
            self._chunk_cache[chunk.chunk_id] = chunk
            self.bm25_index.add_document(chunk.chunk_id, chunk.text)

    def index_chunks(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self.index_chunk(chunk)

    def remove_chunk(self, chunk_id: str) -> None:
        if not self._bm25_enabled or self._native_hybrid_available:
            return
        with self._lock:
            self._chunk_cache.pop(chunk_id, None)
            self.bm25_index.remove_document(chunk_id)

    def remove_by_source(
        self, source_path: str, kb_id: str, tenant_id: str | None = None
    ) -> None:
        if not self._bm25_enabled or self._native_hybrid_available:
            return
        with self._lock:
            removable_ids = [
                chunk_id
                for chunk_id, chunk in self._chunk_cache.items()
                if chunk.source_path == source_path
                and chunk.metadata.get("kb_id", "default") == kb_id
                and (tenant_id is None or chunk.metadata.get("tenant_id") == tenant_id)
            ]
            for chunk_id in removable_ids:
                self._chunk_cache.pop(chunk_id, None)
                self.bm25_index.remove_document(chunk_id)

    def bootstrap_from_parsed_dir(self, parsed_dir: Path) -> int:
        if (
            not self._bm25_enabled
            or self._native_hybrid_available
            or not parsed_dir.exists()
        ):
            return 0
        count = 0
        for artifact in parsed_dir.glob("*.json"):
            try:
                payload = json.loads(artifact.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            raw_chunks = payload.get("chunks", []) if isinstance(payload, dict) else []
            if not isinstance(raw_chunks, list):
                continue
            for raw_chunk in raw_chunks:
                if not isinstance(raw_chunk, dict):
                    continue
                self.index_chunk(Chunk.model_validate(raw_chunk))
                count += 1
        return count

    def clear_index(self) -> None:
        with self._lock:
            self._chunk_cache.clear()
            self.bm25_index.clear()

    async def retrieve(
        self,
        query: str,
        top_k: int,
        kb_id: str = "default",
        tenant_id: str | None = None,
        metadata_filters: dict[str, object] | None = None,
    ) -> list[SearchHit]:
        if not self._bm25_enabled:
            vectors = await self.embedding_client.embed_texts([query])
            if not vectors:
                return []
            return await asyncio.to_thread(
                self.repository.search,
                vectors[0],
                top_k,
                kb_id=kb_id,
                tenant_id=tenant_id,
                metadata_filters=metadata_filters,
            )

        vectors = await self.embedding_client.embed_texts([query])
        if not vectors:
            return []
        native_hybrid_search = getattr(self.repository, "native_hybrid_search", None)
        if callable(native_hybrid_search):
            return await asyncio.to_thread(
                native_hybrid_search,
                vectors[0],
                query,
                top_k,
                kb_id=kb_id,
                tenant_id=tenant_id,
                metadata_filters=metadata_filters,
                dense_weight=self.hybrid_config.vector_weight,
                sparse_weight=self.hybrid_config.bm25_weight,
            )
        vector_hits = await asyncio.to_thread(
            self.repository.search,
            vectors[0],
            top_k * 2,
            kb_id=kb_id,
            tenant_id=tenant_id,
            metadata_filters=metadata_filters,
        )
        vector_results = [(hit.chunk.chunk_id, hit.score) for hit in vector_hits]
        with self._lock:
            chunk_cache_snapshot = dict(self._chunk_cache)
        scoped_chunk_ids = {
            chunk_id
            for chunk_id, chunk in chunk_cache_snapshot.items()
            if chunk.metadata.get("kb_id", "default") == kb_id
            and (tenant_id is None or chunk.metadata.get("tenant_id") == tenant_id)
            and match_metadata_filters(chunk.metadata, metadata_filters)
        }
        bm25_results = self.bm25_index.search(
            query, top_k * 2, allowed_doc_ids=scoped_chunk_ids
        )
        fused_results = reciprocal_rank_fusion(
            vector_results, bm25_results, self.hybrid_config
        )
        fused_results = fused_results[:top_k]
        hits_by_id = {hit.chunk.chunk_id: hit for hit in vector_hits}
        final_hits: list[SearchHit] = []
        for doc_id, hybrid_score in fused_results:
            if doc_id in hits_by_id:
                hit = hits_by_id[doc_id]
                final_hits.append(hit)
            elif doc_id in chunk_cache_snapshot:
                chunk = chunk_cache_snapshot[doc_id]
                final_hits.append(
                    SearchHit(
                        chunk=chunk,
                        score=hybrid_score,
                    )
                )
        return final_hits
