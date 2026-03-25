from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.retrieval.bm25 import BM25Config, BM25Index
from app.retrieval.hybrid_fusion import (
    HybridSearchConfig,
    reciprocal_rank_fusion,
    weighted_score_fusion,
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
        object.__setattr__(
            self,
            "_bm25_enabled",
            os.getenv("RAG_HYBRID_SEARCH_ENABLED", "false").lower()
            in ("true", "1", "yes"),
        )

    def index_chunk(self, chunk: Chunk) -> None:
        if not self._bm25_enabled:
            return
        with self._lock:
            self._chunk_cache[chunk.chunk_id] = chunk
            self.bm25_index.add_document(chunk.chunk_id, chunk.text)

    def remove_chunk(self, chunk_id: str) -> None:
        if not self._bm25_enabled:
            return
        with self._lock:
            self._chunk_cache.pop(chunk_id, None)
            self.bm25_index.remove_document(chunk_id)

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
    ) -> list[SearchHit]:
        if not self._bm25_enabled:
            vectors = await self.embedding_client.embed_texts([query])
            if not vectors:
                return []
            return self.repository.search(
                vectors[0], top_k, kb_id=kb_id, tenant_id=tenant_id
            )

        vectors = await self.embedding_client.embed_texts([query])
        if not vectors:
            return []
        vector_hits = self.repository.search(
            vectors[0], top_k * 2, kb_id=kb_id, tenant_id=tenant_id
        )
        vector_results = [(hit.chunk.chunk_id, hit.score) for hit in vector_hits]
        bm25_results = self.bm25_index.search(query, top_k * 2)
        fused_results = reciprocal_rank_fusion(
            vector_results, bm25_results, self.hybrid_config
        )
        fused_results = fused_results[:top_k]
        fused_doc_ids = {doc_id for doc_id, _ in fused_results}
        hits_by_id = {hit.chunk.chunk_id: hit for hit in vector_hits}
        bm25_scores = dict(bm25_results)
        final_hits: list[SearchHit] = []
        for doc_id, hybrid_score in fused_results:
            if doc_id in hits_by_id:
                hit = hits_by_id[doc_id]
                final_hits.append(hit)
            elif doc_id in self._chunk_cache:
                chunk = self._chunk_cache[doc_id]
                final_hits.append(
                    SearchHit(
                        chunk=chunk,
                        score=hybrid_score,
                    )
                )
        return final_hits
