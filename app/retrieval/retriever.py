from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.core.exceptions import ModelGatewayError
from app.model_client.embeddings import EmbeddingClient
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.query_rewriter import QueryExpansionPlan, QueryRewriter
from app.vectorstore.repository import SearchHit, VectorRepository
from app.wiki.search import WikiSearcher

QUERY_VARIANT_RRF_K = 60


@dataclass
class RetrievalResult:
    hits: list[SearchHit]
    query_plan: QueryExpansionPlan


class Retriever:
    def __init__(
        self,
        repository: VectorRepository,
        embedding_client: EmbeddingClient,
        query_rewriter: QueryRewriter | None = None,
        hybrid_retriever: HybridRetriever | None = None,
        wiki_searcher: WikiSearcher | None = None,
    ) -> None:
        self.repository = repository
        self.embedding_client = embedding_client
        self.query_rewriter = query_rewriter
        self.hybrid_retriever = hybrid_retriever
        self.wiki_searcher = wiki_searcher

    async def retrieve(
        self,
        query: str,
        top_k: int,
        kb_id: str = "default",
        metadata_filters: dict[str, object] | None = None,
    ) -> RetrievalResult:
        query_plan = QueryExpansionPlan(rewritten_query=None, retrieval_queries=[query])
        if self.query_rewriter:
            query_plan = await self.query_rewriter.build_plan(query)

        result_sets: list[list[SearchHit]] = []
        for retrieval_query in query_plan.retrieval_queries:
            result_sets.append(
                await self._search_query(
                    retrieval_query,
                    top_k,
                    kb_id=kb_id,
                    metadata_filters=metadata_filters,
                )
            )
        if query_plan.hyde_query:
            result_sets.append(
                await self._search_query(
                    query_plan.hyde_query,
                    top_k,
                    kb_id=kb_id,
                    metadata_filters=metadata_filters,
                )
            )
        fused_hits = self._fuse_result_sets(result_sets, top_k)
        return RetrievalResult(hits=fused_hits, query_plan=query_plan)

    async def _search_query(
        self,
        query: str,
        top_k: int,
        kb_id: str = "default",
        metadata_filters: dict[str, object] | None = None,
    ) -> list[SearchHit]:
        wiki_hits: list[SearchHit] = []
        if self.wiki_searcher and self.wiki_searcher.enabled:
            wiki_hits = self.wiki_searcher.search(
                query,
                top_k=top_k,
                kb_id=kb_id,
                metadata_filters=metadata_filters,
            )
            if len(wiki_hits) >= top_k:
                return wiki_hits[:top_k]
        if self.hybrid_retriever and self.hybrid_retriever.enabled:
            raw_hits = await self.hybrid_retriever.retrieve(
                query,
                top_k,
                kb_id=kb_id,
                metadata_filters=metadata_filters,
            )
        else:
            vectors = await self.embedding_client.embed_texts([query])
            if not vectors:
                raise ModelGatewayError("embedding service returned empty result")
            raw_hits = await asyncio.to_thread(
                self.repository.search,
                vectors[0],
                top_k,
                kb_id=kb_id,
                metadata_filters=metadata_filters,
            )
        if not wiki_hits:
            return raw_hits
        merged_hits = list(wiki_hits)
        seen_chunk_ids = {hit.chunk.chunk_id for hit in wiki_hits}
        for hit in raw_hits:
            if hit.chunk.chunk_id in seen_chunk_ids:
                continue
            merged_hits.append(hit)
            seen_chunk_ids.add(hit.chunk.chunk_id)
            if len(merged_hits) >= top_k:
                break
        return merged_hits[:top_k]

    def _fuse_result_sets(
        self, result_sets: list[list[SearchHit]], top_k: int
    ) -> list[SearchHit]:
        if not result_sets:
            return []
        if len(result_sets) == 1:
            return result_sets[0][:top_k]

        fused_scores: dict[str, float] = {}
        chunk_by_id: dict[str, SearchHit] = {}
        for hits in result_sets:
            for rank, hit in enumerate(hits, start=1):
                fused_scores[hit.chunk.chunk_id] = fused_scores.get(
                    hit.chunk.chunk_id, 0.0
                ) + 1.0 / (QUERY_VARIANT_RRF_K + rank)
                chunk_by_id.setdefault(hit.chunk.chunk_id, hit)
        ranked = sorted(
            fused_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        return [
            SearchHit(chunk=chunk_by_id[chunk_id].chunk, score=score)
            for chunk_id, score in ranked[:top_k]
        ]
