from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

from app.core.tracing import TraceSession, TraceStore
from app.model_client.embeddings import EmbeddingClient
from app.model_client.rerank import RerankClient
from app.retrieval.context_builder import build_contexts
from app.retrieval.freshness import FreshnessPolicy, prioritize_fresh_hits
from app.retrieval.filters import (
    infer_metadata_filters,
    merge_metadata_filters,
    sanitize_metadata_filters,
)
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.reranker import RetrievalReranker
from app.retrieval.retriever import Retriever
from app.schemas.trace import RetrievalDebugResponse
from app.vectorstore.repository import VectorRepository
from app.wiki.search import WikiSearcher

if TYPE_CHECKING:
    from app.core.config import AppConfig
    from app.core.tracing import TracingManager


class RetrievalPipeline:
    def __init__(
        self,
        config: AppConfig,
        repository: VectorRepository,
        embedding_client: EmbeddingClient,
        rerank_client: RerankClient,
        trace_store: TraceStore,
        tracing_manager: TracingManager,
        query_rewriter: QueryRewriter | None = None,
        hybrid_retriever: HybridRetriever | None = None,
        wiki_searcher: WikiSearcher | None = None,
    ) -> None:
        self.config = config
        self.retriever = Retriever(
            repository,
            embedding_client,
            query_rewriter,
            hybrid_retriever=hybrid_retriever,
            wiki_searcher=wiki_searcher,
        )
        self.reranker = RetrievalReranker(
            rerank_client,
            metadata_weights=config.settings["retrieval"].get("metadata_rerank"),
        )
        self.trace_store = trace_store
        self.tracing_manager = tracing_manager

    async def run(
        self,
        query: str,
        top_k: int | None = None,
        kb_id: str = "default",
        tenant_id: str | None = None,
        session_id: str | None = None,
        sample_id: str | None = None,
        metadata_filters: dict[str, object] | None = None,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        requested_top_k = top_k or self.config.settings["retrieval"]["top_k"]
        rerank_top_k = self.config.settings["retrieval"]["rerank_top_k"]
        final_contexts_limit = self.config.settings["retrieval"]["final_contexts"]
        context_quotas = self.config.settings["retrieval"].get(
            "context_quota",
            {"topic": 2, "raw": 3, "source": 1, "index": 1},
        )
        metadata_rerank = self.config.settings["retrieval"].get("metadata_rerank", {})
        freshness_settings = self.config.settings["retrieval"].get("freshness_policy", {})
        freshness_policy = FreshnessPolicy(
            enabled=bool(freshness_settings.get("enabled", True)),
            allow_stale_fallback=bool(
                freshness_settings.get("allow_stale_fallback", False)
            ),
        )
        inferred_metadata_filters = infer_metadata_filters(query)
        effective_metadata_filters = merge_metadata_filters(
            metadata_filters,
            inferred_metadata_filters,
        )
        public_metadata_filters = sanitize_metadata_filters(effective_metadata_filters)
        public_inferred_filters = sanitize_metadata_filters(inferred_metadata_filters)
        with self.tracing_manager.span(
            "retrieval.run",
            {
                "retrieval.query": query,
                "retrieval.top_k": requested_top_k,
                "retrieval.kb_id": kb_id,
                "retrieval.tenant_id": tenant_id or "",
                "retrieval.session_id": session_id or "",
            },
        ):
            trace = TraceSession()
            retrieval_started = perf_counter()
            retrieval_result = await self.retriever.retrieve(
                query,
                requested_top_k,
                kb_id=kb_id,
                tenant_id=tenant_id,
                metadata_filters=effective_metadata_filters,
            )
            retrieved = retrieval_result.hits
            retrieval_seconds = round(perf_counter() - retrieval_started, 4)
            rerank_seconds = 0.0
            if self.config.rerank_enabled:
                rerank_started = perf_counter()
                reranked = await self.reranker.rerank(query, retrieved, rerank_top_k)
                rerank_seconds = round(perf_counter() - rerank_started, 4)
            else:
                reranked = retrieved[:rerank_top_k]
            freshness_ranked = prioritize_fresh_hits(reranked, freshness_policy)
            retrieved_contexts = build_contexts(retrieved, requested_top_k)
            reranked_contexts = build_contexts(reranked, rerank_top_k)
            contexts = build_contexts(
                freshness_ranked,
                final_contexts_limit,
                quotas=context_quotas,
            )
            trace.record("query", query)
            trace.record("kb_id", kb_id)
            trace.record("tenant_id", tenant_id)
            trace.record("session_id", session_id)
            trace.record("sample_id", sample_id)
            trace.record("metadata_filters", public_metadata_filters)
            trace.record("rewritten_query", retrieval_result.query_plan.rewritten_query)
            trace.record(
                "expanded_queries", retrieval_result.query_plan.retrieval_queries
            )
            trace.record("hyde_query", retrieval_result.query_plan.hyde_query)
            trace.record(
                "retrieved_chunk_ids", [hit.chunk.chunk_id for hit in retrieved]
            )
            trace.record("reranked_chunk_ids", [hit.chunk.chunk_id for hit in reranked])
            trace.record(
                "freshness_ranked_chunk_ids",
                [hit.chunk.chunk_id for hit in freshness_ranked],
            )
            trace.record("retrieved", retrieved_contexts)
            trace.record("reranked", reranked_contexts)
            trace.record(
                "freshness_ranked",
                build_contexts(freshness_ranked, rerank_top_k),
            )
            trace.record("contexts", contexts)
            trace.record(
                "retrieval_params",
                {
                    "requested_top_k": requested_top_k,
                    "rerank_top_k": rerank_top_k,
                    "final_contexts": final_contexts_limit,
                    "rerank_enabled": self.config.rerank_enabled,
                    "context_quota": context_quotas,
                    "metadata_rerank": metadata_rerank,
                    "freshness_policy": freshness_settings or None,
                    "metadata_filters": public_metadata_filters,
                    "inferred_metadata_filters": public_inferred_filters,
                    "rewritten_query": retrieval_result.query_plan.rewritten_query,
                    "expanded_queries": retrieval_result.query_plan.retrieval_queries,
                    "hyde_query": retrieval_result.query_plan.hyde_query,
                },
            )
            trace.record(
                "step_latencies",
                {
                    "retrieval_seconds": retrieval_seconds,
                    "rerank_seconds": rerank_seconds,
                },
            )
            trace.record(
                "embedding_model_alias",
                getattr(self.retriever.embedding_client, "alias", None),
            )
            if self.config.rerank_enabled:
                trace.record(
                    "rerank_model_alias", getattr(self.reranker.client, "alias", None)
                )
            final_trace = trace.finish()
            self.trace_store.save_raw(final_trace)
            return contexts, final_trace

    async def debug(
        self,
        query: str,
        top_k: int | None = None,
        kb_id: str = "default",
        tenant_id: str | None = None,
        session_id: str | None = None,
        metadata_filters: dict[str, object] | None = None,
    ) -> RetrievalDebugResponse:
        contexts, trace = await self.run(
            query,
            top_k,
            kb_id=kb_id,
            tenant_id=tenant_id,
            session_id=session_id,
            metadata_filters=metadata_filters,
        )
        record = self.trace_store.get(trace["trace_id"])
        if record is None:
            raise RuntimeError(f"trace not found: {trace['trace_id']}")
        return RetrievalDebugResponse(
            query=query,
            retrieved=record.retrieved,
            reranked=record.reranked,
            contexts=contexts,
            trace_id=trace["trace_id"],
        )
