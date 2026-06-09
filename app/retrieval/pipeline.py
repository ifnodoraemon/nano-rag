from __future__ import annotations

import logging
from time import perf_counter
from typing import TYPE_CHECKING

from app.core.tracing import TraceSession, TraceStore
from app.model_client.embeddings import EmbeddingClient
from app.model_client.rerank import RerankClient
from app.retrieval.context_builder import build_contexts
from app.retrieval.evidence_planner import EvidencePlanner
from app.retrieval.freshness import FreshnessPolicy, prioritize_fresh_hits
from app.retrieval.filters import (
    infer_metadata_filters,
    merge_metadata_filters,
    sanitize_metadata_filters,
)
from app.retrieval.hybrid_retriever import HybridRetriever

from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.query_router import QueryRoute, QueryRouter
from app.retrieval.reranker import RetrievalReranker
from app.retrieval.retriever import Retriever
from app.schemas.trace import RetrievalDebugResponse
from app.utils.text import parse_bool_env
from app.vectorstore.repository import SearchHit, VectorRepository
from app.wiki.search import WikiSearcher

if TYPE_CHECKING:
    from app.core.config import AppConfig
    from app.core.tracing import TracingManager

logger = logging.getLogger(__name__)


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
        query_router: QueryRouter | None = None,
        evidence_planner: EvidencePlanner | None = None,
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
        self.query_router = query_router or QueryRouter()
        self.evidence_planner = evidence_planner or EvidencePlanner()


    async def run(
        self,
        query: str,
        top_k: int | None = None,
        kb_id: str = "default",
        session_id: str | None = None,
        sample_id: str | None = None,
        metadata_filters: dict[str, object] | None = None,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        requested_top_k = top_k or self.config.settings["retrieval"]["top_k"]
        rerank_top_k = self.config.settings["retrieval"]["rerank_top_k"]
        final_contexts_limit = self.config.settings["retrieval"]["final_contexts"]
        max_context_text_chars = int(
            self.config.settings["retrieval"].get("max_context_text_chars", 6000)
        )
        context_quotas = self.config.settings["retrieval"].get(
            "context_quota",
            {"topic": 2, "raw": 3, "source": 1, "index": 1},
        )
        metadata_rerank = self.config.settings["retrieval"].get("metadata_rerank", {})
        freshness_settings = self.config.settings["retrieval"].get("freshness_policy", {})
        route_started = perf_counter()
        query_route = await self.query_router.route(query)
        route_seconds = round(perf_counter() - route_started, 4)
        include_stale_versions = _setting_enabled(
            freshness_settings.get("include_stale", False)
        ) or query_route.route in {"version", "conflict"}
        freshness_policy = FreshnessPolicy(
            enabled=_setting_enabled(freshness_settings.get("enabled", True)),
            include_stale=include_stale_versions,
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
                "retrieval.session_id": session_id or "",
            },
        ):
            trace = TraceSession()
            retrieval_started = perf_counter()
            retrieval_result = await self.retriever.retrieve(
                query,
                requested_top_k,
                kb_id=kb_id,
                metadata_filters=effective_metadata_filters,
            )
            retrieved = retrieval_result.hits
            retrieval_seconds = round(perf_counter() - retrieval_started, 4)
            rerank_seconds = 0.0
            rerank_error: str | None = None
            if self.config.rerank_enabled:
                rerank_started = perf_counter()
                try:
                    reranked = await self.reranker.rerank(
                        query, retrieved, rerank_top_k, query_route=query_route
                    )
                except Exception as exc:
                    logger.warning("rerank failed; falling back to retrieval order: %s", exc)
                    rerank_error = exc.__class__.__name__
                    reranked = _prioritize_by_query_route(retrieved, query_route)
                rerank_seconds = round(perf_counter() - rerank_started, 4)
            else:
                reranked = _prioritize_by_query_route(retrieved, query_route)
            freshness_ranked = prioritize_fresh_hits(reranked, freshness_policy)
            retrieved_contexts = build_contexts(
                retrieved, requested_top_k, max_text_chars=max_context_text_chars
            )
            reranked_contexts = build_contexts(
                reranked, rerank_top_k, max_text_chars=max_context_text_chars
            )
            contexts = build_contexts(
                freshness_ranked,
                final_contexts_limit,
                quotas=context_quotas,
                query=query,
                max_text_chars=max_context_text_chars,
            )
            contexts = _promote_visual_sibling_contexts(
                contexts,
                freshness_ranked,
                final_contexts_limit,
                max_context_text_chars,
                query_route,
            )
            evidence_plan = await self.evidence_planner.plan(
                query,
                contexts,
                query_route=query_route,
            )
            contexts = self.evidence_planner.annotate_contexts(contexts, evidence_plan)
            trace.record("query", query)
            trace.record("kb_id", kb_id)
            trace.record("session_id", session_id)
            trace.record("sample_id", sample_id)
            trace.record("metadata_filters", public_metadata_filters)
            trace.record("query_route", query_route.as_dict())
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
                build_contexts(
                    freshness_ranked,
                    rerank_top_k,
                    max_text_chars=max_context_text_chars,
                ),
            )
            trace.record("contexts", contexts)
            trace.record("evidence_plan", evidence_plan)
            trace.record(
                "retrieval_params",
                {
                    "requested_top_k": requested_top_k,
                    "rerank_top_k": rerank_top_k,
                    "final_contexts": final_contexts_limit,
                    "max_context_text_chars": max_context_text_chars,
                    "rerank_enabled": self.config.rerank_enabled,
                    "rerank_error": rerank_error,
                    "context_quota": context_quotas,
                    "metadata_rerank": metadata_rerank,
                    "freshness_policy": freshness_settings or None,
                    "freshness_include_stale": include_stale_versions,
                    "metadata_filters": public_metadata_filters,
                    "inferred_metadata_filters": public_inferred_filters,
                    "query_route": query_route.as_dict(),
                    "evidence_plan": evidence_plan,
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
                    "query_route_seconds": route_seconds,
                    "evidence_planner_seconds": float(
                        str(evidence_plan.get("planner_seconds", 0.0) or 0.0)
                    ),
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
        session_id: str | None = None,
        metadata_filters: dict[str, object] | None = None,
    ) -> RetrievalDebugResponse:
        contexts, trace = await self.run(
            query,
            top_k,
            kb_id=kb_id,
            session_id=session_id,
            metadata_filters=metadata_filters,
        )
        trace_id = str(trace["trace_id"])
        record = self.trace_store.get(trace_id)
        if record is None:
            raise RuntimeError(f"trace not found: {trace['trace_id']}")
        return RetrievalDebugResponse(
            query=query,
            retrieved=record.retrieved,
            reranked=record.reranked,
            contexts=contexts,
            trace_id=trace_id,
        )


def _prioritize_by_query_route(
    hits: list[SearchHit], query_route: QueryRoute
) -> list[SearchHit]:
    adjustments = [
        _query_route_adjustment(hit, query_route)
        for hit in hits
    ]
    if not any(adjustment > 0 for adjustment in adjustments):
        return hits
    adjusted = [
        SearchHit(
            chunk=hit.chunk,
            score=round(hit.score + adjustment, 6),
        )
        for hit, adjustment in zip(hits, adjustments, strict=True)
    ]
    return sorted(adjusted, key=lambda item: item.score, reverse=True)





def _promote_visual_sibling_contexts(
    contexts: list[dict[str, object]],
    ranked_hits: list[SearchHit],
    limit: int,
    max_text_chars: int,
    query_route: QueryRoute,
) -> list[dict[str, object]]:
    if query_route.route != "visual" or len(contexts) >= limit:
        return contexts
    if not any(_is_visual_context(context) for context in contexts):
        return contexts
    selected_ids = {str(context.get("chunk_id")) for context in contexts}
    candidate_contexts = build_contexts(
        ranked_hits,
        max(len(ranked_hits), limit),
        query=None,
        max_text_chars=max_text_chars,
    )
    for candidate in candidate_contexts:
        if len(contexts) >= limit:
            break
        chunk_id = str(candidate.get("chunk_id"))
        if chunk_id in selected_ids:
            continue
        if not _is_text_sibling_for_visual_contexts(candidate, contexts):
            continue
        contexts.append(candidate)
        selected_ids.add(chunk_id)
    return contexts


def _is_visual_context(context: dict[str, object]) -> bool:
    return (
        context.get("modality") == "image"
        or context.get("chunk_kind")
        in {"rendered_page_image", "embedded_image", "media_object", "document_page"}
        or context.get("chunk_strategy")
        in {"rendered_page_image", "embedded_image", "page_attachment"}
    )


def _is_text_sibling_for_visual_contexts(
    candidate: dict[str, object],
    visual_contexts: list[dict[str, object]],
) -> bool:
    if candidate.get("modality") != "text":
        return False
    source = candidate.get("source")
    page_number = candidate.get("page_number")
    for visual in visual_contexts:
        if not _is_visual_context(visual):
            continue
        if source and visual.get("source") == source:
            visual_page = visual.get("page_number")
            if visual_page is None or page_number is None or visual_page == page_number:
                return True
    return False


def _query_route_adjustment(hit: SearchHit, query_route: QueryRoute) -> float:
    metadata = hit.chunk.metadata or {}
    route = query_route.route
    preferred = set(query_route.preferred_chunk_kinds)
    chunk_kind = metadata.get("chunk_kind")
    chunk_strategy = metadata.get("chunk_strategy")
    adjustment = 0.0
    if chunk_kind in preferred or chunk_strategy in preferred:
        adjustment += 0.04
    if chunk_kind == "table_row" and route == "table":
        adjustment += 0.04
    if metadata.get("node_type") == "clause" and route in {"version", "conflict"}:
        adjustment += 0.03
    if metadata.get("node_type") == "definition" and route == "definition":
        adjustment += 0.03
    if metadata.get("is_latest_version") is True and query_route.requires_current_version:
        adjustment += 0.03
    if route == "visual":
        if hit.chunk.modality == "image":
            adjustment += 0.08
        elif hit.chunk.modality == "document":
            adjustment += 0.05
        if chunk_kind in {"rendered_page_image", "embedded_image", "media_object"}:
            adjustment += 0.05
        if chunk_kind in {"document_page", "document_attachment"}:
            adjustment += 0.03
        if metadata.get("attachment_scope") in {"page_image", "embedded_image"}:
            adjustment += 0.03
    return adjustment


def _setting_enabled(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return parse_bool_env(str(value))
