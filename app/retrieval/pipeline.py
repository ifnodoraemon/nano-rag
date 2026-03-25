from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

from app.core.tracing import TraceSession, TraceStore
from app.model_client.embeddings import EmbeddingClient
from app.model_client.rerank import RerankClient
from app.retrieval.context_builder import build_contexts
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.reranker import RetrievalReranker
from app.retrieval.retriever import Retriever
from app.schemas.trace import RetrievalDebugResponse
from app.vectorstore.repository import VectorRepository

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
    ) -> None:
        self.config = config
        self.retriever = Retriever(repository, embedding_client, query_rewriter)
        self.reranker = RetrievalReranker(rerank_client)
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
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        requested_top_k = top_k or self.config.settings["retrieval"]["top_k"]
        rerank_top_k = self.config.settings["retrieval"]["rerank_top_k"]
        final_contexts_limit = self.config.settings["retrieval"]["final_contexts"]
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
            retrieved = await self.retriever.retrieve(
                query, requested_top_k, kb_id=kb_id, tenant_id=tenant_id
            )
            retrieval_seconds = round(perf_counter() - retrieval_started, 4)
            rerank_seconds = 0.0
            if self.config.rerank_enabled:
                rerank_started = perf_counter()
                reranked = await self.reranker.rerank(query, retrieved, rerank_top_k)
                rerank_seconds = round(perf_counter() - rerank_started, 4)
            else:
                reranked = retrieved[:rerank_top_k]
            retrieved_contexts = build_contexts(retrieved, requested_top_k)
            reranked_contexts = build_contexts(reranked, rerank_top_k)
            contexts = build_contexts(reranked, final_contexts_limit)
            trace.record("query", query)
            trace.record("kb_id", kb_id)
            trace.record("tenant_id", tenant_id)
            trace.record("session_id", session_id)
            trace.record("sample_id", sample_id)
            trace.record(
                "retrieved_chunk_ids", [hit.chunk.chunk_id for hit in retrieved]
            )
            trace.record("reranked_chunk_ids", [hit.chunk.chunk_id for hit in reranked])
            trace.record("retrieved", retrieved_contexts)
            trace.record("reranked", reranked_contexts)
            trace.record("contexts", contexts)
            trace.record(
                "retrieval_params",
                {
                    "requested_top_k": requested_top_k,
                    "rerank_top_k": rerank_top_k,
                    "final_contexts": final_contexts_limit,
                    "rerank_enabled": self.config.rerank_enabled,
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
            self.trace_store.save(final_trace)
            return contexts, final_trace

    async def debug(
        self,
        query: str,
        top_k: int | None = None,
        kb_id: str = "default",
        tenant_id: str | None = None,
        session_id: str | None = None,
    ) -> RetrievalDebugResponse:
        contexts, trace = await self.run(
            query, top_k, kb_id=kb_id, tenant_id=tenant_id, session_id=session_id
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
