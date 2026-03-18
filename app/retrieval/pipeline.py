from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.tracing import TraceSession
from app.model_client.embeddings import EmbeddingClient
from app.model_client.rerank import RerankClient
from app.retrieval.context_builder import build_contexts
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
        trace_store,
        tracing_manager: TracingManager,
    ) -> None:
        self.config = config
        self.retriever = Retriever(repository, embedding_client)
        self.reranker = RetrievalReranker(rerank_client)
        self.trace_store = trace_store
        self.tracing_manager = tracing_manager

    async def run(self, query: str, top_k: int | None = None) -> tuple[list[dict[str, object]], dict[str, object]]:
        requested_top_k = top_k or self.config.settings["retrieval"]["top_k"]
        with self.tracing_manager.span(
            "retrieval.run",
            {"retrieval.query": query, "retrieval.top_k": requested_top_k},
        ):
            trace = TraceSession()
            retrieved = await self.retriever.retrieve(query, requested_top_k)
            reranked = await self.reranker.rerank(query, retrieved, self.config.settings["retrieval"]["rerank_top_k"])
            retrieved_contexts = build_contexts(retrieved, requested_top_k)
            reranked_contexts = build_contexts(reranked, self.config.settings["retrieval"]["rerank_top_k"])
            contexts = build_contexts(reranked, self.config.settings["retrieval"]["final_contexts"])
            trace.record("query", query)
            trace.record("retrieved_chunk_ids", [hit.chunk.chunk_id for hit in retrieved])
            trace.record("reranked_chunk_ids", [hit.chunk.chunk_id for hit in reranked])
            trace.record("retrieved", retrieved_contexts)
            trace.record("reranked", reranked_contexts)
            trace.record("contexts", contexts)
            final_trace = trace.finish()
            self.trace_store.save(final_trace)
            return contexts, final_trace

    async def debug(self, query: str, top_k: int | None = None) -> RetrievalDebugResponse:
        contexts, trace = await self.run(query, top_k)
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
