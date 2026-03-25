from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.exceptions import ModelGatewayError
from app.model_client.embeddings import EmbeddingClient
from app.retrieval.query_rewriter import QueryRewriter, QueryRewriterConfig
from app.vectorstore.repository import SearchHit, VectorRepository

if TYPE_CHECKING:
    from app.model_client.generation import GenerationClient


class Retriever:
    def __init__(
        self,
        repository: VectorRepository,
        embedding_client: EmbeddingClient,
        query_rewriter: QueryRewriter | None = None,
    ) -> None:
        self.repository = repository
        self.embedding_client = embedding_client
        self.query_rewriter = query_rewriter

    async def retrieve(
        self,
        query: str,
        top_k: int,
        kb_id: str = "default",
        tenant_id: str | None = None,
    ) -> list[SearchHit]:
        effective_query = query
        if self.query_rewriter:
            effective_query = await self.query_rewriter.rewrite(query)
        vectors = await self.embedding_client.embed_texts([effective_query])
        if not vectors:
            raise ModelGatewayError("embedding service returned empty result")
        return self.repository.search(
            vectors[0], top_k, kb_id=kb_id, tenant_id=tenant_id
        )
