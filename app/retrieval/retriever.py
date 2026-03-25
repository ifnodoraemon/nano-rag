from app.model_client.embeddings import EmbeddingClient
from app.vectorstore.repository import SearchHit, VectorRepository


class Retriever:
    def __init__(self, repository: VectorRepository, embedding_client: EmbeddingClient) -> None:
        self.repository = repository
        self.embedding_client = embedding_client

    async def retrieve(
        self,
        query: str,
        top_k: int,
        kb_id: str = "default",
        tenant_id: str | None = None,
    ) -> list[SearchHit]:
        vector = (await self.embedding_client.embed_texts([query]))[0]
        return self.repository.search(vector, top_k, kb_id=kb_id, tenant_id=tenant_id)
