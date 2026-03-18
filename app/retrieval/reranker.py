from app.model_client.rerank import RerankClient
from app.vectorstore.repository import SearchHit


class RetrievalReranker:
    def __init__(self, client: RerankClient) -> None:
        self.client = client

    async def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        results = await self.client.rerank(query, [hit.chunk.text for hit in hits], top_k)
        reranked: list[SearchHit] = []
        for result in results:
            hit = hits[result.index]
            reranked.append(SearchHit(chunk=hit.chunk, score=result.score))
        return reranked
