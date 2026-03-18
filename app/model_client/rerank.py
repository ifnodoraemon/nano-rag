from __future__ import annotations

from app.model_client.base import GatewayClient
from app.model_client.schemas import RerankResult

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.config import AppConfig


class RerankClient(GatewayClient):
    def __init__(self, config: AppConfig) -> None:
        super().__init__(config, config.settings["timeout"]["rerank_seconds"])
        self.alias = config.models["rerank"]["default_alias"]

    async def rerank(self, query: str, documents: list[str], top_k: int) -> list[RerankResult]:
        if not documents:
            return []
        payload = {"model": self.alias, "query": query, "documents": documents, "top_n": top_k}
        data = await self.post("/rerank", payload)
        results = data.get("results", [])
        return [
            RerankResult(index=result["index"], score=float(result["relevance_score"]), document=documents[result["index"]])
            for result in results
        ]
