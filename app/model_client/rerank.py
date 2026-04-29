from __future__ import annotations

import os

from app.model_client.base import GatewayClient
from app.model_client.schemas import RerankResult

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.config import AppConfig


class RerankClient(GatewayClient):
    def __init__(self, config: AppConfig) -> None:
        if not config.rerank_enabled:
            self.config = config
            self.capability = "rerank"
            self.base_url = ""
            self.api_key = ""
            self.timeout = config.settings["timeout"]["rerank_seconds"]
            self._client = None
            self.alias = "disabled"
            self.path = ""
            return
        super().__init__(config, config.settings["timeout"]["rerank_seconds"], "rerank")
        self.alias = config.models["rerank"]["default_alias"]
        self.path = (
            os.getenv("RERANK_API_PATH")
            or config.models.get("rerank", {}).get("path")
            or ""
        )

    async def rerank(
        self, query: str, documents: list[str], top_k: int
    ) -> list[RerankResult]:
        if not documents:
            return []
        payload = {
            "model": self.alias,
            "query": query,
            "documents": documents,
            "top_n": top_k,
        }
        data = await self.post(self.path, payload)
        results = data.get("results", [])
        valid_results: list[RerankResult] = []
        for result in results:
            idx = result.get("index", -1)
            if 0 <= idx < len(documents):
                valid_results.append(
                    RerankResult(
                        index=idx,
                        score=float(result.get("relevance_score", 0.0)),
                        document=documents[idx],
                    )
                )
        valid_results.sort(key=lambda r: r.score, reverse=True)
        return valid_results[:top_k]
