from __future__ import annotations

import os

from app.core.exceptions import ConfigurationError, ModelGatewayError
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
        ).strip()
        if not self.path:
            # An empty path used to POST to the bare base URL, which silently
            # hits whatever the gateway happens to serve there.
            raise ConfigurationError(
                "rerank is enabled but no endpoint path is configured; set "
                "RERANK_API_PATH (e.g. /reranks) or disable rerank"
            )
        if not self.path.startswith("/"):
            self.path = f"/{self.path}"

    async def close(self) -> None:
        if hasattr(self, "provider_client"):
            await super().close()

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
        results = data.get("results")
        if not isinstance(results, list):
            raise ModelGatewayError(
                f"rerank provider returned no 'results' array (got {type(results).__name__})"
            )
        valid_results: list[RerankResult] = []
        for result in results:
            if not isinstance(result, dict):
                raise ModelGatewayError("rerank provider returned a non-object result")
            idx = result.get("index", -1)
            if not isinstance(idx, int) or not 0 <= idx < len(documents):
                # Out-of-range indices used to be dropped, silently shrinking
                # the candidate list instead of surfacing a broken response.
                raise ModelGatewayError(
                    f"rerank provider returned an out-of-range index {idx!r} "
                    f"for {len(documents)} document(s)"
                )
            raw_score = result.get("relevance_score")
            if raw_score is None:
                raise ModelGatewayError(
                    "rerank provider omitted relevance_score for an entry"
                )
            valid_results.append(
                RerankResult(
                    index=idx,
                    score=float(raw_score),
                    document=documents[idx],
                )
            )
        valid_results.sort(key=lambda r: r.score, reverse=True)
        return valid_results[:top_k]
