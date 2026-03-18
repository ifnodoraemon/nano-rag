from __future__ import annotations

from app.model_client.base import GatewayClient

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.config import AppConfig


class EmbeddingClient(GatewayClient):
    def __init__(self, config: AppConfig) -> None:
        super().__init__(config, config.settings["timeout"]["embeddings_seconds"])
        self.alias = config.models["embedding"]["default_alias"]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": self.alias, "input": texts}
        data = await self.post("/embeddings", payload)
        return [item["embedding"] for item in data.get("data", [])]
