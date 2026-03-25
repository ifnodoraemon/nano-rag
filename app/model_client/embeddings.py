from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

from app.model_client.base import GatewayClient

if TYPE_CHECKING:
    from app.core.config import AppConfig

DEFAULT_BATCH_SIZE = int(os.getenv("RAG_EMBED_BATCH_SIZE", "100"))
DEFAULT_CONCURRENCY_LIMIT = int(os.getenv("RAG_EMBED_CONCURRENCY", "5"))


class EmbeddingClient(GatewayClient):
    def __init__(self, config: AppConfig) -> None:
        super().__init__(
            config, config.settings["timeout"]["embeddings_seconds"], "embedding"
        )
        self.alias = config.models["embedding"]["default_alias"]
        self._semaphore = asyncio.Semaphore(DEFAULT_CONCURRENCY_LIMIT)

    async def embed_texts(
        self, texts: list[str], batch_size: int = DEFAULT_BATCH_SIZE
    ) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            async with self._semaphore:
                payload = {"model": self.alias, "input": batch}
                data = await self.post("/embeddings", payload)
                for item in data.get("data", []):
                    idx = item.get("index", 0)
                    if 0 <= idx < len(batch) and results[i + idx] is None:
                        results[i + idx] = item.get("embedding")

        return [r if r is not None else [] for r in results]
