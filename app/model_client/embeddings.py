from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

from app.core.exceptions import ModelGatewayError
from app.model_client.multimodal_embedding import (
    EmbedItem,
    GeminiMultimodalEmbedding,
    TextItem,
)

if TYPE_CHECKING:
    from app.core.config import AppConfig

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Embedding facade.

    The wire-level client is multimodal (Gemini Embedding 2). This class
    keeps the legacy ``embed_texts(list[str])`` signature so retrieval and
    test fakes do not have to change, and adds ``embed_items`` for ingestion
    paths that need to embed images alongside text.
    """

    def __init__(self, config: "AppConfig") -> None:
        self.config = config
        self._provider = GeminiMultimodalEmbedding(config)
        self.alias = self._provider.alias
        self.dimension = self._provider.dimension

    async def embed_texts(
        self, texts: Sequence[str], batch_size: int | None = None
    ) -> list[list[float]]:
        # batch_size kept for backward compat; concurrency is controlled
        # inside the provider.
        del batch_size
        if not texts:
            return []
        items: list[list[EmbedItem]] = [[TextItem(text)] for text in texts]
        vectors = await self._provider.embed_batch(items)
        if len(vectors) != len(texts):
            raise ModelGatewayError(
                "embedding API returned an inconsistent number of vectors"
            )
        return vectors

    async def embed_items(
        self, items: Sequence[Sequence[EmbedItem]]
    ) -> list[list[float]]:
        if not items:
            return []
        vectors = await self._provider.embed_batch(items)
        if len(vectors) != len(items):
            raise ModelGatewayError(
                "embedding API returned an inconsistent number of vectors"
            )
        return vectors

    async def close(self) -> None:
        await self._provider.close()
