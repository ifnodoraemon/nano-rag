from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence, Union

import httpx

from app.core.exceptions import ModelGatewayError
from app.model_client.mock_gateway import mock_embeddings

if TYPE_CHECKING:
    from app.core.config import AppConfig

logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY_LIMIT = int(os.getenv("RAG_EMBED_CONCURRENCY", "5"))


@dataclass(frozen=True)
class TextItem:
    text: str


@dataclass(frozen=True)
class ImageItem:
    data: bytes
    mime_type: str


@dataclass(frozen=True)
class FileItem:
    path: Path
    mime_type: str | None = None

    def resolve_mime(self) -> str:
        if self.mime_type:
            return self.mime_type
        guessed, _ = mimetypes.guess_type(self.path.name)
        return guessed or "application/octet-stream"


EmbedItem = Union[TextItem, ImageItem, FileItem]


class GeminiMultimodalEmbedding:
    """Direct adapter for Gemini Embedding 2 (gemini-embedding-2-preview).

    The Gemini multimodal embedding API is not OpenAI-compatible and cannot be
    routed through Bifrost. This client follows the same direct-call pattern
    used by ``app.model_client.document_parser.DocumentParserClient``.
    """

    def __init__(self, config: "AppConfig") -> None:
        self.config = config
        section = config.models.get("embedding", {})
        self.alias = (
            os.getenv("EMBEDDING_MODEL_ALIAS")
            or section.get("default_alias")
            or "gemini-embedding-2-preview"
        )
        self.base_url = self._resolve_base_url(section)
        self.api_key = self._resolve_api_key(section)
        self.dimension = int(section.get("dimension", 1536))
        self.timeout = int(
            config.settings.get("timeout", {}).get("embeddings_seconds", 60)
        )
        self._semaphore = asyncio.Semaphore(DEFAULT_CONCURRENCY_LIMIT)
        self._client: httpx.AsyncClient | None = None

    @staticmethod
    def _resolve_base_url(section: dict) -> str:
        raw = (
            os.getenv("EMBEDDING_API_BASE_URL")
            or section.get("base_url")
            or "https://generativelanguage.googleapis.com"
        )
        base = str(raw).rstrip("/") if raw else "https://generativelanguage.googleapis.com"
        for suffix in ("/v1beta/openai", "/v1/openai", "/openai", "/v1beta", "/v1"):
            if base.endswith(suffix):
                return base[: -len(suffix)]
        return base

    @staticmethod
    def _resolve_api_key(section: dict) -> str:
        return (
            os.getenv("EMBEDDING_API_KEY")
            or section.get("api_key")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("MODEL_GATEWAY_API_KEY")
            or ""
        )

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _build_part(item: EmbedItem) -> dict:
        if isinstance(item, TextItem):
            return {"text": item.text}
        if isinstance(item, ImageItem):
            return {
                "inline_data": {
                    "mime_type": item.mime_type,
                    "data": base64.b64encode(item.data).decode("ascii"),
                }
            }
        if isinstance(item, FileItem):
            return {
                "inline_data": {
                    "mime_type": item.resolve_mime(),
                    "data": base64.b64encode(item.path.read_bytes()).decode("ascii"),
                }
            }
        raise TypeError(f"unsupported embed item type: {type(item).__name__}")

    def _build_payload(self, items: Sequence[EmbedItem]) -> dict:
        return {
            "content": {"parts": [self._build_part(item) for item in items]},
            "output_dimensionality": self.dimension,
        }

    @staticmethod
    def _extract_vector(body: dict) -> list[float]:
        embeddings = body.get("embeddings")
        if isinstance(embeddings, list) and embeddings:
            values = embeddings[0].get("values")
            if isinstance(values, list) and values:
                return values
        embedding = body.get("embedding")
        if isinstance(embedding, dict):
            values = embedding.get("values")
            if isinstance(values, list) and values:
                return values
        raise ModelGatewayError(f"embedding API returned no values: {body}")

    async def embed_one(self, items: Sequence[EmbedItem]) -> list[float]:
        if not items:
            raise ValueError("embed_one requires at least one item")
        if self.config.gateway_mode == "mock":
            return self._mock_vector(items)
        if not self.api_key:
            raise ModelGatewayError(
                "EMBEDDING_API_KEY (or GEMINI_API_KEY) is not configured for "
                "the multimodal embedding client."
            )
        payload = self._build_payload(items)
        url = f"{self.base_url}/v1beta/models/{self.alias}:embedContent"
        try:
            response = await self._get_client().post(
                url,
                headers={
                    "x-goog-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ModelGatewayError(
                "embedding gateway timeout on Gemini :embedContent"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ModelGatewayError(
                f"embedding request failed: {exc.response.status_code} "
                f"{exc.response.text.strip()}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelGatewayError(f"embedding connection failed: {exc}") from exc
        vector = self._extract_vector(response.json())
        if len(vector) != self.dimension:
            logger.warning(
                "embedding dimension mismatch: expected %d, got %d",
                self.dimension,
                len(vector),
            )
        return vector

    async def embed_batch(
        self, batches: Sequence[Sequence[EmbedItem]]
    ) -> list[list[float]]:
        if not batches:
            return []

        async def _one(items: Sequence[EmbedItem]) -> list[float]:
            async with self._semaphore:
                return await self.embed_one(items)

        return list(await asyncio.gather(*(_one(items) for items in batches)))

    def _mock_vector(self, items: Sequence[EmbedItem]) -> list[float]:
        text_parts: list[str] = []
        for item in items:
            if isinstance(item, TextItem):
                text_parts.append(item.text)
            elif isinstance(item, ImageItem):
                text_parts.append(f"<image:{item.mime_type}:{len(item.data)}>")
            elif isinstance(item, FileItem):
                text_parts.append(
                    f"<file:{item.path.name}:{item.resolve_mime()}>"
                )
        joined = "\n".join(text_parts) or "<empty>"
        body = mock_embeddings([joined], dimensions=self.dimension)
        return body["data"][0]["embedding"]
