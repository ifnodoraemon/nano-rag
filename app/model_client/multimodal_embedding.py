from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Sequence, Union

import httpx

from app.core.exceptions import ModelGatewayError

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
class AudioItem:
    data: bytes
    mime_type: str


@dataclass(frozen=True)
class VideoItem:
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


EmbedItem = Union[TextItem, ImageItem, AudioItem, VideoItem, FileItem]


class MultimodalEmbeddingProvider(Protocol):
    alias: str
    dimension: int

    async def embed_one(self, items: Sequence[EmbedItem]) -> list[float]: ...

    async def embed_batch(
        self, batches: Sequence[Sequence[EmbedItem]]
    ) -> list[list[float]]: ...

    async def close(self) -> None: ...


class _BaseProvider:
    """Common HTTP plumbing shared by every provider."""

    alias: str
    dimension: int

    def __init__(self, config: "AppConfig") -> None:
        self.config = config
        self.timeout = int(
            config.settings.get("timeout", {}).get("embeddings_seconds", 60)
        )
        self._semaphore = asyncio.Semaphore(DEFAULT_CONCURRENCY_LIMIT)
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def embed_batch(
        self, batches: Sequence[Sequence[EmbedItem]]
    ) -> list[list[float]]:
        if not batches:
            return []

        async def _one(items: Sequence[EmbedItem]) -> list[float]:
            async with self._semaphore:
                return await self.embed_one(items)

        return list(await asyncio.gather(*(_one(items) for items in batches)))

    async def embed_one(self, items: Sequence[EmbedItem]) -> list[float]:
        raise NotImplementedError

class GeminiMultimodalEmbedding(_BaseProvider):
    """Direct adapter for Gemini Embedding 2 (gemini-embedding-2-preview).

    Gemini's :embedContent shape is provider-native, so it is called directly.
    """

    def __init__(self, config: "AppConfig") -> None:
        super().__init__(config)
        section = config.models.get("embedding", {})
        self.alias = (
            os.getenv("EMBEDDING_MODEL_ALIAS")
            or section.get("default_alias")
            or "gemini-embedding-2-preview"
        )
        self.base_url = self._resolve_base_url(section)
        self.api_key = self._resolve_api_key(section)
        self.dimension = int(section.get("dimension", 1536))

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
            or ""
        )

    @staticmethod
    def _build_part(item: EmbedItem) -> dict:
        if isinstance(item, TextItem):
            return {"text": item.text}
        if isinstance(item, (ImageItem, AudioItem, VideoItem)):
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
        if not self.api_key:
            raise ModelGatewayError(
                "EMBEDDING_API_KEY is not configured for the Gemini multimodal embedding client."
            )
        url = f"{self.base_url}/v1beta/models/{self.alias}:embedContent"
        try:
            response = await self._get_client().post(
                url,
                headers={
                    "x-goog-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                json=self._build_payload(items),
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


class DashScopeMultimodalEmbedding(_BaseProvider):
    """Aliyun DashScope multimodal-embedding-v1 / qwen3-vl-embedding.

    Endpoint:
      POST {base_url}/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding
    Auth: Authorization: Bearer <DASHSCOPE_API_KEY>
    Payload:
      {model, input: {contents: [{text}|{image}|{video}]}, parameters: {...}}
    """

    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com"
    DEFAULT_PATH = (
        "/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding"
    )

    def __init__(self, config: "AppConfig") -> None:
        super().__init__(config)
        section = config.models.get("embedding", {})
        self.alias = (
            os.getenv("EMBEDDING_MODEL_ALIAS")
            or section.get("default_alias")
            or "multimodal-embedding-v1"
        )
        self.base_url = (
            os.getenv("EMBEDDING_API_BASE_URL")
            or section.get("base_url")
            or self.DEFAULT_BASE_URL
        ).rstrip("/")
        self.api_key = (
            os.getenv("EMBEDDING_API_KEY")
            or section.get("api_key")
            or ""
        )
        self.dimension = int(section.get("dimension", 1536))
        self.enable_fusion = os.getenv(
            "DASHSCOPE_EMBED_ENABLE_FUSION", "true"
        ).lower() in ("true", "1", "yes")

    @staticmethod
    def _build_content(item: EmbedItem) -> dict:
        if isinstance(item, TextItem):
            return {"text": item.text}
        if isinstance(item, ImageItem):
            data_url = f"data:{item.mime_type};base64,{base64.b64encode(item.data).decode('ascii')}"
            return {"image": data_url}
        if isinstance(item, VideoItem):
            data_url = f"data:{item.mime_type};base64,{base64.b64encode(item.data).decode('ascii')}"
            return {"video": data_url}
        if isinstance(item, AudioItem):
            data_url = f"data:{item.mime_type};base64,{base64.b64encode(item.data).decode('ascii')}"
            return {"audio": data_url}
        if isinstance(item, FileItem):
            mime = item.resolve_mime()
            data_url = f"data:{mime};base64,{base64.b64encode(item.path.read_bytes()).decode('ascii')}"
            if mime.startswith("image/"):
                return {"image": data_url}
            if mime.startswith("video/"):
                return {"video": data_url}
            if mime.startswith("audio/"):
                return {"audio": data_url}
            return {"text": item.path.name}
        raise TypeError(f"unsupported embed item type: {type(item).__name__}")

    def _build_payload(self, items: Sequence[EmbedItem]) -> dict:
        params: dict = {"dimension": self.dimension}
        if self.enable_fusion:
            params["enable_fusion"] = True
        return {
            "model": self.alias,
            "input": {"contents": [self._build_content(item) for item in items]},
            "parameters": params,
        }

    @staticmethod
    def _extract_vector(body: dict) -> list[float]:
        output = body.get("output") or {}
        embeddings = output.get("embeddings") or []
        if not embeddings:
            raise ModelGatewayError(
                f"DashScope embedding API returned no embeddings: {body}"
            )
        first = embeddings[0]
        values = first.get("embedding") or first.get("values") or first.get("vector")
        if not isinstance(values, list) or not values:
            raise ModelGatewayError(
                f"DashScope embedding API returned no values: {body}"
            )
        return values

    async def embed_one(self, items: Sequence[EmbedItem]) -> list[float]:
        if not items:
            raise ValueError("embed_one requires at least one item")
        if not self.api_key:
            raise ModelGatewayError(
                "EMBEDDING_API_KEY is not configured for the DashScope multimodal embedding client."
            )
        url = f"{self.base_url}{self.DEFAULT_PATH}"
        try:
            response = await self._get_client().post(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=self._build_payload(items),
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ModelGatewayError(
                "embedding gateway timeout on DashScope multimodal-embedding"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ModelGatewayError(
                f"embedding request failed: {exc.response.status_code} "
                f"{exc.response.text.strip()}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelGatewayError(f"embedding connection failed: {exc}") from exc
        return self._extract_vector(response.json())


class VLLMMultimodalEmbedding(_BaseProvider):
    """Self-hosted vLLM serving Qwen3-VL-Embedding via OpenAI-compatible /v1/embeddings.

    Multimodal payload follows vLLM's chat-style content parts convention:
      input: [{type:"text", text:...}, {type:"image_url", image_url:{url:"data:..."}}]

    The exact accepted shape varies by vLLM version. Override
    VLLM_EMBED_PAYLOAD_STYLE=string|messages to tune for legacy text-only or
    fully-chat-templated servers respectively.
    """

    def __init__(self, config: "AppConfig") -> None:
        super().__init__(config)
        section = config.models.get("embedding", {})
        self.alias = (
            os.getenv("EMBEDDING_MODEL_ALIAS")
            or section.get("default_alias")
            or "Qwen/Qwen3-VL-Embedding-8B"
        )
        self.base_url = (
            os.getenv("EMBEDDING_API_BASE_URL")
            or section.get("base_url")
            or "http://localhost:8001/v1"
        ).rstrip("/")
        self.api_key = (
            os.getenv("EMBEDDING_API_KEY")
            or section.get("api_key")
            or "EMPTY"
        )
        self.dimension = int(section.get("dimension", 1536))

    @staticmethod
    def _build_part(item: EmbedItem) -> dict:
        if isinstance(item, TextItem):
            return {"type": "text", "text": item.text}
        if isinstance(item, ImageItem):
            data_url = f"data:{item.mime_type};base64,{base64.b64encode(item.data).decode('ascii')}"
            return {"type": "image_url", "image_url": {"url": data_url}}
        if isinstance(item, VideoItem):
            data_url = f"data:{item.mime_type};base64,{base64.b64encode(item.data).decode('ascii')}"
            return {"type": "video_url", "video_url": {"url": data_url}}
        if isinstance(item, AudioItem):
            data_url = f"data:{item.mime_type};base64,{base64.b64encode(item.data).decode('ascii')}"
            return {"type": "audio_url", "audio_url": {"url": data_url}}
        if isinstance(item, FileItem):
            mime = item.resolve_mime()
            data_url = f"data:{mime};base64,{base64.b64encode(item.path.read_bytes()).decode('ascii')}"
            kind = (
                "image_url"
                if mime.startswith("image/")
                else "video_url"
                if mime.startswith("video/")
                else "audio_url"
                if mime.startswith("audio/")
                else "image_url"
            )
            return {"type": kind, kind: {"url": data_url}}
        raise TypeError(f"unsupported embed item type: {type(item).__name__}")

    def _build_payload(self, items: Sequence[EmbedItem]) -> dict:
        # Default to vLLM messages-style for vision-language embeddings.
        # Some pre-0.14 servers need a flat string list — the user can build
        # a thin proxy or fork this provider.
        parts = [self._build_part(item) for item in items]
        return {
            "model": self.alias,
            "input": [{"role": "user", "content": parts}],
            "encoding_format": "float",
        }

    @staticmethod
    def _extract_vector(body: dict) -> list[float]:
        data = body.get("data") or []
        if not data:
            raise ModelGatewayError(
                f"vLLM embedding API returned no data: {body}"
            )
        embedding = data[0].get("embedding")
        if not isinstance(embedding, list) or not embedding:
            raise ModelGatewayError(
                f"vLLM embedding API returned no embedding: {body}"
            )
        return embedding

    async def embed_one(self, items: Sequence[EmbedItem]) -> list[float]:
        if not items:
            raise ValueError("embed_one requires at least one item")
        url = f"{self.base_url}/embeddings"
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            response = await self._get_client().post(
                url, headers=headers, json=self._build_payload(items)
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ModelGatewayError(
                "embedding gateway timeout on vLLM /v1/embeddings"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise ModelGatewayError(
                f"embedding request failed: {exc.response.status_code} "
                f"{exc.response.text.strip()}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelGatewayError(f"embedding connection failed: {exc}") from exc
        return self._extract_vector(response.json())


def create_multimodal_embedding(
    config: "AppConfig",
) -> MultimodalEmbeddingProvider:
    """Factory: pick a provider based on EMBEDDING_PROVIDER env / config."""
    raw = (
        os.getenv("EMBEDDING_PROVIDER")
        or config.models.get("embedding", {}).get("provider")
        or "gemini"
    )
    provider = str(raw).strip().lower()
    if provider in ("gemini", "google", "vertex"):
        return GeminiMultimodalEmbedding(config)
    if provider in ("dashscope", "aliyun", "qwen-cloud"):
        return DashScopeMultimodalEmbedding(config)
    if provider in ("vllm", "qwen-vllm", "openai-compat"):
        return VLLMMultimodalEmbedding(config)
    raise ModelGatewayError(
        f"unknown EMBEDDING_PROVIDER '{provider}'. "
        "Supported: gemini, dashscope, vllm."
    )
