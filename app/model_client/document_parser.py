from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from app.core.exceptions import ModelGatewayError, ParsingError
from app.utils.text import parse_bool_env

if TYPE_CHECKING:
    from app.core.config import AppConfig


_MODEL_PARSER_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
_CONTENT_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}
_EMPTY_SENTINEL = "__PARSER_EMPTY__"


class DocumentParserClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.timeout = int(config.settings["timeout"].get("document_parser_seconds", 120))
        self.enabled = self._resolve_enabled()
        self.alias = self._resolve_alias()
        self.base_url = self._resolve_base_url()
        self.api_key = self._resolve_api_key()
        self._client: httpx.AsyncClient | None = None

    def _resolve_enabled(self) -> bool:
        raw = os.getenv("DOCUMENT_PARSER_ENABLED")
        if raw is not None:
            return parse_bool_env(raw)
        section = self.config.models.get("document_parser", {})
        value = section.get("enabled", True)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return parse_bool_env(value)
        return bool(value)

    def _resolve_alias(self) -> str:
        return (
            os.getenv("DOCUMENT_PARSER_MODEL")
            or self.config.models.get("document_parser", {}).get("default_alias")
            or self.config.models.get("generation", {}).get("default_alias")
            or "gemini-2.5-flash"
        )

    def _resolve_base_url(self) -> str:
        raw = (
            os.getenv("DOCUMENT_PARSER_API_BASE_URL")
            or self.config.models.get("document_parser", {}).get("base_url")
            or self.config.gateway_base_url
        )
        if not raw:
            return "https://generativelanguage.googleapis.com"
        base_url = str(raw).rstrip("/")
        for suffix in ("/v1beta/openai", "/v1/openai", "/openai", "/v1beta", "/v1"):
            if base_url.endswith(suffix):
                return base_url[: -len(suffix)]
        return base_url

    def _resolve_api_key(self) -> str:
        return (
            os.getenv("DOCUMENT_PARSER_API_KEY")
            or self.config.models.get("document_parser", {}).get("api_key")
            or self.config.gateway_for("generation")["api_key"]
        )

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in _MODEL_PARSER_SUFFIXES

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def parse_file(self, path: Path) -> str:
        if not self.enabled or not self.supports(path):
            raise ParsingError(f"model parser is not enabled for {path.suffix}")
        if self.config.gateway_mode == "mock":
            raise ParsingError("document parser is not available in mock mode")
        mime_type = self._guess_content_type(path)
        file_uri = await self._upload_file(path, mime_type)
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "Convert this file into clean Markdown for retrieval. "
                                "Preserve headings, table values, numbered items, place names, and key numeric data. "
                                f"If the file is unreadable or has no extractable content, return exactly `{_EMPTY_SENTINEL}`."
                            )
                        },
                        {"file_data": {"mime_type": mime_type, "file_uri": file_uri}},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "text/plain",
            },
        }
        client = self._get_client()
        try:
            response = await client.post(
                f"{self.base_url}/v1beta/models/{self.alias}:generateContent",
                headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ModelGatewayError("document parser timeout on Gemini generateContent") from exc
        except httpx.HTTPStatusError as exc:
            raise ModelGatewayError(
                f"document parser request failed: {exc.response.status_code} {exc.response.text.strip()}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelGatewayError(f"document parser connection failed: {exc}") from exc
        text = self._extract_text(response.json())
        if not text or text.strip() == _EMPTY_SENTINEL:
            raise ParsingError(f"document parser returned empty content for {path.name}")
        return text.strip()

    async def _upload_file(self, path: Path, mime_type: str) -> str:
        client = self._get_client()
        start_headers = {
            "x-goog-api-key": self.api_key,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(path.stat().st_size),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
        }
        start_payload = {"file": {"display_name": path.name}}
        try:
            start_response = await client.post(
                f"{self.base_url}/upload/v1beta/files",
                headers=start_headers,
                json=start_payload,
            )
            start_response.raise_for_status()
            upload_url = start_response.headers.get("X-Goog-Upload-URL")
            if not upload_url:
                raise ParsingError(f"document parser did not return an upload URL for {path.name}")
            if not upload_url.startswith(self.base_url):
                raise ParsingError(
                    f"upload URL from upstream does not match expected base: {self.base_url}"
                )
            finalize_headers = {
                "x-goog-api-key": self.api_key,
                "X-Goog-Upload-Offset": "0",
                "X-Goog-Upload-Command": "upload, finalize",
                "Content-Type": mime_type,
            }
            finalize_response = await client.post(
                upload_url,
                headers=finalize_headers,
                content=path.read_bytes(),
            )
            finalize_response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ModelGatewayError("document parser timeout while uploading file") from exc
        except httpx.HTTPStatusError as exc:
            raise ModelGatewayError(
                f"document parser upload failed: {exc.response.status_code} {exc.response.text.strip()}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ModelGatewayError(f"document parser upload connection failed: {exc}") from exc
        file_payload = finalize_response.json().get("file") or {}
        file_uri = file_payload.get("uri")
        if not file_uri:
            raise ParsingError(f"document parser did not return a file URI for {path.name}")
        return str(file_uri)

    def _extract_text(self, payload: dict) -> str:
        texts: list[str] = []
        for candidate in payload.get("candidates", []) or []:
            content = candidate.get("content") or {}
            for part in content.get("parts", []) or []:
                text = part.get("text")
                if text:
                    texts.append(text)
        return "\n".join(texts).strip()

    def _guess_content_type(self, path: Path) -> str:
        suffix = path.suffix.lower()
        guessed = _CONTENT_TYPES.get(suffix)
        if guessed:
            return guessed
        mime_type, _ = mimetypes.guess_type(path.name)
        return mime_type or "application/octet-stream"
