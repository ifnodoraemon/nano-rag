from __future__ import annotations

import base64
import asyncio
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from pypdf import PdfReader, PdfWriter

from app.core.exceptions import ConfigurationError, ModelGatewayError, ParsingError
from app.model_client.base import AsyncJsonProviderClient, ProviderConfig
from app.utils.text import parse_bool_env

if TYPE_CHECKING:
    from app.core.config import AppConfig


_MODEL_PARSER_SUFFIXES = {
    ".doc",
    ".pdf",
    ".ppt",
    ".xls",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}
_CONTENT_TYPES = {
    ".doc": "application/msword",
    ".pdf": "application/pdf",
    ".ppt": "application/vnd.ms-powerpoint",
    ".xls": "application/vnd.ms-excel",
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
        self.provider = self._resolve_provider()
        self.alias = self._resolve_alias()
        self.base_url = self._resolve_base_url()
        self.api_key = self._resolve_api_key()
        self.openai_client = AsyncJsonProviderClient(
            ProviderConfig(
                capability="document_parser",
                provider=self.provider,
                model=str(self.alias or ""),
                base_url=self.base_url,
                api_key=self.api_key,
            ),
            self.timeout,
        )
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

    def _resolve_provider(self) -> str:
        raw = (
            os.getenv("DOCUMENT_PARSER_PROVIDER")
            or self.config.models.get("document_parser", {}).get("provider")
            or "gemini"
        )
        provider = str(raw).strip().lower()
        if not self.enabled:
            return provider or "disabled"
        if provider in {"gemini", "google"}:
            return "gemini"
        if provider in {"qwen", "dashscope", "vllm", "openai-compat"}:
            return "qwen"
        raise ParsingError(
            f"unknown DOCUMENT_PARSER_PROVIDER '{provider}'. Supported: gemini, qwen."
        )

    def _resolve_alias(self) -> str:
        return (
            os.getenv("DOCUMENT_PARSER_MODEL")
            or self.config.models.get("document_parser", {}).get("default_alias")
        )

    def _resolve_base_url(self) -> str:
        if not self.enabled:
            return ""
        raw = (
            os.getenv("DOCUMENT_PARSER_API_BASE_URL")
            or self.config.models.get("document_parser", {}).get("base_url")
        )
        if not raw:
            raise ParsingError("DOCUMENT_PARSER_API_BASE_URL must be configured.")
        base_url = str(raw).rstrip("/")
        if self.provider == "qwen":
            return base_url
        for suffix in ("/v1beta/openai", "/v1/openai", "/openai", "/v1beta", "/v1"):
            if base_url.endswith(suffix):
                return base_url[: -len(suffix)]
        return base_url

    def _resolve_api_key(self) -> str:
        return (
            os.getenv("DOCUMENT_PARSER_API_KEY")
            or self.config.models.get("document_parser", {}).get("api_key")
            or ""
        )

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in _MODEL_PARSER_SUFFIXES

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout, follow_redirects=False
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        await self.openai_client.close()

    async def parse_file(self, path: Path) -> str:
        if not self.enabled or not self.supports(path):
            raise ParsingError(f"model parser is not enabled for {path.suffix}")
        if not self.alias:
            raise ParsingError("DOCUMENT_PARSER_MODEL must be configured.")
        if not self.api_key:
            raise ParsingError("DOCUMENT_PARSER_API_KEY must be configured.")
        if self.provider == "qwen":
            return await self._parse_file_openai_compatible(path)
        if self.provider != "gemini":
            raise ParsingError(f"unsupported document parser provider: {self.provider}")
        if path.suffix.lower() == ".pdf":
            batched = await self._parse_pdf_in_page_batches(path)
            if batched is not None:
                return batched
        return await self._parse_file_gemini_once(path)

    async def _parse_pdf_in_page_batches(self, path: Path) -> str | None:
        batch_size = self._pdf_page_batch_size()
        if batch_size <= 0:
            return None
        try:
            reader = PdfReader(str(path))
            page_count = len(reader.pages)
        except Exception:
            # pypdf is only the local page-splitting helper here, not the
            # configured parser; if it cannot split the file we parse the
            # whole file with the model parser instead (same parser, same
            # completeness guarantees — its empty-sentinel contract still
            # fails unreadable documents loudly).
            return None
        if page_count <= batch_size:
            return None
        parts: list[str] = []
        with tempfile.TemporaryDirectory(prefix="nano-rag-pdf-") as tmp_dir:
            batches: list[tuple[int, int, Path]] = []
            for start in range(0, page_count, batch_size):
                end = min(start + batch_size, page_count)
                writer = PdfWriter()
                for index in range(start, end):
                    writer.add_page(reader.pages[index])
                batch_path = Path(tmp_dir) / f"{path.stem}_pages_{start + 1}_{end}.pdf"
                with batch_path.open("wb") as handle:
                    writer.write(handle)
                batches.append((start + 1, end, batch_path))
            semaphore = asyncio.Semaphore(self._pdf_page_batch_concurrency())

            async def parse_batch(batch: tuple[int, int, Path]) -> tuple[int, int, str]:
                start_page, end_page, batch_path = batch
                async with semaphore:
                    # No per-batch swallowing: a failed batch means pages are
                    # missing from the parsed output, so keeping the surviving
                    # batches with an in-band warning header is silent partial
                    # data. Each call has already been retried by the base
                    # client; a failure here is terminal for the document.
                    try:
                        text = await self._parse_file_gemini_once(batch_path)
                    except Exception as exc:
                        raise ModelGatewayError(
                            f"document parser batch pages {start_page}-{end_page} "
                            f"failed for {path.name}: {exc}"
                        ) from exc
                    return start_page, end_page, text.strip()

            for start_page, end_page, text in await asyncio.gather(
                *(parse_batch(batch) for batch in batches)
            ):
                parts.append(f"# Pages {start_page}-{end_page}\n\n{text}")
        return "\n\n".join(parts).strip()

    def _pdf_page_batch_size(self) -> int:
        raw = os.getenv("DOCUMENT_PARSER_PDF_PAGE_BATCH_SIZE", "3")
        try:
            return max(1, int(raw))
        except ValueError as exc:
            raise ConfigurationError(
                f"DOCUMENT_PARSER_PDF_PAGE_BATCH_SIZE must be an integer, got {raw!r}"
            ) from exc

    def _pdf_page_batch_concurrency(self) -> int:
        raw = os.getenv("DOCUMENT_PARSER_PDF_PAGE_BATCH_CONCURRENCY", "2")
        try:
            return max(1, int(raw))
        except ValueError as exc:
            raise ConfigurationError(
                f"DOCUMENT_PARSER_PDF_PAGE_BATCH_CONCURRENCY must be an integer, got {raw!r}"
            ) from exc

    async def _parse_file_gemini_once(self, path: Path) -> str:
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

    async def _parse_file_openai_compatible(self, path: Path) -> str:
        mime_type = self._guess_content_type(path)
        data_url = (
            f"data:{mime_type};base64,"
            f"{base64.b64encode(path.read_bytes()).decode('ascii')}"
        )
        content_part = self._build_qwen_file_part(path, mime_type, data_url)
        payload = {
            "temperature": 0,
        }
        data = await self.openai_client.chat_completions(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Convert this file into clean Markdown for retrieval. "
                                "Preserve headings, table values, numbered items, place names, and key numeric data. "
                                f"If the file is unreadable or has no extractable content, return exactly `{_EMPTY_SENTINEL}`."
                            ),
                        },
                        content_part,
                    ],
                }
            ],
            self.alias,
            **payload,
        )
        text = self._extract_chat_text(data)
        if not text or text.strip() == _EMPTY_SENTINEL:
            raise ParsingError(f"document parser returned empty content for {path.name}")
        return text.strip()

    def _build_qwen_file_part(self, path: Path, mime_type: str, data_url: str) -> dict:
        style = os.getenv("DOCUMENT_PARSER_QWEN_FILE_PART_STYLE", "").strip().lower()
        if not style:
            style = "image_url" if mime_type.startswith("image/") else "file"
        if style == "image_url":
            return {"type": "image_url", "image_url": {"url": data_url}}
        if style == "file":
            return {
                "type": "file",
                "file": {"filename": path.name, "file_data": data_url},
            }
        raise ParsingError(
            "DOCUMENT_PARSER_QWEN_FILE_PART_STYLE must be image_url or file."
        )

    async def _upload_file(self, path: Path, mime_type: str) -> str:
        return await self._upload_file_resumable(path, mime_type)

    async def _upload_file_resumable(self, path: Path, mime_type: str) -> str:
        client = self._get_client()
        file_bytes = path.read_bytes()
        file_size = len(file_bytes)
        start_headers = {
            "x-goog-api-key": self.api_key,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(file_size),
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
                content=file_bytes,
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
        return self._extract_file_uri(finalize_response.json(), path)

    def _extract_file_uri(self, payload: dict, path: Path) -> str:
        file_payload = payload.get("file") or {}
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

    def _extract_chat_text(self, payload: dict) -> str:
        texts: list[str] = []
        for choice in payload.get("choices", []) or []:
            message = choice.get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("text"):
                        texts.append(str(part["text"]))
        return "\n".join(texts).strip()

    def _guess_content_type(self, path: Path) -> str:
        suffix = path.suffix.lower()
        content_type = _CONTENT_TYPES.get(suffix)
        if not content_type:
            raise ParsingError(f"document parser content type is not configured for {suffix}")
        return content_type
