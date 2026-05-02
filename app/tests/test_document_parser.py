import base64
from pathlib import Path

import pytest

from app.core.exceptions import ParsingError
from app.ingestion.parser_docling import parse_document
from app.ingestion.pipeline import IngestionPipeline
from app.model_client.document_parser import DocumentParserClient
from app.core.config import AppConfig
from app.core.tracing import TracingManager
from app.vectorstore.repository import InMemoryVectorRepository


class FakeEmbeddingClient:
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:  # noqa: ARG002
        return [[0.0, 0.0]]

    async def embed_items(self, items):  # noqa: ANN001
        return await self.embed_texts([
            "\n".join(getattr(item, "text", "") for item in batch)
            for batch in items
        ])


class FakeDocumentParser:
    def supports(self, path: Path) -> bool:  # noqa: ARG002
        return True

    async def parse_file(self, path: Path) -> str:  # noqa: ARG002
        return "# Parsed PDF\n\n徐套乡区片综合地价为 62000 元/亩。"


@pytest.mark.asyncio
async def test_parse_document_uses_model_parser_for_pdf(tmp_path) -> None:
    pdf_path = tmp_path / "notice.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    text = await parse_document(pdf_path, FakeDocumentParser())

    assert "徐套乡区片综合地价" in text


@pytest.mark.asyncio
async def test_ingestion_pipeline_rejects_empty_parsed_content(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RAG_INGEST_ALLOWED_DIRS", str(tmp_path))
    monkeypatch.setenv("PARSED_OUTPUT_DIR", str(tmp_path / "parsed"))

    pdf_path = tmp_path / "notice.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    async def fake_parse_document(path, document_parser=None):  # noqa: ANN001, ARG001
        return ""

    monkeypatch.setattr("app.ingestion.pipeline.parse_document", fake_parse_document)

    config = AppConfig(
        config_dir=tmp_path,
        settings={"chunk": {"size": 200, "overlap": 20}, "timeout": {"document_parser_seconds": 30}},
        models={"model_gateway": {"base_url": "", "api_key": ""}},
        prompts={},
    )
    pipeline = IngestionPipeline(
        config=config,
        repository=InMemoryVectorRepository(),
        embedding_client=FakeEmbeddingClient(),
        tracing_manager=TracingManager("test-service", ""),
    )

    with pytest.raises(ParsingError) as exc_info:
        await pipeline.run(str(pdf_path), kb_id="default")

    assert "returned empty content" in str(exc_info.value)


def test_document_parser_base_url_strips_openai_suffix(tmp_path) -> None:
    config = AppConfig(
        config_dir=tmp_path,
        settings={"timeout": {"document_parser_seconds": 30}},
        models={
            "model_gateway": {
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                "api_key": "secret",
            },
            "generation": {"default_alias": "gemini-3.1-pro-preview"},
            "document_parser": {
                "enabled": True,
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                "api_key": "secret",
            },
        },
        prompts={},
    )

    client = DocumentParserClient(config)

    assert client.base_url == "https://generativelanguage.googleapis.com"


def test_document_parser_disabled_does_not_require_base_url(tmp_path) -> None:
    config = AppConfig(
        config_dir=tmp_path,
        settings={"timeout": {"document_parser_seconds": 30}},
        models={
            "document_parser": {
                "enabled": False,
                "provider": "unknown-provider",
            },
        },
        prompts={},
    )

    client = DocumentParserClient(config)

    assert client.enabled is False
    assert client.base_url == ""


@pytest.mark.asyncio
async def test_document_parser_uses_direct_resumable_upload(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MODEL_GATEWAY_MODE", "live")
    pdf_path = tmp_path / "notice.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    class FakeResponse:
        def __init__(self, payload: dict, headers: dict | None = None) -> None:
            self._payload = payload
            self.headers = headers or {}

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self._payload

    class FakeAsyncClient:
        def __init__(self) -> None:
            self.calls = []

        async def post(self, url: str, **kwargs):  # noqa: ANN003
            self.calls.append({"url": url, **kwargs})
            if url.endswith("/upload/v1beta/files"):
                return FakeResponse(
                    {},
                    headers={
                        "X-Goog-Upload-URL": (
                            "https://generativelanguage.googleapis.com"
                            "/upload/v1beta/files/session-1"
                        )
                    },
                )
            if url.endswith("/upload/v1beta/files/session-1"):
                return FakeResponse({"file": {"uri": "files/notice"}})
            return FakeResponse(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "text": "# Parsed Notice\n\n员工应在出差结束后 15 个自然日内提交差旅报销申请。"
                                    }
                                ]
                            }
                        }
                    ]
                }
            )

    config = AppConfig(
        config_dir=tmp_path,
        settings={"timeout": {"document_parser_seconds": 30}},
        models={
            "model_gateway": {
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                "api_key": "parser-secret",
            },
            "generation": {"default_alias": "gemini-3.1-pro-preview"},
            "document_parser": {
                "enabled": True,
                "default_alias": "gemini-3.1-pro-preview",
                "base_url": "https://generativelanguage.googleapis.com",
                "api_key": "parser-secret",
            },
        },
        prompts={},
    )
    client = DocumentParserClient(config)
    fake_http = FakeAsyncClient()
    client._client = fake_http  # noqa: SLF001

    text = await client.parse_file(pdf_path)

    assert "差旅报销申请" in text
    start_call = fake_http.calls[0]
    finalize_call = fake_http.calls[1]
    assert start_call["url"] == "https://generativelanguage.googleapis.com/upload/v1beta/files"
    assert start_call["headers"]["X-Goog-Upload-Protocol"] == "resumable"
    assert start_call["headers"]["X-Goog-Upload-Command"] == "start"
    assert "files" not in start_call
    assert finalize_call["url"] == (
        "https://generativelanguage.googleapis.com/upload/v1beta/files/session-1"
    )
    assert finalize_call["headers"]["X-Goog-Upload-Command"] == "upload, finalize"
    assert finalize_call["content"] == b"%PDF-1.4 fake"


@pytest.mark.asyncio
async def test_document_parser_uses_qwen_openai_compatible_file_part(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("DOCUMENT_PARSER_PROVIDER", "qwen")
    monkeypatch.setenv("DOCUMENT_PARSER_QWEN_FILE_PART_STYLE", "file")
    pdf_path = tmp_path / "notice.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "# Parsed Notice\n\n员工应在出差结束后 15 个自然日内提交差旅报销申请。"
                        }
                    }
                ]
            }

    class FakeAsyncClient:
        def __init__(self) -> None:
            self.calls = []

        async def post(self, url: str, **kwargs):  # noqa: ANN003
            self.calls.append({"url": url, **kwargs})
            return FakeResponse()

    config = AppConfig(
        config_dir=tmp_path,
        settings={"timeout": {"document_parser_seconds": 30}},
        models={
            "model_gateway": {"base_url": "", "api_key": ""},
            "document_parser": {
                "enabled": True,
                "provider": "qwen",
                "default_alias": "qwen-vl-plus",
                "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                "api_key": "dashscope-secret",
            },
        },
        prompts={},
    )
    client = DocumentParserClient(config)
    fake_http = FakeAsyncClient()
    client.openai_client._client = fake_http  # noqa: SLF001

    text = await client.parse_file(pdf_path)

    assert "差旅报销申请" in text
    call = fake_http.calls[0]
    assert call["url"] == (
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
    )
    assert call["headers"]["Authorization"] == "Bearer dashscope-secret"
    payload = call["json"]
    assert payload["model"] == "qwen-vl-plus"
    file_part = payload["messages"][0]["content"][1]
    assert file_part["type"] == "file"
    assert file_part["file"]["filename"] == "notice.pdf"
    assert file_part["file"]["file_data"] == (
        "data:application/pdf;base64,"
        + base64.b64encode(b"%PDF-1.4 fake").decode("ascii")
    )


@pytest.mark.asyncio
async def test_document_parser_uses_vllm_image_url_part(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DOCUMENT_PARSER_PROVIDER", "vllm")
    monkeypatch.delenv("DOCUMENT_PARSER_QWEN_FILE_PART_STYLE", raising=False)
    image_path = tmp_path / "receipt.png"
    image_path.write_bytes(b"fake-image")

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "# Parsed Receipt"}}]}

    class FakeAsyncClient:
        def __init__(self) -> None:
            self.calls = []

        async def post(self, url: str, **kwargs):  # noqa: ANN003
            self.calls.append({"url": url, **kwargs})
            return FakeResponse()

    config = AppConfig(
        config_dir=tmp_path,
        settings={"timeout": {"document_parser_seconds": 30}},
        models={
            "model_gateway": {"base_url": "", "api_key": ""},
            "document_parser": {
                "enabled": True,
                "provider": "vllm",
                "default_alias": "Qwen/Qwen2.5-VL-7B-Instruct",
                "base_url": "http://vllm:8000/v1",
                "api_key": "EMPTY",
            },
        },
        prompts={},
    )
    client = DocumentParserClient(config)
    fake_http = FakeAsyncClient()
    client.openai_client._client = fake_http  # noqa: SLF001

    text = await client.parse_file(image_path)

    assert text == "# Parsed Receipt"
    call = fake_http.calls[0]
    assert call["url"] == "http://vllm:8000/v1/chat/completions"
    image_part = call["json"]["messages"][0]["content"][1]
    assert image_part == {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64,"
            + base64.b64encode(b"fake-image").decode("ascii")
        },
    }
