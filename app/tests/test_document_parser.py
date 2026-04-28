import json
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
            "generation": {"default_alias": "gemini-2.5-flash"},
            "document_parser": {"enabled": True},
        },
        prompts={},
    )

    client = DocumentParserClient(config)

    assert client.base_url == "https://generativelanguage.googleapis.com"


@pytest.mark.asyncio
async def test_document_parser_uses_bifrost_multipart_upload(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MODEL_GATEWAY_MODE", "live")
    pdf_path = tmp_path / "notice.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    class FakeResponse:
        def __init__(self, payload: dict) -> None:
            self._payload = payload

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
                "base_url": "http://bifrost:8080/v1",
                "api_key": "bifrost-local",
            },
            "generation": {"default_alias": "google/gemini-2.5-flash"},
            "document_parser": {
                "enabled": True,
                "default_alias": "gemini-2.5-flash",
                "base_url": "http://bifrost:8080/genai",
                "api_key": "bifrost-local",
            },
        },
        prompts={},
    )
    client = DocumentParserClient(config)
    fake_http = FakeAsyncClient()
    client._client = fake_http  # noqa: SLF001

    text = await client.parse_file(pdf_path)

    assert "差旅报销申请" in text
    upload_call = fake_http.calls[0]
    assert upload_call["url"] == "http://bifrost:8080/genai/upload/v1beta/files"
    assert "files" in upload_call
    assert "X-Goog-Upload-Protocol" not in upload_call["headers"]
    metadata = json.loads(upload_call["files"]["metadata"][1])
    assert metadata == {"file": {"displayName": "notice.pdf"}}
    assert upload_call["files"]["file"] == (
        "notice.pdf",
        b"%PDF-1.4 fake",
        "application/pdf",
    )
