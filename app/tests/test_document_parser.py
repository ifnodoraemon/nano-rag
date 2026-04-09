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
