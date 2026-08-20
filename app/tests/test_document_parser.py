import base64
import json
import zipfile
from pathlib import Path

import httpx
import pytest
from pypdf import PdfWriter

from app.core.exceptions import ModelGatewayError, ParsingError
from app.ingestion.parser_docling import parse_document
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.structured_parser import StructuredDocumentParser
from app.model_client.document_parser import DocumentParserClient
from app.core.config import AppConfig
from app.core.tracing import TracingManager
from app.schemas.chunk import Chunk


def _parsed_chunks(parsed_dir: Path) -> list[Chunk]:
    """Read the committed parsed artifact (the durable source of truth that
    replaced the in-memory vector repository) and hydrate its chunks.
    Each test ingests a single document, so there is exactly one artifact."""
    artifact_path = next(iter(sorted(parsed_dir.glob("*.json"))))
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    return [Chunk.model_validate(raw) for raw in artifact["chunks"]]


class FakeDocumentParser:
    def supports(self, path: Path) -> bool:  # noqa: ARG002
        return True

    async def parse_file(self, path: Path) -> str:  # noqa: ARG002
        return "# Parsed PDF\n\n徐套乡区片综合地价为 62000 元/亩。"


class EmptyDocumentParser:
    def supports(self, path: Path) -> bool:  # noqa: ARG002
        return True

    async def parse_file(self, path: Path) -> str:  # noqa: ARG002
        return ""


class DisabledStringDocumentParser(FakeDocumentParser):
    enabled = "false"

    async def parse_file(self, path: Path) -> str:  # noqa: ARG002
        raise AssertionError("disabled parser should not parse files")


class FakeGenerationClient:
    async def generate(self, messages):  # noqa: ANN001
        return {"content": '{"entities": [], "relations": []}'}


class FailingGenerationClient:
    async def generate(self, messages):  # noqa: ANN001, ARG002
        raise ModelGatewayError("graph extraction timeout")


@pytest.mark.asyncio
async def test_parse_document_uses_model_parser_for_pdf(tmp_path) -> None:
    pdf_path = tmp_path / "notice.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    text = await parse_document(pdf_path, FakeDocumentParser())

    assert "徐套乡区片综合地价" in text


@pytest.mark.asyncio
async def test_parse_document_respects_string_false_parser_enabled(tmp_path) -> None:
    pdf_path = tmp_path / "notice.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    with pytest.raises(ParsingError) as exc_info:
        await parse_document(pdf_path, DisabledStringDocumentParser())

    assert "configured document parser model" in str(exc_info.value)


@pytest.mark.asyncio
async def test_structured_parser_respects_string_false_parser_enabled(tmp_path) -> None:
    pdf_path = tmp_path / "notice.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    parser = StructuredDocumentParser(DisabledStringDocumentParser())

    with pytest.raises(ParsingError) as exc_info:
        await parser.parse(
            pdf_path,
            doc_id="doc",
            kb_id="default",
            source_path="/tmp/notice.pdf",
        )

    assert "requires a configured multimodal document parser" in str(exc_info.value)


@pytest.mark.asyncio
async def test_ingestion_pipeline_rejects_empty_parsed_content(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RAG_INGEST_ALLOWED_DIRS", str(tmp_path))
    monkeypatch.setenv("PARSED_OUTPUT_DIR", str(tmp_path / "parsed"))
    monkeypatch.setattr("app.ingestion.loader._cached_allowed_dirs", None)

    pdf_path = tmp_path / "notice.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    config = AppConfig(
        config_dir=tmp_path,
        settings={"chunk": {"size": 200, "overlap": 20}, "timeout": {"document_parser_seconds": 30}},
        models={"model_gateway": {"base_url": "", "api_key": ""}},
        prompts={},
    )
    pipeline = IngestionPipeline(
        config=config,
        generation_client=FakeGenerationClient(),
        tracing_manager=TracingManager("test-service", ""),
        document_parser=EmptyDocumentParser(),
    )

    with pytest.raises(ParsingError) as exc_info:
        await pipeline.run(str(pdf_path), kb_id="default")

    assert "returned empty content" in str(exc_info.value)


@pytest.mark.asyncio
async def test_ingestion_pipeline_keeps_document_when_graph_extraction_fails(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("RAG_INGEST_ALLOWED_DIRS", str(tmp_path))
    monkeypatch.setenv("PARSED_OUTPUT_DIR", str(tmp_path / "parsed"))
    monkeypatch.setattr("app.ingestion.loader._cached_allowed_dirs", None)
    doc_path = tmp_path / "matrix.md"
    doc_path.write_text("# Matrix\n\n| ID | Value |\n| --- | --- |\n| A-02 | pass |", encoding="utf-8")

    config = AppConfig(
        config_dir=tmp_path,
        settings={"timeout": {"document_parser_seconds": 30}},
        models={"model_gateway": {"base_url": "", "api_key": ""}},
        prompts={},
    )
    pipeline = IngestionPipeline(
        config=config,
        generation_client=FailingGenerationClient(),
        tracing_manager=TracingManager("test-service", ""),
    )

    response = await pipeline.run(str(doc_path), kb_id="default")

    assert response.documents == 1
    assert response.chunks == 2
    assert len(_parsed_chunks(tmp_path / "parsed")) == 2


@pytest.mark.asyncio
async def test_pdf_ingestion_adds_page_attachment_chunks(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RAG_INGEST_ALLOWED_DIRS", str(tmp_path))
    monkeypatch.setenv("PARSED_OUTPUT_DIR", str(tmp_path / "parsed"))
    monkeypatch.setattr("app.ingestion.loader._cached_allowed_dirs", None)
    pdf_path = tmp_path / "notice.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_blank_page(width=72, height=72)
    with pdf_path.open("wb") as handle:
        writer.write(handle)

    config = AppConfig(
        config_dir=tmp_path,
        settings={"timeout": {"document_parser_seconds": 30}},
        models={"model_gateway": {"base_url": "", "api_key": ""}},
        prompts={},
    )
    pipeline = IngestionPipeline(
        config=config,
        generation_client=FakeGenerationClient(),
        tracing_manager=TracingManager("test-service", ""),
        document_parser=FakeDocumentParser(),
    )

    response = await pipeline.run(str(pdf_path), kb_id="default")

    assert response.documents == 1
    assert response.chunks == 3
    chunks = _parsed_chunks(tmp_path / "parsed")
    page_chunks = [chunk for chunk in chunks if chunk.metadata.get("chunk_strategy") == "page_attachment"]
    assert len(page_chunks) == 2
    assert [chunk.metadata["page_number"] for chunk in page_chunks] == [1, 2]
    assert all(chunk.modality == "document" for chunk in page_chunks)
    assert all(Path(str(chunk.media_uri)).is_file() for chunk in page_chunks)


@pytest.mark.asyncio
async def test_pdf_ingestion_adds_rendered_page_image_chunks(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RAG_INGEST_ALLOWED_DIRS", str(tmp_path))
    monkeypatch.setenv("PARSED_OUTPUT_DIR", str(tmp_path / "parsed"))
    monkeypatch.setattr("app.ingestion.loader._cached_allowed_dirs", None)
    pdf_path = tmp_path / "notice.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with pdf_path.open("wb") as handle:
        writer.write(handle)

    def fake_render(path, output_dir):  # noqa: ANN001
        image_path = output_dir / "page-1.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"PNGDATA")
        return [(1, image_path)]

    monkeypatch.setattr("app.ingestion.pipeline._render_pdf_page_images", fake_render)
    config = AppConfig(
        config_dir=tmp_path,
        settings={"timeout": {"document_parser_seconds": 30}},
        models={"model_gateway": {"base_url": "", "api_key": ""}},
        prompts={},
    )
    pipeline = IngestionPipeline(
        config=config,
        generation_client=FakeGenerationClient(),
        tracing_manager=TracingManager("test-service", ""),
        document_parser=FakeDocumentParser(),
    )

    response = await pipeline.run(str(pdf_path), kb_id="default")

    assert response.chunks == 3
    chunks = _parsed_chunks(tmp_path / "parsed")
    rendered = [chunk for chunk in chunks if chunk.metadata.get("chunk_strategy") == "rendered_page_image"]
    assert len(rendered) == 1
    assert rendered[0].modality == "image"
    assert rendered[0].metadata["page_number"] == 1
    assert Path(str(rendered[0].media_uri)).read_bytes() == b"PNGDATA"


@pytest.mark.asyncio
async def test_pptx_ingestion_extracts_embedded_images(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RAG_INGEST_ALLOWED_DIRS", str(tmp_path))
    monkeypatch.setenv("PARSED_OUTPUT_DIR", str(tmp_path / "parsed"))
    monkeypatch.setattr("app.ingestion.loader._cached_allowed_dirs", None)
    pptx_path = tmp_path / "deck.pptx"
    slide = """
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
       xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
  <p:cSld><p:spTree>
    <p:sp><p:txBody><a:p><a:r><a:t>Visual Report</a:t></a:r></a:p></p:txBody></p:sp>
  </p:spTree></p:cSld>
</p:sld>
    """.strip()
    with zipfile.ZipFile(pptx_path, "w") as archive:
        archive.writestr("ppt/slides/slide1.xml", slide)
        archive.writestr("ppt/media/image1.png", b"PNGDATA")

    config = AppConfig(
        config_dir=tmp_path,
        settings={"timeout": {"document_parser_seconds": 30}},
        models={"model_gateway": {"base_url": "", "api_key": ""}},
        prompts={},
    )
    pipeline = IngestionPipeline(
        config=config,
        generation_client=FakeGenerationClient(),
        tracing_manager=TracingManager("test-service", ""),
    )

    response = await pipeline.run(str(pptx_path), kb_id="default")

    assert response.documents == 1
    chunks = _parsed_chunks(tmp_path / "parsed")
    assert any(chunk.metadata.get("chunk_strategy") == "document_attachment" for chunk in chunks)
    embedded = [chunk for chunk in chunks if chunk.metadata.get("chunk_strategy") == "embedded_image"]
    assert len(embedded) == 1
    assert Path(str(embedded[0].media_uri)).read_bytes() == b"PNGDATA"


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
async def test_document_parser_batches_large_pdf_pages(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MODEL_GATEWAY_MODE", "live")
    monkeypatch.setenv("DOCUMENT_PARSER_PDF_PAGE_BATCH_SIZE", "2")
    pdf_path = tmp_path / "standard.pdf"
    writer = PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=72, height=72)
    with pdf_path.open("wb") as handle:
        writer.write(handle)

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
            self.upload_count = 0
            self.generate_count = 0

        async def post(self, url: str, **kwargs):  # noqa: ANN003
            self.calls.append({"url": url, **kwargs})
            if url.endswith("/upload/v1beta/files"):
                self.upload_count += 1
                return FakeResponse(
                    {},
                    headers={
                        "X-Goog-Upload-URL": (
                            "https://generativelanguage.googleapis.com"
                            f"/upload/v1beta/files/session-{self.upload_count}"
                        )
                    },
                )
            if "/upload/v1beta/files/session-" in url:
                return FakeResponse({"file": {"uri": f"files/batch-{self.upload_count}"}})
            self.generate_count += 1
            return FakeResponse(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": f"batch {self.generate_count} markdown"}]
                            }
                        }
                    ]
                }
            )

    config = AppConfig(
        config_dir=tmp_path,
        settings={"timeout": {"document_parser_seconds": 30}},
        models={
            "document_parser": {
                "enabled": True,
                "provider": "gemini",
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

    assert "# Pages 1-2" in text
    assert "# Pages 3-3" in text
    assert fake_http.upload_count == 2
    assert fake_http.generate_count == 2


@pytest.mark.asyncio
async def test_document_parser_keeps_successful_pdf_batches(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("MODEL_GATEWAY_MODE", "live")
    monkeypatch.setenv("DOCUMENT_PARSER_PDF_PAGE_BATCH_SIZE", "1")
    pdf_path = tmp_path / "standard.pdf"
    writer = PdfWriter()
    for _ in range(2):
        writer.add_blank_page(width=72, height=72)
    with pdf_path.open("wb") as handle:
        writer.write(handle)

    class FakeResponse:
        def __init__(self, payload: dict, fail: bool = False, headers: dict | None = None) -> None:
            self._payload = payload
            self.fail = fail
            self.headers = headers or {}
            self.text = "upstream timeout"

        def raise_for_status(self) -> None:
            if self.fail:
                raise httpx.TimeoutException("timeout")

        def json(self) -> dict:
            return self._payload

    class FakeAsyncClient:
        def __init__(self) -> None:
            self.upload_count = 0
            self.generate_count = 0

        async def post(self, url: str, **kwargs):  # noqa: ANN003, ARG002
            if url.endswith("/upload/v1beta/files"):
                self.upload_count += 1
                return FakeResponse(
                    {},
                    headers={
                        "X-Goog-Upload-URL": (
                            "https://generativelanguage.googleapis.com"
                            f"/upload/v1beta/files/session-{self.upload_count}"
                        )
                    },
                )
            if "/upload/v1beta/files/session-" in url:
                return FakeResponse({"file": {"uri": f"files/batch-{self.upload_count}"}})
            self.generate_count += 1
            if self.generate_count == 2:
                raise httpx.TimeoutException("timeout")
            return FakeResponse(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [{"text": "first page markdown"}]
                            }
                        }
                    ]
                }
            )

    config = AppConfig(
        config_dir=tmp_path,
        settings={"timeout": {"document_parser_seconds": 30}},
        models={
            "document_parser": {
                "enabled": True,
                "provider": "gemini",
                "default_alias": "gemini-3.1-pro-preview",
                "base_url": "https://generativelanguage.googleapis.com",
                "api_key": "parser-secret",
            },
        },
        prompts={},
    )
    client = DocumentParserClient(config)
    client._client = FakeAsyncClient()  # noqa: SLF001

    text = await client.parse_file(pdf_path)

    assert "first page markdown" in text
    assert "# Parser warnings" in text
    assert "Pages 2-2" in text


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
