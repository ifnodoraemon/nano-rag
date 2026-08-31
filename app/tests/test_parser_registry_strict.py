"""No-degradation semantics for the parser registry and graph extractor."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.core.exceptions import ModelOutputError, ParsingError
from app.ingestion.graph_extractor import GraphExtractor
from app.ingestion.parser_registry import parse_content


class TextCapablePdfParser:
    """A configured model parser; the old code let pypdf-extractable text
    bypass it entirely."""

    enabled = True

    def supports(self, path: Path) -> bool:  # noqa: ARG002
        return True

    async def parse_file(self, path: Path) -> str:  # noqa: ARG002
        return "# Model-parsed PDF\n\nMultimodal extraction result."


class FailingModelParser:
    enabled = True

    def supports(self, path: Path) -> bool:  # noqa: ARG002
        return True

    async def parse_file(self, path: Path) -> str:  # noqa: ARG002
        raise ModelGatewayErrorStub("parser exploded")


class ModelGatewayErrorStub(Exception):
    pass


@pytest.mark.asyncio
async def test_configured_model_parser_wins_over_local_pdf_text(tmp_path) -> None:
    """P0 regression: a PDF with pypdf-extractable text must still go through
    the configured multimodal parser (pypdf loses tables/layout/scans)."""
    from pypdf import PdfWriter

    pdf_path = tmp_path / "real.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with pdf_path.open("wb") as handle:
        writer.write(handle)

    parsed = await parse_content(pdf_path, TextCapablePdfParser())

    assert parsed.parser_name == "multimodal-llm-tree"
    assert "Model-parsed PDF" in parsed.text


@pytest.mark.asyncio
async def test_model_parser_failure_is_not_swallowed(tmp_path) -> None:
    from pypdf import PdfWriter

    pdf_path = tmp_path / "real.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with pdf_path.open("wb") as handle:
        writer.write(handle)

    with pytest.raises(ModelGatewayErrorStub):
        await parse_content(pdf_path, FailingModelParser())


@pytest.mark.asyncio
async def test_non_utf8_text_file_fails_loudly(tmp_path) -> None:
    """errors='replace' silently produced mojibake; strict decode surfaces the
    bad file at ingest time instead of corrupting the corpus."""
    txt_path = tmp_path / "bad.txt"
    txt_path.write_bytes(b"valid utf-8 prefix \xff\xfe invalid bytes")

    with pytest.raises(ParsingError) as exc_info:
        await parse_content(txt_path, None)

    assert "not valid UTF-8" in str(exc_info.value)


@pytest.mark.asyncio
async def test_empty_pdf_without_parser_requires_multimodal_parser(tmp_path) -> None:
    """A scanned PDF yields no local text: raise with an actionable message
    instead of committing an empty document."""
    from pypdf import PdfWriter

    pdf_path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with pdf_path.open("wb") as handle:
        writer.write(handle)

    with pytest.raises(ParsingError) as exc_info:
        await parse_content(pdf_path, None)

    assert "multimodal document parser" in str(exc_info.value)


def _document_for_extraction():
    from app.tests.test_graph_index import _document

    return _document("doc-a", "a.md", "Evidence from A.")


class InvalidJsonClient:
    async def generate(self, messages):  # noqa: ANN001, ARG002
        return {"content": "not json at all (model rambled)"}


class NullRelationsClient:
    async def generate(self, messages):  # noqa: ANN001, ARG002
        return {"content": '{"entities": [], "relations": null}'}


class HallucinatedNodeClient:
    async def generate(self, messages):  # noqa: ANN001, ARG002
        return {
            "content": (
                '{"entities": [{"name": "Ghost", "entity_type": "concept",'
                ' "source_node_ids": ["doc-a:node:does-not-exist", "doc-a:node:1"]}],'
                ' "relations": []}'
            )
        }


@pytest.mark.asyncio
async def test_graph_extraction_invalid_json_raises() -> None:
    extractor = GraphExtractor(InvalidJsonClient())
    with pytest.raises(ModelOutputError):
        await extractor.extract(_document_for_extraction())


@pytest.mark.asyncio
async def test_graph_extraction_null_relations_raises() -> None:
    extractor = GraphExtractor(NullRelationsClient())
    with pytest.raises(ModelOutputError):
        await extractor.extract(_document_for_extraction())


@pytest.mark.asyncio
async def test_graph_extraction_drops_hallucinated_node_ids() -> None:
    """source_node_ids that are not real nodes would violate the graph store
    FK; they are dropped from the entity, real ids are kept."""
    extractor = GraphExtractor(HallucinatedNodeClient())
    graph = await extractor.extract(_document_for_extraction())
    ghost = next(entity for entity in graph.entities if entity.name == "Ghost")
    assert ghost.source_node_ids == ["doc-a:node:1"]


def test_structured_parser_provenance_is_not_fabricated() -> None:
    """page_number=1 / parser_confidence=1.0 were hardcoded fake provenance."""
    from app.ingestion.structured_parser import StructuredDocumentParser

    from app.schemas.structured import DocumentNode, NodeProvenance, NodeType

    parser = StructuredDocumentParser(None)
    blocks = parser._parse_markdown_blocks("# Title\n\nBody text here.")  # noqa: SLF001
    assert blocks, "fixture must produce blocks"
    parent = DocumentNode(
        node_id="doc:root",
        doc_id="doc",
        kb_id="default",
        node_type=NodeType.ROOT,
        title="doc",
        provenance=NodeProvenance(source_document_id="doc"),
    )
    node = parser._build_node(
        blocks[-1],
        doc_id="doc",
        kb_id="default",
        parent=parent,
        hierarchy_path=[],
        leaf_index=0,
    )
    assert node.provenance.page_number is None
    assert "parser_confidence" not in node.metadata
