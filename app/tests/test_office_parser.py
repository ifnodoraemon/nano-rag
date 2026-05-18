from __future__ import annotations

import zipfile

import pytest

from app.ingestion.office_parser import parse_office_document
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.structured_parser import StructuredDocumentParser
from app.schemas.structured import NodeType


def test_parse_csv_as_markdown_table(tmp_path) -> None:
    path = tmp_path / "prices.csv"
    path.write_text("region,level,price\nCN,pro,99\n", encoding="utf-8")

    parsed = parse_office_document(path)

    assert "| region | level | price |" in parsed
    assert "| CN | pro | 99 |" in parsed


def test_parse_docx_paragraphs_and_tables(tmp_path) -> None:
    path = tmp_path / "policy.docx"
    document_xml = """
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Leave Policy</w:t></w:r></w:p>
    <w:tbl>
      <w:tr><w:tc><w:p><w:r><w:t>Type</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Days</w:t></w:r></w:p></w:tc></w:tr>
      <w:tr><w:tc><w:p><w:r><w:t>PTO</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>10</w:t></w:r></w:p></w:tc></w:tr>
    </w:tbl>
  </w:body>
</w:document>
    """.strip()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)

    parsed = parse_office_document(path)

    assert "Leave Policy" in parsed
    assert "| Type | Days |" in parsed
    assert "| PTO | 10 |" in parsed


def test_parse_xlsx_shared_strings(tmp_path) -> None:
    path = tmp_path / "matrix.xlsx"
    shared = """
<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <si><t>region</t></si><si><t>price</t></si><si><t>CN</t></si>
</sst>
    """.strip()
    sheet = """
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row><c t="s"><v>0</v></c><c t="s"><v>1</v></c></row>
    <row><c t="s"><v>2</v></c><c><v>99</v></c></row>
  </sheetData>
</worksheet>
    """.strip()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/sharedStrings.xml", shared)
        archive.writestr("xl/worksheets/sheet1.xml", sheet)

    parsed = parse_office_document(path)

    assert "# Sheet 1" in parsed
    assert "| region | price |" in parsed
    assert "| CN | 99 |" in parsed


def test_parse_xlsx_preserves_sparse_cell_columns(tmp_path) -> None:
    path = tmp_path / "sparse.xlsx"
    sheet = """
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row><c r="A1"><v>left</v></c><c r="C1"><v>right</v></c></row>
    <row><c r="A2"><v>1</v></c><c r="C2"><v>3</v></c></row>
  </sheetData>
</worksheet>
    """.strip()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", sheet)

    parsed = parse_office_document(path)

    assert "| left |  | right |" in parsed
    assert "| 1 |  | 3 |" in parsed


def test_parse_json_array_as_table(tmp_path) -> None:
    path = tmp_path / "records.json"
    path.write_text('[{"region":"CN","price":99},{"region":"US","price":120}]', encoding="utf-8")

    parsed = parse_office_document(path)

    assert "| region | price |" in parsed
    assert "| CN | 99 |" in parsed


def test_parse_pptx_slide_text_and_table(tmp_path) -> None:
    path = tmp_path / "deck.pptx"
    slide = """
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
       xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
  <p:cSld><p:spTree>
    <p:sp><p:txBody><a:p><a:r><a:t>Q1 Review</a:t></a:r></a:p></p:txBody></p:sp>
    <p:graphicFrame><a:graphic><a:graphicData><a:tbl>
      <a:tr><a:tc><a:txBody><a:p><a:r><a:t>Metric</a:t></a:r></a:p></a:txBody></a:tc><a:tc><a:txBody><a:p><a:r><a:t>Value</a:t></a:r></a:p></a:txBody></a:tc></a:tr>
      <a:tr><a:tc><a:txBody><a:p><a:r><a:t>NPS</a:t></a:r></a:p></a:txBody></a:tc><a:tc><a:txBody><a:p><a:r><a:t>42</a:t></a:r></a:p></a:txBody></a:tc></a:tr>
    </a:tbl></a:graphicData></a:graphic></p:graphicFrame>
  </p:spTree></p:cSld>
</p:sld>
    """.strip()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ppt/slides/slide1.xml", slide)

    parsed = parse_office_document(path)

    assert "# Slide 1" in parsed
    assert "Q1 Review" in parsed
    assert "| Metric | Value |" in parsed
    assert "| NPS | 42 |" in parsed


@pytest.mark.asyncio
async def test_structured_parser_accepts_office_documents(tmp_path) -> None:
    path = tmp_path / "matrix.csv"
    path.write_text("region,price\nCN,99\n", encoding="utf-8")

    document = await StructuredDocumentParser().parse(
        path,
        doc_id="doc-1",
        kb_id="default",
        source_path="matrix.csv",
    )

    chunks = list(document.iter_leaves())
    assert chunks
    assert chunks[0].table is not None


@pytest.mark.asyncio
async def test_structured_parser_marks_clauses_and_definitions(tmp_path) -> None:
    path = tmp_path / "standard.md"
    path.write_text(
        "# 标准\n\n第十二条 适用范围\n\n术语A：用于测试的定义。\n",
        encoding="utf-8",
    )

    document = await StructuredDocumentParser().parse(
        path,
        doc_id="doc-std",
        kb_id="default",
        source_path="standard.md",
    )
    nodes = list(document.iter_nodes())

    clause = next(node for node in nodes if node.node_type == NodeType.CLAUSE)
    definition = next(node for node in nodes if node.node_type == NodeType.DEFINITION)
    assert clause.metadata["clause_id"] == "第十二条"
    assert definition.metadata["definition_term"] == "术语A"


@pytest.mark.asyncio
async def test_table_row_chunks_are_created_for_retrieval(tmp_path) -> None:
    path = tmp_path / "table.csv"
    path.write_text("region,price\nCN,99\nUS,120\n", encoding="utf-8")
    document = await StructuredDocumentParser().parse(
        path,
        doc_id="doc-table",
        kb_id="default",
        source_path="table.csv",
    )
    pipeline = IngestionPipeline.__new__(IngestionPipeline)
    chunks = pipeline._structured_document_to_chunks(  # noqa: SLF001
        document,
        source_path="table.csv",
        title="table",
        metadata={"kb_id": "default"},
    )

    row_chunks = [chunk for chunk in chunks if chunk.metadata.get("chunk_kind") == "table_row"]
    assert len(row_chunks) == 2
    assert row_chunks[0].text == "Table row: region=CN; price=99"
    assert row_chunks[0].metadata["chunk_strategy"] == "table_row"
    assert any(chunk.metadata.get("chunk_strategy") == "table_summary" for chunk in chunks)
