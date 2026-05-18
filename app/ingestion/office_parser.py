from __future__ import annotations

import csv
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from app.core.exceptions import ParsingError

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
}


def parse_office_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return _parse_csv(path, delimiter="\t" if suffix == ".tsv" else ",")
    if suffix == ".json":
        return _parse_json(path)
    if suffix == ".jsonl":
        return _parse_jsonl(path)
    if suffix in {".yaml", ".yml"}:
        return _parse_yaml(path)
    if suffix == ".xml":
        return _parse_xml(path)
    if suffix == ".docx":
        return _parse_docx(path)
    if suffix == ".xlsx":
        return _parse_xlsx(path)
    if suffix == ".pptx":
        return _parse_pptx(path)
    raise ParsingError(f"Unsupported office document type: {path.suffix}")


def _parse_csv(path: Path, *, delimiter: str = ",") -> str:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    rows = list(csv.reader(text.splitlines(), delimiter=delimiter))
    if not rows:
        return ""
    return _rows_to_markdown(rows)


def _parse_json(path: Path) -> str:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ParsingError(f"failed to parse JSON from {path.name}: {exc}") from exc
    return _json_to_markdown(payload)


def _parse_jsonl(path: Path) -> str:
    rows: list[object] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8-sig", errors="replace").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ParsingError(
                f"failed to parse JSONL line {line_number} from {path.name}: {exc}"
            ) from exc
    return _json_to_markdown(rows)


def _parse_yaml(path: Path) -> str:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - requirements include pyyaml.
        raise ParsingError("PyYAML is required to parse YAML documents") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8-sig", errors="replace"))
    return _json_to_markdown(payload)


def _parse_xml(path: Path) -> str:
    try:
        root = ElementTree.parse(path).getroot()
    except (OSError, ElementTree.ParseError) as exc:
        raise ParsingError(f"failed to parse XML from {path.name}: {exc}") from exc
    lines: list[str] = []
    _xml_to_lines(root, lines, depth=0)
    return "\n".join(lines).strip()


def _parse_docx(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            document_xml = archive.read("word/document.xml")
    except (KeyError, OSError, zipfile.BadZipFile) as exc:
        raise ParsingError(f"failed to read DOCX document.xml from {path.name}: {exc}") from exc
    root = ElementTree.fromstring(document_xml)
    blocks: list[str] = []
    for child in root.findall(".//w:body/*", NS):
        if child.tag.endswith("}p"):
            paragraph = _paragraph_text(child)
            if paragraph:
                blocks.append(paragraph)
        elif child.tag.endswith("}tbl"):
            rows = [
                [_cell_text(cell) for cell in row.findall("./w:tc", NS)]
                for row in child.findall("./w:tr", NS)
            ]
            if rows:
                blocks.append(_rows_to_markdown(rows))
    return "\n\n".join(blocks).strip()


def _parse_xlsx(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            shared_strings = _xlsx_shared_strings(archive)
            sheet_names = [
                name
                for name in archive.namelist()
                if re.match(r"xl/worksheets/sheet\d+\.xml$", name)
            ]
            sections = []
            for index, name in enumerate(sorted(sheet_names), start=1):
                rows = _xlsx_sheet_rows(archive.read(name), shared_strings)
                if rows:
                    sections.append(f"# Sheet {index}\n\n{_rows_to_markdown(rows)}")
    except (OSError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
        raise ParsingError(f"failed to read XLSX workbook from {path.name}: {exc}") from exc
    return "\n\n".join(sections).strip()


def _parse_pptx(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            slide_names = [
                name
                for name in archive.namelist()
                if re.match(r"ppt/slides/slide\d+\.xml$", name)
            ]
            sections = []
            for index, name in enumerate(sorted(slide_names, key=_natural_key), start=1):
                slide = _pptx_slide_markdown(archive.read(name), index)
                if slide:
                    sections.append(slide)
    except (OSError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
        raise ParsingError(f"failed to read PPTX deck from {path.name}: {exc}") from exc
    return "\n\n".join(sections).strip()


def _paragraph_text(node: ElementTree.Element) -> str:
    return "".join(text.text or "" for text in node.findall(".//w:t", NS)).strip()


def _cell_text(node: ElementTree.Element) -> str:
    parts = [_paragraph_text(paragraph) for paragraph in node.findall(".//w:p", NS)]
    return " ".join(part for part in parts if part).strip()


def _xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        payload = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ElementTree.fromstring(payload)
    values: list[str] = []
    for item in root.findall("./s:si", NS):
        values.append("".join(text.text or "" for text in item.findall(".//s:t", NS)))
    return values


def _xlsx_sheet_rows(payload: bytes, shared_strings: list[str]) -> list[list[str]]:
    root = ElementTree.fromstring(payload)
    rows: list[list[str]] = []
    for row in root.findall(".//s:sheetData/s:row", NS):
        values_by_col: dict[int, str] = {}
        fallback_col = 0
        for cell in row.findall("./s:c", NS):
            col = _xlsx_cell_col(cell.attrib.get("r")) or fallback_col
            values_by_col[col] = _xlsx_cell_value(cell, shared_strings)
            fallback_col = col + 1
        if any(value.strip() for value in values_by_col.values()):
            width = max(values_by_col) + 1 if values_by_col else 0
            rows.append([values_by_col.get(index, "") for index in range(width)])
    return rows


def _xlsx_cell_value(cell: ElementTree.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    value_node = cell.find("./s:v", NS)
    inline_text = "".join(text.text or "" for text in cell.findall(".//s:is//s:t", NS))
    if inline_text:
        return inline_text.strip()
    raw = (value_node.text if value_node is not None else "") or ""
    if cell_type == "s":
        try:
            return shared_strings[int(raw)]
        except (ValueError, IndexError):
            return raw
    return raw.strip()


def _xlsx_cell_col(ref: str | None) -> int | None:
    if not ref:
        return None
    letters = "".join(char for char in ref if char.isalpha()).upper()
    if not letters:
        return None
    value = 0
    for char in letters:
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value - 1


def _pptx_slide_markdown(payload: bytes, slide_index: int) -> str:
    root = ElementTree.fromstring(payload)
    tables = root.findall(".//a:tbl", NS)
    table_nodes = {id(node) for table in tables for node in table.iter()}
    text_blocks: list[str] = []
    for shape in root.findall(".//p:sp", NS):
        if any(id(parent) in table_nodes for parent in shape.iter()):
            continue
        text = " ".join(
            (node.text or "").strip()
            for node in shape.findall(".//a:t", NS)
            if (node.text or "").strip()
        ).strip()
        if text:
            text_blocks.append(text)
    parts = [f"# Slide {slide_index}"]
    parts.extend(text_blocks)
    for table in tables:
        rows = [
            [
                " ".join(
                    (text.text or "").strip()
                    for text in cell.findall(".//a:t", NS)
                    if (text.text or "").strip()
                ).strip()
                for cell in row.findall("./a:tc", NS)
            ]
            for row in table.findall("./a:tr", NS)
        ]
        if rows:
            parts.append(_rows_to_markdown(rows))
    return "\n\n".join(part for part in parts if part).strip()


def _json_to_markdown(value: object) -> str:
    if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        keys: list[str] = []
        for item in value:
            for key in item:
                key_text = str(key)
                if key_text not in keys:
                    keys.append(key_text)
        rows = [keys]
        for item in value:
            rows.append([_scalar_preview(item.get(key, "")) for key in keys])
        return _rows_to_markdown(rows)
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"## {key}\n\n{_json_to_markdown(item)}")
            else:
                lines.append(f"- {key}: {_scalar_preview(item)}")
        return "\n\n".join(lines).strip()
    return _scalar_preview(value)


def _xml_to_lines(node: ElementTree.Element, lines: list[str], *, depth: int) -> None:
    tag = node.tag.rsplit("}", 1)[-1]
    text = " ".join((node.text or "").split())
    attrs = " ".join(f'{key}="{value}"' for key, value in sorted(node.attrib.items()))
    prefix = "  " * depth
    if text:
        lines.append(f"{prefix}- {tag}{' ' + attrs if attrs else ''}: {text}")
    elif attrs:
        lines.append(f"{prefix}- {tag} {attrs}")
    for child in node:
        _xml_to_lines(child, lines, depth=depth + 1)


def _scalar_preview(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header = normalized[0]
    body = normalized[1:]
    lines = [
        "| " + " | ".join(_escape_cell(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(_escape_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


def _escape_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _natural_key(value: str) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]
