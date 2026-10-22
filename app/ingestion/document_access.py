"""Format-agnostic document access, so the model faces three verbs instead of N formats.

The temptation is one tool set per format: read_docx_paragraphs, read_pdf_pages,
read_xlsx_sheets, read_pptx_slides, read_image... That set grows with every format a
user happens to have, and each tool embeds an opinion about how that format is
organised.

This module collapses all of them into one abstraction. Every document, whatever its
container, is exposed as an ordered list of *parts*, and a part is either text or
pixels - the only two things a model can read:

    inspect(path)            -> {"kind", "parts": [{id, kind, size_hint}, ...]}
    read_text(path, part)    -> the text of a part
    read_image(path, part)   -> PNG bytes for a part the model wants to see

An adapter per container (OOXML, PDF, image, plain text, HTML) fills that list in. A
new format means a new adapter, NOT a new tool: the model's protocol never changes,
and nothing here decides what a heading is.

Adapters only decode containers. They never classify content, and they never read a page
for the model: a PDF page and an image are handed over AS PIXELS, because reading them is
exactly the visual work a multimodal model is for. Local text extraction (pypdf text
layers, OCR) would be the local parsing this project removed, and it silently drops
tables, columns and scans. DOCX/XLSX/PPTX are different only in that their content
already IS text inside a zip - unpacking that is decoding, not interpretation.
"""

from __future__ import annotations

import base64
import re
import subprocess
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from app.core.exceptions import ParsingError

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
S = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"

MAX_PART_CHARS = 60_000
MAX_PARTS = 500
MAX_IMAGE_BYTES = 8_000_000


@dataclass
class Part:
    """One addressable unit of a document: text or pixels."""

    part_id: str
    kind: str  # "text" | "image"
    label: str = ""
    text: str | None = None
    image_bytes: bytes | None = None
    mime: str = "image/png"
    extra: dict[str, Any] = field(default_factory=dict)

    def size_hint(self) -> int:
        if self.kind == "text":
            return len(self.text or "")
        return len(self.image_bytes or b"")


@dataclass
class Document:
    kind: str
    parts: list[Part]

    def manifest(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "part_count": len(self.parts),
            "parts": [
                {
                    "id": part.part_id,
                    "kind": part.kind,
                    "label": part.label,
                    "size_hint": part.size_hint(),
                    **({"extra": part.extra} if part.extra else {}),
                }
                for part in self.parts
            ],
        }


# --------------------------------------------------------------------------- adapters


def _ooxml_paragraphs(xml: bytes) -> list[tuple[str, str]]:
    """(style name, text) in document order - raw facts, no interpretation."""
    root = ET.fromstring(xml)
    paragraphs: list[tuple[str, str]] = []
    for node in root.iter(f"{{{W}}}p"):
        style = ""
        properties = node.find(f"{{{W}}}pPr")
        if properties is not None:
            style_node = properties.find(f"{{{W}}}pStyle")
            if style_node is not None:
                style = style_node.get(f"{{{W}}}val") or ""
        text = "".join(t.text or "" for t in node.iter(f"{{{W}}}t")).strip()
        if text:
            paragraphs.append((style, text))
    return paragraphs


def _ooxml_tables(xml: bytes) -> list[list[list[str]]]:
    root = ET.fromstring(xml)
    tables: list[list[list[str]]] = []
    for table in root.iter(f"{{{W}}}tbl"):
        rows: list[list[str]] = []
        for row in table.iter(f"{{{W}}}tr"):
            cells = [
                "".join(t.text or "" for t in cell.iter(f"{{{W}}}t")).strip()
                for cell in row.iter(f"{{{W}}}tc")
            ]
            if any(cells):
                rows.append(cells)
        if rows:
            tables.append(rows)
    return tables


def _docx_parts(path: Path) -> list[Part]:
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
        try:
            styles_xml = archive.read("word/styles.xml")
        except KeyError:
            styles_xml = b""
    paragraphs = _ooxml_paragraphs(xml)
    tables = _ooxml_tables(xml)
    style_names = _style_names(styles_xml) if styles_xml else {}

    parts: list[Part] = []
    # Paragraphs are exposed in one part per contiguous run, keeping the numbering
    # and the declared style next to the text so the model can reason about depth.
    buffer: list[str] = []
    buffer_len = 0
    def flush() -> None:
        nonlocal buffer, buffer_len
        if buffer:
            parts.append(
                Part(
                    part_id=f"p{len(parts) + 1}",
                    kind="text",
                    label="paragraphs",
                    text="\n".join(buffer),
                )
            )
            buffer, buffer_len = [], 0

    for style, text in paragraphs:
        style_label = style_names.get(style, style)
        rendered = f"[style:{style_label}] {text}" if style_label else text
        if buffer_len + len(rendered) > MAX_PART_CHARS:
            flush()
        buffer.append(rendered)
        buffer_len += len(rendered) + 1
    flush()

    for index, table in enumerate(tables, start=1):
        rows = "\n".join(" | ".join(cells) for cells in table)
        parts.append(
            Part(
                part_id=f"t{index}",
                kind="text",
                label=f"table-{index}",
                text=rows,
            )
        )
    if not parts:
        raise ParsingError(f"{path.name} contains no readable text")
    return parts


def _pptx_parts(path: Path) -> list[Part]:
    parts: list[Part] = []
    with zipfile.ZipFile(path) as archive:
        slides = sorted(
            name
            for name in archive.namelist()
            if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
        )
        for index, name in enumerate(slides, start=1):
            root = ET.fromstring(archive.read(name))
            texts = [
                (node.text or "").strip()
                for node in root.iter(f"{{{A}}}t")
                if (node.text or "").strip()
            ]
            if texts:
                parts.append(
                    Part(
                        part_id=f"s{index}",
                        kind="text",
                        label=f"slide-{index}",
                        text="\n".join(texts),
                    )
                )
    if not parts:
        raise ParsingError(f"{path.name} contains no readable text")
    return parts


def _xlsx_parts(path: Path) -> list[Part]:
    parts: list[Part] = []
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.iter(f"{{{S}}}si"):
                shared.append("".join(t.text or "" for t in item.iter(f"{{{S}}}t")))
        sheets = sorted(n for n in names if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n))
        for index, name in enumerate(sheets, start=1):
            root = ET.fromstring(archive.read(name))
            rows: list[str] = []
            for row in root.iter(f"{{{S}}}row"):
                cells: list[str] = []
                for cell in row.iter(f"{{{S}}}c"):
                    value_node = cell.find(f"{{{S}}}v")
                    value = value_node.text if value_node is not None else ""
                    if cell.get("t") == "s" and value and value.isdigit():
                        position = int(value)
                        value = shared[position] if position < len(shared) else value
                    if value:
                        cells.append(str(value))
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                parts.append(
                    Part(
                        part_id=f"w{index}",
                        kind="text",
                        label=f"sheet-{index}",
                        text="\n".join(rows),
                    )
                )
    if not parts:
        raise ParsingError(f"{path.name} contains no readable cells")
    return parts


def _pdf_parts(path: Path, max_pages: int = MAX_PARTS) -> list[Part]:
    """Every PDF page becomes an image part.

    pypdf is used ONLY to learn how many pages exist - a container fact. No text is
    extracted locally: reading a page is visual work, and the model is the thing that
    can do it. Extracting a text layer here would be the local parsing this project
    deliberately removed, and it silently loses tables, columns and scans.
    """
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(str(path))
        page_count = len(reader.pages)
    except (PdfReadError, OSError, ValueError) as exc:
        raise ParsingError(f"cannot open PDF {path.name}: {exc}") from exc
    if page_count == 0:
        raise ParsingError(f"PDF {path.name} has no pages")
    return [
        Part(
            part_id=f"pg{index + 1}",
            kind="image",
            label=f"page-{index + 1}",
            extra={"page": index + 1},
        )
        for index in range(min(page_count, max_pages))
    ]


def _image_parts(path: Path) -> list[Part]:
    payload = path.read_bytes()
    if len(payload) > MAX_IMAGE_BYTES:
        raise ParsingError(
            f"{path.name} is {len(payload)} bytes, above the {MAX_IMAGE_BYTES}-byte "
            "image limit"
        )
    suffix = path.suffix.lower().lstrip(".")
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "webp": "image/webp", "gif": "image/gif", "bmp": "image/bmp"}.get(
        suffix, "image/png"
    )
    return [
        Part(part_id="img1", kind="image", label="image", image_bytes=payload, mime=mime)
    ]


def _text_parts(path: Path) -> list[Part]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ParsingError(
            f"{path.name} is not valid UTF-8 text (decode error at byte {exc.start})"
        ) from exc
    chunks = [
        text[i : i + MAX_PART_CHARS] for i in range(0, len(text), MAX_PART_CHARS)
    ] or [""]
    return [
        Part(
            part_id=f"text{index + 1}",
            kind="text",
            label=f"chunk-{index + 1}",
            text=chunk,
        )
        for index, chunk in enumerate(chunks)
    ]


def _html_parts(path: Path) -> list[Part]:
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ParsingError(
            f"{path.name} is not valid UTF-8 text (decode error at byte {exc.start})"
        ) from exc
    # Present the markup itself rather than stripping it: the tags carry the heading
    # structure, and interpreting them is the model's job, not ours.
    return [
        Part(part_id=f"html{i // MAX_PART_CHARS + 1}", kind="text", label="markup",
             text=raw[i : i + MAX_PART_CHARS])
        for i in range(0, max(len(raw), 1), MAX_PART_CHARS)
    ]


def _render_pdf_page(path: Path, page_number: int) -> bytes:
    """Render one page to PNG using poppler (already a runtime dependency)."""
    import shutil
    import tempfile

    renderer = shutil.which("pdftoppm")
    if not renderer:
        raise ParsingError(
            "pdftoppm is required to read a PDF page that has no text layer; "
            "install poppler-utils"
        )
    with tempfile.TemporaryDirectory(prefix="nano-rag-page-") as tmp:
        prefix = Path(tmp) / "page"
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [renderer, "-png", "-r", "150", "-f", str(page_number), "-l", str(page_number),
             str(path), str(prefix)],
            capture_output=True,
            timeout=120,
            check=False,
        )
        if result.returncode != 0:
            raise ParsingError(
                f"pdftoppm failed on page {page_number} of {path.name}: "
                f"{result.stderr.decode(errors='replace')[:200]}"
            )
        produced = sorted(Path(tmp).glob("page*.png"))
        if not produced:
            raise ParsingError(f"pdftoppm produced no image for page {page_number}")
        return produced[0].read_bytes()


def _style_names(styles_xml: bytes) -> dict[str, str]:
    try:
        root = ET.fromstring(styles_xml)
    except ET.ParseError:
        return {}
    names: dict[str, str] = {}
    for style in root.iter(f"{{{W}}}style"):
        style_id = style.get(f"{{{W}}}styleId") or ""
        name_node = style.find(f"{{{W}}}name")
        name = name_node.get(f"{{{W}}}val") if name_node is not None else ""
        if style_id:
            names[style_id] = name or style_id
    return names


# ------------------------------------------------------------------------ dispatcher

_OOXML_DISPATCH = {
    ".docx": _docx_parts,
    ".docm": _docx_parts,
    ".pptx": _pptx_parts,
    ".xlsx": _xlsx_parts,
}


def load_document(path: Path) -> Document:
    """Open any supported container and expose it as parts. Read-only."""
    suffix = path.suffix.lower()
    if suffix in _OOXML_DISPATCH:
        return Document(kind=suffix.lstrip("."), parts=_OOXML_DISPATCH[suffix](path))
    if suffix == ".pdf":
        return Document(kind="pdf", parts=_pdf_parts(path))
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}:
        return Document(kind="image", parts=_image_parts(path))
    if suffix in {".html", ".htm"}:
        return Document(kind="html", parts=_html_parts(path))
    return Document(kind="text", parts=_text_parts(path))


class DocumentAccess:
    """Confines reads to the ingest roots and serves the three verbs."""

    def __init__(self, allowed_roots: list[Path]) -> None:
        self.allowed_roots = [root.resolve() for root in allowed_roots]
        self._cache: dict[str, Document] = {}

    def _document(self, raw_path: str) -> tuple[Path, Document]:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        resolved = path.resolve()
        if not any(
            resolved == root or root in resolved.parents for root in self.allowed_roots
        ):
            raise ParsingError(
                f"{raw_path} is outside the configured ingest roots "
                f"({', '.join(str(root) for root in self.allowed_roots)})"
            )
        if not resolved.exists():
            raise ParsingError(f"document not found: {raw_path}")
        key = str(resolved)
        if key not in self._cache:
            self._cache[key] = load_document(resolved)
        return resolved, self._cache[key]

    def _part(self, document: Document, part_id: str) -> Part:
        for part in document.parts:
            if part.part_id == part_id:
                return part
        known = ", ".join(part.part_id for part in document.parts[:20])
        raise ParsingError(
            f"no part named {part_id!r}; known parts: {known}"
        )

    def inspect(self, path: str) -> dict[str, Any]:
        _resolved, document = self._document(path)
        return document.manifest()

    def read_text(self, path: str, part_id: str) -> dict[str, Any]:
        _resolved, document = self._document(path)
        part = self._part(document, part_id)
        if part.kind != "text":
            return {
                "part_id": part_id,
                "kind": part.kind,
                "note": "this part is an image; request it with read_image",
            }
        return {
            "part_id": part_id,
            "label": part.label,
            "chars": len(part.text or ""),
            "text": (part.text or "")[:MAX_PART_CHARS],
        }

    def read_image(self, path: str, part_id: str) -> dict[str, Any]:
        resolved, document = self._document(path)
        part = self._part(document, part_id)
        if part.kind == "image" and part.image_bytes is None and document.kind == "pdf":
            page = int(part_id.removeprefix("pg"))
            payload = _render_pdf_page(resolved, page)
        elif part.image_bytes is not None:
            payload = part.image_bytes
        else:
            return {
                "part_id": part_id,
                "kind": part.kind,
                "note": "this part is text; request it with read_text",
            }
        return {
            "part_id": part_id,
            "mime": part.mime,
            "base64": base64.b64encode(payload).decode("ascii"),
            "bytes": len(payload),
        }


# Tool schemas: THREE verbs, independent of container format.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "inspect_document",
            "description": (
                "Describe a document: its container kind and the ordered parts it "
                "exposes. Each part is either text or image. Call this first."
            ),
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_text",
            "description": (
                "Return the text of one part (paragraphs, a table, a slide, a sheet, "
                "a PDF page). DOCX paragraphs carry their declared style name. "
                "Read-only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "part_id": {"type": "string"},
                },
                "required": ["path", "part_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_image",
            "description": (
                "Return one part as an image (base64) so you can look at it - used for "
                "scanned PDF pages and picture files. Read-only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "part_id": {"type": "string"},
                },
                "required": ["path", "part_id"],
            },
        },
    },
]
