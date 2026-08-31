from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup

from app.core.exceptions import ParsingError
from app.ingestion.office_parser import parse_office_document
from app.model_client.document_parser import DocumentParserClient
from app.utils.text import parse_bool_env


PLAIN_TEXT_SUFFIXES = {
    ".c",
    ".cpp",
    ".cs",
    ".css",
    ".go",
    ".java",
    ".js",
    ".jsx",
    ".log",
    ".md",
    ".markdown",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".sql",
    ".tex",
    ".ts",
    ".tsx",
    ".txt",
}
HTML_SUFFIXES = {".html", ".htm"}
STRUCTURED_TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".tsv",
    ".xml",
    ".yaml",
    ".yml",
}
OOXML_SUFFIXES = {".docx", ".pptx", ".xlsx"}
MODEL_PARSER_SUFFIXES = {".doc", ".pdf", ".ppt", ".xls"}


@dataclass(frozen=True)
class ParsedContent:
    text: str
    parser_name: str
    metadata: dict[str, object] = field(default_factory=dict)


def supported_text_suffixes() -> set[str]:
    return PLAIN_TEXT_SUFFIXES | HTML_SUFFIXES | STRUCTURED_TEXT_SUFFIXES | OOXML_SUFFIXES | MODEL_PARSER_SUFFIXES


def model_parser_available(
    document_parser: DocumentParserClient | None, path: Path
) -> bool:
    """True when a configured, enabled model parser can take this file."""
    if document_parser is None:
        return False
    if not _parser_enabled(getattr(document_parser, "enabled", True)):
        return False
    return document_parser.supports(path)


async def parse_content(
    path: Path,
    document_parser: DocumentParserClient | None = None,
) -> ParsedContent:
    suffix = path.suffix.lower()
    # A configured, enabled multimodal parser wins for every suffix it
    # supports: local extraction (pypdf/office) loses layout, tables and
    # images, so it may only run when no model parser is configured.
    if model_parser_available(document_parser, path):
        parsed = await document_parser.parse_file(path)
        if parsed.strip():
            return ParsedContent(text=parsed, parser_name="multimodal-llm-tree")
        raise ParsingError(f"Document parser returned empty content for {path.name}")
    if suffix in PLAIN_TEXT_SUFFIXES:
        text = await asyncio.to_thread(_read_text_strict, path)
        return ParsedContent(text=text, parser_name="plain-text-tree")
    if suffix in HTML_SUFFIXES:
        html = await asyncio.to_thread(_read_text_strict, path)
        text = await asyncio.to_thread(_html_to_text, html)
        return ParsedContent(text=text, parser_name="html-text-tree")
    if suffix == ".pdf":
        # No model parser configured: deterministic local pypdf extraction.
        # A scanned PDF yields no text here and must fail loudly instead of
        # producing an empty document.
        parsed_pdf = await asyncio.to_thread(_parse_pdf_text, path)
        if parsed_pdf.strip():
            return ParsedContent(text=parsed_pdf, parser_name="pdf-text-tree")
        raise ParsingError(
            f"PDF text extraction returned empty content for {path.name}. "
            "If this is a scanned or image-heavy file, configure a multimodal document parser model."
        )
    if suffix in STRUCTURED_TEXT_SUFFIXES | OOXML_SUFFIXES:
        parsed = await asyncio.to_thread(parse_office_document, path)
        if parsed.strip():
            return ParsedContent(
                text=parsed,
                parser_name=f"{suffix.lstrip('.')}-structured-tree",
            )
        raise ParsingError(f"Structured document parsing returned empty content for {path.name}")
    if suffix in MODEL_PARSER_SUFFIXES or suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ParsingError(
            f"{path.suffix or 'file'} parsing requires a configured document parser model."
        )
    raise ParsingError(f"Unsupported file type: {path.suffix}")


def _parser_enabled(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return parse_bool_env(str(value))


def _read_text_strict(path: Path) -> str:
    # Strict UTF-8: undecodable bytes mean the file is not the text it claims
    # to be, and replacing bytes would silently corrupt the corpus.
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ParsingError(
            f"{path.name} is not valid UTF-8 text "
            f"(decode error at byte {exc.start}); convert the file before ingest"
        ) from exc


def _html_to_text(html: str) -> str:
    return BeautifulSoup(html, "html.parser").get_text("\n")


def _parse_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:  # pragma: no cover - pypdf is in requirements
        raise ParsingError(
            "pypdf is required for local PDF extraction but is not installed"
        ) from exc
    try:
        reader = PdfReader(str(path))
        parts = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                parts.append(f"# Page {index}\n\n{text.strip()}")
        return "\n\n".join(parts).strip()
    except (OSError, PdfReadError, ValueError) as exc:
        raise ParsingError(f"cannot extract text from PDF {path.name}: {exc}") from exc
