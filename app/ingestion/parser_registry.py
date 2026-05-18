from __future__ import annotations

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


async def parse_content(
    path: Path,
    document_parser: DocumentParserClient | None = None,
) -> ParsedContent:
    suffix = path.suffix.lower()
    if suffix in PLAIN_TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8", errors="replace")
        return ParsedContent(text=text, parser_name="plain-text-tree")
    if suffix in HTML_SUFFIXES:
        html = path.read_text(encoding="utf-8", errors="replace")
        text = BeautifulSoup(html, "html.parser").get_text("\n")
        return ParsedContent(text=text, parser_name="html-text-tree")
    if suffix == ".pdf":
        parsed_pdf = _parse_pdf_text(path)
        if parsed_pdf.strip():
            return ParsedContent(text=parsed_pdf, parser_name="pdf-text-tree")
    if suffix in STRUCTURED_TEXT_SUFFIXES | OOXML_SUFFIXES:
        parsed = parse_office_document(path)
        if parsed.strip():
            return ParsedContent(
                text=parsed,
                parser_name=f"{suffix.lstrip('.')}-structured-tree",
            )
        raise ParsingError(f"Structured document parsing returned empty content for {path.name}")
    parser_enabled = _parser_enabled(getattr(document_parser, "enabled", True))
    if document_parser and parser_enabled and document_parser.supports(path):
        parsed = await document_parser.parse_file(path)
        if parsed.strip():
            return ParsedContent(text=parsed, parser_name="multimodal-llm-tree")
        raise ParsingError(f"Document parser returned empty content for {path.name}")
    if suffix in MODEL_PARSER_SUFFIXES or suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ParsingError(
            f"{path.suffix or 'file'} parsing requires a configured document parser model."
        )
    raise ParsingError(f"Unsupported file type: {path.suffix}")


def _parser_enabled(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return parse_bool_env(str(value))


def _parse_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError:
        return ""
    try:
        reader = PdfReader(str(path))
        parts = []
        for index, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                parts.append(f"# Page {index}\n\n{text.strip()}")
        return "\n\n".join(parts).strip()
    except (OSError, PdfReadError, ValueError):
        return ""
