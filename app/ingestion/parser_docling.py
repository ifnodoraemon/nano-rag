from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from app.core.exceptions import ParsingError
from app.model_client.document_parser import DocumentParserClient


async def parse_document(
    path: Path, document_parser: DocumentParserClient | None = None
) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".html":
        html = path.read_text(encoding="utf-8", errors="ignore")
        return BeautifulSoup(html, "html.parser").get_text("\n")
    parser_enabled = bool(getattr(document_parser, "enabled", True))
    if document_parser and parser_enabled and document_parser.supports(path):
        parsed = await document_parser.parse_file(path)
        if parsed.strip():
            return parsed
    if suffix == ".pdf":
        raise ParsingError(
            "PDF parsing requires DOCUMENT_PARSER_ENABLED=true and a configured document parser model."
        )
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        raise ParsingError(
            "Image parsing requires a configured document parser model."
        )
    raise ParsingError(f"Unsupported file type: {path.suffix}")
