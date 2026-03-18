from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from app.core.exceptions import ParsingError


def parse_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".html":
        html = path.read_text(encoding="utf-8", errors="ignore")
        return BeautifulSoup(html, "html.parser").get_text("\n")
    if suffix == ".pdf":
        return _parse_pdf(path)
    raise ParsingError(f"Unsupported file type: {path.suffix}")


def _parse_pdf(path: Path) -> str:
    try:
        from docling.document_converter import DocumentConverter  # type: ignore
    except ImportError:
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    converter = DocumentConverter()
    result = converter.convert(str(path))
    if not hasattr(result, "document"):
        raise ParsingError(f"Docling failed for {path}")
    return result.document.export_to_markdown()
