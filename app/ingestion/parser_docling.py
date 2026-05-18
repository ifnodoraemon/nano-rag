from __future__ import annotations

from pathlib import Path

from app.ingestion.parser_registry import parse_content
from app.model_client.document_parser import DocumentParserClient


async def parse_document(
    path: Path, document_parser: DocumentParserClient | None = None
) -> str:
    return (await parse_content(path, document_parser)).text
