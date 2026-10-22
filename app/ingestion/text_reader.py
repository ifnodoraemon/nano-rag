"""Strict text reading for files that are handed to the parser as text.

Dependency-free leaf module: the document-parser client needs this, and the
parser registry needs the client's type, so putting it here keeps the two from
importing each other in a cycle.
"""

from __future__ import annotations

from pathlib import Path

from app.core.exceptions import ParsingError


def read_inline_text(path: Path) -> str:
    """Read a text-like file as strict UTF-8.

    Undecodable bytes mean the file is not the text it claims to be; replacing
    bytes would silently corrupt the corpus, so this raises.
    """
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ParsingError(
            f"{path.name} is not valid UTF-8 text "
            f"(decode error at byte {exc.start}); convert the file before ingest"
        ) from exc
