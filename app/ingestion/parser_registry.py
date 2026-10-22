"""Document parsing: always the provider's multimodal model.

This build has a single parsing path. Every accepted file is turned into Markdown
by the configured multimodal document parser; there is no local extraction (no
pypdf text mining, no Office zip walking, no HTML stripping) and therefore no
"which parser ran" matrix, no quality cliff between formats, and no silent
fallback when the provider is unavailable.

Deliberate consequences:

- A missing or disabled document parser makes ingest fail loudly with a
  configuration error instead of falling back to a lesser local parse.
- A provider that returns nothing usable raises ``ParsingError`` rather than
  producing an empty document that would be undiscoverable in the index later.
- Text-like files are still handed to the model - as text - so a text-only model
  can parse them; binary formats need a multimodal model and fail loudly if the
  model cannot read them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from app.core.exceptions import ConfigurationError, ParsingError
from app.ingestion.parse_cache import ParseCache, content_hash
from app.ingestion.suffixes import INLINE_TEXT_SUFFIXES, SUPPORTED_SUFFIXES
from app.ingestion.text_reader import read_inline_text
from app.utils.text import parse_bool_env

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from app.model_client.document_parser import DocumentParserClient

__all__ = [
    "ParsedContent",
    "parse_content",
    "read_inline_text",
    "require_document_parser",
    "supported_text_suffixes",
]


@dataclass(frozen=True)
class ParsedContent:
    text: str
    parser_name: str
    metadata: dict[str, object] = field(default_factory=dict)


def supported_text_suffixes() -> set[str]:
    """Every suffix the ingest loader accepts; all go through the model parser."""
    return set(SUPPORTED_SUFFIXES)


def _parser_enabled(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return parse_bool_env(str(value))


def require_document_parser(
    document_parser: DocumentParserClient | None,
) -> DocumentParserClient:
    """Return the parser or fail: there is no local parsing path to fall back to."""
    if document_parser is None:
        raise ConfigurationError(
            "document parsing requires the provider's multimodal parser, and this "
            "build has no local extraction. Configure DOCUMENT_PARSER_API_BASE_URL, "
            "DOCUMENT_PARSER_API_KEY and DOCUMENT_PARSER_MODEL."
        )
    # An object that exists is presumed usable; only an explicit falsy `enabled`
    # means the operator switched parsing off (the real client always exposes it).
    if not _parser_enabled(getattr(document_parser, "enabled", True)):
        raise ConfigurationError(
            "DOCUMENT_PARSER_ENABLED is false, but this build has no local "
            "extraction: every document is parsed by the provider's multimodal "
            "model. Set DOCUMENT_PARSER_ENABLED=true and configure the parser."
        )
    return document_parser


async def parse_content(
    path: Path,
    document_parser: DocumentParserClient | None = None,
    cache: ParseCache | None = None,
) -> ParsedContent:
    """Parse one file into Markdown using the multimodal document parser.

    With a cache, identical bytes parsed under the same configuration are reused
    instead of calling the model again; ``metadata["parse_cache"]`` records which
    happened so a hit is never invisible.
    """
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ParsingError(f"Unsupported file type: {path.suffix}")

    parser = require_document_parser(document_parser)
    if not parser.supports(path):
        raise ParsingError(
            f"{path.suffix or 'file'} is not supported by the configured document "
            "parser; check DOCUMENT_PARSER_PROVIDER and the model's capabilities."
        )

    digest = content_hash(path) if cache and cache.enabled else ""
    if cache and cache.enabled:
        cached = cache.get(digest)
        if cached is not None:
            return ParsedContent(
                text=cached.text,
                parser_name=cached.parser_name,
                metadata={"parse_cache": "hit", "content_hash": digest},
            )

    text = await parser.parse_file(path)
    if not text.strip():
        raise ParsingError(f"Document parser returned empty content for {path.name}")

    parser_name = "multimodal-llm-tree"
    if suffix in INLINE_TEXT_SUFFIXES:
        parser_name = "multimodal-llm-inline-text"
    if cache and cache.enabled:
        cache.put(digest, text=text, parser_name=parser_name)
    return ParsedContent(
        text=text,
        parser_name=parser_name,
        metadata={
            "parse_cache": "miss" if cache and cache.enabled else "disabled",
            "content_hash": digest,
        },
    )
