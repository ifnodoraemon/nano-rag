"""Which file types this system accepts, and how they reach the parser.

Single source of truth shared by the ingest loader (extension allowlist) and the
document-parser client (what it can send). Kept dependency-free so neither side
imports the other.

There is exactly one parsing path: the provider's multimodal model. Nothing is
extracted locally - no pypdf text extraction, no Office/zip file walking, no
HTML stripping - so the suffix sets below only decide *how the bytes are handed
to the model*, never whether a local fallback applies.

- ``TEXT_SUFFIXES``: the file already is text, so its content is sent to the
  model as a text part (no upload, no base64) and works even with a text-only
  model.
- ``BINARY_SUFFIXES``: the model needs the actual file (page layout, tables,
  images, embedded objects), so it is sent as an uploaded/inlined document part.
"""

from __future__ import annotations

# Plain text and source code: read as UTF-8 and handed to the model as text.
TEXT_SUFFIXES = frozenset(
    {
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
)

# Structured text: still text, but the model is better than a naive splitter at
# preserving table shape and record boundaries.
STRUCTURED_TEXT_SUFFIXES = frozenset(
    {".csv", ".json", ".jsonl", ".tsv", ".xml", ".yaml", ".yml"}
)

HTML_SUFFIXES = frozenset({".html", ".htm"})

# Formats whose value is in layout, tables and images: the model must see the
# file itself.
OOXML_SUFFIXES = frozenset({".docx", ".pptx", ".xlsx"})
LEGACY_OFFICE_SUFFIXES = frozenset({".doc", ".ppt", ".xls"})
PDF_SUFFIXES = frozenset({".pdf"})
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"})

# Everything the ingest loader will accept.
SUPPORTED_SUFFIXES = frozenset(
    TEXT_SUFFIXES
    | STRUCTURED_TEXT_SUFFIXES
    | HTML_SUFFIXES
    | OOXML_SUFFIXES
    | LEGACY_OFFICE_SUFFIXES
    | PDF_SUFFIXES
    | IMAGE_SUFFIXES
)

# Everything the parser client can send, minus the text-only group it handles by
# inlining content.
BINARY_SUFFIXES = frozenset(
    OOXML_SUFFIXES | LEGACY_OFFICE_SUFFIXES | PDF_SUFFIXES | IMAGE_SUFFIXES
)

# Suffixes that arrive as text from the caller rather than as a file upload.
INLINE_TEXT_SUFFIXES = frozenset(TEXT_SUFFIXES | STRUCTURED_TEXT_SUFFIXES | HTML_SUFFIXES)

MIME_TYPES = {
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xls": "application/vnd.ms-excel",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".ppt": "application/vnd.ms-powerpoint",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".html": "text/html",
    ".htm": "text/html",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".json": "application/json",
    ".jsonl": "application/json",
    ".xml": "application/xml",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
}


def is_supported(suffix: str) -> bool:
    return suffix.lower() in SUPPORTED_SUFFIXES


def content_type(suffix: str, fallback: str = "application/octet-stream") -> str:
    return MIME_TYPES.get(suffix.lower(), fallback)
