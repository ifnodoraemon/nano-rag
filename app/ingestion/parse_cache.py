"""Content-addressed cache for parsed document text.

Parsing now goes through the provider's multimodal model for every file, so a
re-ingest that did not change a single byte would still cost a model call per
document (plus a graph-extraction call). This cache makes the parse step
content-addressed: identical bytes parsed by the same parser configuration are
parsed once.

Two things are deliberately not conflated:

- **Cache key** = SHA-256 of the file bytes. Content, not path or mtime: a
  re-saved file that did not change is a hit, and a touched file that did change
  is a miss.
- **Fingerprint** = what can change the *output* for the same bytes: the parser
  provider/model, the chunk configuration and the artifact schema version.
  Switching parser models must not serve text produced by the previous model, so
  the fingerprint is part of the on-disk path - a configuration change simply
  cannot read entries written under a different configuration, with no
  invalidation bookkeeping to get wrong.

Corrupt entries raise instead of being ignored: silently re-parsing (or worse,
silently skipping) would hide a broken disk behind plausible-looking behaviour,
which is the pattern this codebase removes everywhere else.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from app.core.exceptions import StoreError

logger = logging.getLogger(__name__)

# Bump when the stored shape or the parsing contract changes: old entries then
# become unreachable instead of being misinterpreted.
CACHE_SCHEMA_VERSION = 1

_CACHE_DIR_NAME = ".parse_cache"


@dataclass(frozen=True)
class CachedParse:
    text: str
    parser_name: str


def content_hash(path: Path) -> str:
    """SHA-256 of a file's bytes, streamed so large PDFs are not held in memory."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise StoreError(f"cannot read {path} to compute its content hash: {exc}") from exc
    return digest.hexdigest()


def build_fingerprint(
    *,
    parser_provider: str | None,
    parser_model: str | None,
    chunk_size: object,
    chunk_overlap: object,
) -> str:
    """Fingerprint everything that changes parse/chunk output for the same bytes."""
    payload = json.dumps(
        {
            "schema": CACHE_SCHEMA_VERSION,
            "parser_provider": parser_provider or "",
            "parser_model": parser_model or "",
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class ParseCache:
    """Stores parsed text keyed by (content hash, configuration fingerprint)."""

    def __init__(self, root: Path, fingerprint: str, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.fingerprint = fingerprint
        # The fingerprint segment means a parser/chunk change starts a fresh
        # namespace; entries from the old configuration are left for the cleanup
        # script rather than being silently reinterpreted.
        self.root = root / _CACHE_DIR_NAME / fingerprint[:16]

    def _path(self, digest: str) -> Path:
        return self.root / digest[:2] / f"{digest}.json"

    def get(self, digest: str) -> CachedParse | None:
        """Return the cached parse, or None when absent."""
        if not self.enabled:
            return None
        path = self._path(digest)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StoreError(
                f"parse cache entry {path} is unreadable ({exc}); delete "
                f"{self.root.parent} to rebuild the cache"
            ) from exc
        if not isinstance(raw, dict):
            raise StoreError(f"parse cache entry {path} is not a JSON object")
        text = raw.get("text")
        if not isinstance(text, str) or not text.strip():
            raise StoreError(
                f"parse cache entry {path} has no usable text; delete "
                f"{self.root.parent} to rebuild the cache"
            )
        if raw.get("fingerprint") != self.fingerprint:
            # Only reachable if the directory layout is bypassed; treat as a miss
            # rather than serving text a different configuration produced.
            logger.warning(
                "parse cache fingerprint mismatch for %s; ignoring the entry", path
            )
            return None
        return CachedParse(text=text, parser_name=str(raw.get("parser_name") or "cache"))

    def put(self, digest: str, *, text: str, parser_name: str) -> None:
        """Store a parse result; atomic so a crash cannot leave a torn entry."""
        if not self.enabled:
            return
        target = self._path(digest)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(
                {
                    "fingerprint": self.fingerprint,
                    "schema": CACHE_SCHEMA_VERSION,
                    "parser_name": parser_name,
                    "text": text,
                },
                ensure_ascii=False,
            )
            handle = tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=target.parent,
                prefix=f".{digest}.",
                suffix=".tmp",
                delete=False,
            )
            try:
                with handle:
                    handle.write(payload)
                os.replace(handle.name, target)
            except BaseException:
                Path(handle.name).unlink(missing_ok=True)
                raise
        except OSError as exc:
            # A cache that cannot be written is a broken disk, not a cache miss:
            # failing loudly keeps the degradation visible.
            raise StoreError(f"cannot write parse cache entry {target}: {exc}") from exc
