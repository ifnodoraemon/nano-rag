"""Shared retrieval hit type.

``SearchHit`` pairs a ``Chunk`` with a relevance score. It is the currency of the
document-level (BM25 wiki) discovery path and the agentic deep-read path — it is
not specific to dense vector retrieval, so it lives in ``app.retrieval`` rather
than the (now removed) vectorstore module.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.chunk import Chunk


@dataclass
class SearchHit:
    chunk: Chunk
    score: float
