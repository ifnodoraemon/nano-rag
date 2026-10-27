"""Index capacity: a real limit that fails loudly, and warns before it.

Measured on 595 Chinese statutes: 24,705 indexed pages, 363 MB resident, ~147 MB per
10,000 pages. Article-level indexing multiplies page count ~40x, so the old default of
50,000 was exhausted by roughly 1,200 statutes - a size ordinary corpora reach, where
ingest would then fail.
"""
from __future__ import annotations

import pytest

from app.core.exceptions import RetrievalError
from app.retrieval.bm25 import BM25Config, BM25Index
from app.wiki.search import DEFAULT_MAX_DOCUMENTS


def test_the_default_capacity_covers_a_realistic_corpus():
    # 595 statutes produce ~24,705 pages; the default must not be near that.
    assert DEFAULT_MAX_DOCUMENTS >= 100_000


def test_capacity_exhaustion_raises_rather_than_dropping():
    """A silent drop would make documents unretrievable with no signal at all."""
    index = BM25Index(BM25Config(max_documents=2))
    index.add_document("a", "alpha beta")
    index.add_document("b", "gamma delta")
    with pytest.raises(RetrievalError, match="capacity reached"):
        index.add_document("c", "epsilon zeta")
    assert index.document_count == 2


def test_reaching_capacity_is_reported_before_it_is_hit(caplog):
    index = BM25Index(BM25Config(max_documents=8))
    with caplog.at_level("WARNING"):
        for i in range(8):
            index.add_document(f"d{i}", f"term{i} other")
    assert any("of capacity" in record.message for record in caplog.records), (
        "an operator should learn about index growth while there is still headroom"
    )
