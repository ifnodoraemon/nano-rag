"""Focus sections against documents that have no heading structure.

Real corpora contain heading-less documents (scanned PDFs, exported emails, flat
web pages). A focus section cannot resolve against structure that does not exist,
so that must read the document rather than fail - but a document that DOES have
headings and still matches nothing is a planner/artifact disagreement and must
still fail loudly.
"""
from __future__ import annotations

import pytest

from app.agentic.discovery import select_chunks_for_focus
from app.core.exceptions import ModelOutputError


def _chunk(chunk_id: str, text: str, *, headings=None, title: str = "") -> dict:
    metadata = {"kb_id": "default"}
    if headings is not None:
        metadata["hierarchy_path"] = headings
    return {
        "chunk_id": chunk_id,
        "doc_id": "d1",
        "chunk_index": 0,
        "text": text,
        "source_path": "x.md",
        "title": title,
        "metadata": metadata,
    }


def test_flat_document_reads_instead_of_failing() -> None:
    """No headings anywhere: a focus section cannot resolve, so read the document."""
    chunks = [
        _chunk("c1", "第1集剧情简介", title="精武陈真"),
        _chunk("c2", "第18集剧情简介", title="精武陈真"),
    ]
    selected = select_chunks_for_focus(
        chunks, {"第18集剧情"}, doc_id="d1", max_chunks=24
    )
    assert [chunk["chunk_id"] for chunk in selected] == ["c1", "c2"]


def test_flat_document_respects_the_chunk_budget() -> None:
    chunks = [_chunk(f"c{index}", f"text {index}") for index in range(10)]
    selected = select_chunks_for_focus(chunks, {"Any Section"}, doc_id="d1", max_chunks=3)
    assert len(selected) == 3


def test_structured_document_with_unmatched_focus_still_fails() -> None:
    """The loud failure is reserved for the real contract violation."""
    chunks = [
        _chunk("c1", "reimbursement text", headings=["差旅报销"]),
        _chunk("c2", "leave text", headings=["请假"]),
    ]
    with pytest.raises(ModelOutputError, match="headed chunks"):
        select_chunks_for_focus(chunks, {"不存在的章节"}, doc_id="d1", max_chunks=24)


def test_structured_document_matches_its_headings() -> None:
    chunks = [
        _chunk("c1", "reimbursement text", headings=["差旅报销"]),
        _chunk("c2", "leave text", headings=["请假"]),
    ]
    selected = select_chunks_for_focus(chunks, {"差旅报销"}, doc_id="d1", max_chunks=24)
    assert [chunk["chunk_id"] for chunk in selected] == ["c1"]


def test_mixed_document_fails_when_named_section_does_not_exist() -> None:
    """One headed chunk is enough to make the document structured."""
    chunks = [
        _chunk("c1", "flat preamble"),
        _chunk("c2", "headed body", headings=["附录"]),
    ]
    with pytest.raises(ModelOutputError, match="do not match any section"):
        select_chunks_for_focus(chunks, {"正文"}, doc_id="d1", max_chunks=24)


def test_empty_focus_reads_the_whole_document() -> None:
    chunks = [_chunk("c1", "a", headings=["A"]), _chunk("c2", "b", headings=["B"])]
    selected = select_chunks_for_focus(chunks, set(), doc_id="d1", max_chunks=24)
    assert len(selected) == 2
