"""Scope preference: read from the document, applied as a multiplier, never a filter.

The bug this addresses, from the end-to-end run: "男的多少岁能结婚？" retrieved three
autonomous-region derogations (all: twenty) instead of the national rule (twenty-two),
because the derogations say 结婚年龄 verbatim. The model refused, correctly, but the
retrieval had already lost.

The design constraint: no per-domain table. A list of "which institution outranks which"
works for one corpus and is worthless for a contract or a manual.
"""
from __future__ import annotations

import pytest

from app.agentic.scope import (
    MATCH_FACTOR,
    MISMATCH_FACTOR,
    document_matches,
    preference_for,
)
from app.core.exceptions import ModelOutputError


def test_a_general_question_prefers_documents_with_no_narrow_scope():
    general_doc = {"applies_to": [], "authority_rank": 1}
    regional_doc = {"applies_to": ["本自治州户籍的少数民族"], "authority_rank": 2}
    assert document_matches(general_doc, "general")
    assert not document_matches(regional_doc, "general")


def test_a_specific_question_prefers_documents_that_name_a_scope():
    general_doc = {"applies_to": [], "authority_rank": 1}
    regional_doc = {"applies_to": ["宁夏"], "authority_rank": 2}
    assert document_matches(regional_doc, "specific")
    assert not document_matches(general_doc, "specific")


def test_a_document_with_no_recorded_scope_matches_everything():
    """Nothing known about a document is not evidence against it."""
    assert document_matches({}, "general")
    assert document_matches({}, "specific")
    assert document_matches({"applies_to": [], "authority_rank": None}, "general")


def test_curation_is_generic_not_china_specific():
    """A contract that binds named subsidiaries is 'specific', same as a regional law."""
    contract = {"applies_to": ["中国大陆及香港地区子公司"], "authority_rank": 2}
    assert document_matches(contract, "specific")
    manual = {"applies_to": ["适用于所有生产基地"], "authority_rank": 2}
    assert document_matches(manual, "specific")


def test_an_unspecified_question_gets_no_preference():
    pref = preference_for("unspecified")
    assert pref.is_neutral
    assert pref.matches == 1.0 and pref.others == 1.0


def test_the_preference_is_a_multiplier_not_a_filter():
    """Both factors stay above zero: a regional rule must remain findable."""
    pref = preference_for("general")
    assert 0 < pref.others < 1 < pref.matches
    assert pref.matches == MATCH_FACTOR
    assert pref.others == MISMATCH_FACTOR


class _Client:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    async def generate(self, messages, **kwargs):  # noqa: ANN001, ANN003
        return {"content": self.payload}


@pytest.mark.asyncio
async def test_document_scope_is_read_from_the_document():
    from app.agentic.scope import describe_document_scope

    client = _Client('{"applies_to": ["本省行政区域内"], "authority_rank": 2}')
    scope = await describe_document_scope("第一条 本省...", client, title="某省条例")  # type: ignore[arg-type]
    assert scope["applies_to"] == ["本省行政区域内"]
    assert scope["authority_rank"] == 2


@pytest.mark.asyncio
async def test_a_document_stating_no_scope_is_not_an_error():
    from app.agentic.scope import describe_document_scope

    client = _Client('{"applies_to": []}')
    scope = await describe_document_scope("某条款", client, title="x")  # type: ignore[arg-type]
    assert scope["applies_to"] == []
    assert scope["authority_rank"] is None


@pytest.mark.asyncio
async def test_a_malformed_scope_reply_raises():
    from app.agentic.scope import describe_document_scope

    client = _Client("这份文件适用于全国")
    with pytest.raises(ModelOutputError, match="unparseable"):
        await describe_document_scope("内容", client, title="x")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_an_invalid_authority_rank_raises():
    from app.agentic.scope import describe_document_scope

    client = _Client('{"applies_to": [], "authority_rank": 0}')
    with pytest.raises(ModelOutputError, match="invalid authority_rank"):
        await describe_document_scope("内容", client, title="x")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_query_scope_rejects_an_unknown_verdict():
    from app.agentic.scope import query_scope

    client = _Client('{"scope": "national"}')
    with pytest.raises(ModelOutputError, match="unknown verdict"):
        await query_scope("q", client)  # type: ignore[arg-type]
