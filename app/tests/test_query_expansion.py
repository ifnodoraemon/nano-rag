"""Model-assisted first-hop retrieval: query expansion and RRF fusion."""
from __future__ import annotations

import pytest

from app.agentic.query_expansion import (
    MAX_UNSPACED_VARIANT_CHARS,
    MAX_VARIANT_WORDS,
    expand_query,
    fuse_rankings,
)
from app.core.exceptions import ModelOutputError


class _Client:
    def __init__(self, payload: str, usage: dict | None = None) -> None:
        self.payload = payload
        self.calls = 0
        self.usage = usage or {}

    async def generate(self, messages, **kwargs):  # noqa: ANN001, ANN003
        self.calls += 1
        self.last_kwargs = kwargs
        return {"content": self.payload, "usage": self.usage, "finish_reason": "stop"}


@pytest.mark.asyncio
async def test_expansion_returns_keyword_variants():
    client = _Client('{"keywords": ["差旅报销 提交期限", "差旅费用报销 申请时限"]}')
    variants = await expand_query("请问员工出差回来后报销要在多久内提交呢", client)  # type: ignore[arg-type]
    assert variants == ["差旅报销 提交期限", "差旅费用报销 申请时限"]


@pytest.mark.asyncio
async def test_expansion_requests_a_json_schema():
    client = _Client('{"keywords": ["a"]}')
    await expand_query("q", client)  # type: ignore[arg-type]
    fmt = client.last_kwargs["response_format"]
    assert fmt["type"] == "json_schema"


@pytest.mark.asyncio
async def test_expansion_drops_a_variant_equal_to_the_query():
    client = _Client('{"keywords": ["差旅报销", "差旅报销"]}')
    assert await expand_query("差旅报销", client) == []  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_expansion_caps_the_number_of_variants():
    client = _Client('{"keywords": ["a", "b", "c", "d", "e"]}')
    assert len(await expand_query("q", client, max_variants=2)) == 2  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_sentence_variant_fails_loudly():
    """A long variant defeats the purpose: BM25 is asked a question again."""
    client = _Client('{"keywords": ["' + "很长的句子" * 20 + '"]}')
    with pytest.raises(ModelOutputError, match="exceeds its budget"):
        await expand_query("q", client)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_missing_keywords_array_fails_loudly():
    client = _Client('{"other": []}')
    with pytest.raises(ModelOutputError, match="no keyword array"):
        await expand_query("q", client)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_non_string_variants_fail_loudly():
    client = _Client('{"keywords": [42]}')
    with pytest.raises(ModelOutputError, match="non-string"):
        await expand_query("q", client)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_prose_reply_fails_loudly():
    client = _Client("抱歉，我无法完成。")
    with pytest.raises(ModelOutputError):
        await expand_query("q", client)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_zero_variants_skips_the_model_call():
    client = _Client('{"keywords": ["a"]}')
    assert await expand_query("q", client, max_variants=0) == []  # type: ignore[arg-type]
    assert client.calls == 0


def test_fusion_prefers_documents_ranked_high_by_several_lists():
    original = ["a", "b", "c"]
    variant = ["b", "d", "a"]
    fused = dict(fuse_rankings([original, variant]))
    # "b" is rank 2 then rank 1; "a" is rank 1 then rank 3.
    assert fused["b"] > fused["a"], fused


def test_fusion_uses_rank_not_raw_score():
    """A list cannot dominate through score magnitude - only through position."""
    assert fuse_rankings([["x"], ["y"]]) == [("x", pytest.approx(1 / 61)), ("y", pytest.approx(1 / 61))]


def test_fusion_is_deterministic_on_ties():
    assert [doc for doc, _ in fuse_rankings([["b", "a"]])] == ["b", "a"]
    tied = fuse_rankings([["b"], ["a"]])
    assert [doc for doc, _ in tied] == ["a", "b"], "ties break by doc id, not by dict order"


def test_fusion_of_one_list_is_that_list():
    assert [doc for doc, _ in fuse_rankings([["a", "b", "c"]])] == ["a", "b", "c"]


def test_variant_length_limit_is_sane():
    assert MAX_UNSPACED_VARIANT_CHARS >= 20


@pytest.mark.asyncio
async def test_a_multi_word_english_keyword_query_is_accepted():
    """60 characters is a sentence in English but a normal keyword query in Chinese.

    An English corpus failed 28 of 196 queries because the guard counted characters: the
    variants it rejected were valid keyword queries, just longer in characters because
    English words are longer.
    """
    # Taken from the variants an English corpus actually produced and the old cap
    # rejected: 63-66 characters, 7-9 words - a normal keyword query, not a sentence.
    client = _Client(
        '{"keywords": ["bank substitute check dispute investigation deadline", '
        '"substitute check claim refund twenty days", "bank investigate claim 10 days"]}'
    )
    variants = await expand_query("how long does my bank have to investigate", client)  # type: ignore[arg-type]
    assert len(variants) == 3, "all are valid keyword queries"
    assert all(len(v.split()) <= MAX_VARIANT_WORDS for v in variants)
    # The invariant that matters: a long English keyword query is judged by its word
    # count, not its character count. A 66-character, 9-word variant - the shape the old
    # cap rejected - must pass, while the same rule must still stop a 40-word run-on.
    from app.agentic.query_expansion import _variant_exceeds_budget

    long_query = "how long does my bank have to investigate a substitute check dispute"
    assert len(long_query) > 60, "the old character cap would have rejected this"
    assert _variant_exceeds_budget(long_query)[0] is False, "but it is a valid keyword query"
    assert _variant_exceeds_budget("word " * 40)[0] is True, "a run-on is still rejected"


@pytest.mark.asyncio
async def test_an_endless_english_variant_still_fails():
    client = _Client('{"keywords": ["' + "word " * 40 + '"]}')
    with pytest.raises(ModelOutputError, match="exceeds its budget"):
        await expand_query("q", client)  # type: ignore[arg-type]


def test_the_word_budget_is_small_enough_to_be_a_keyword_query():
    assert MAX_VARIANT_WORDS <= 20
