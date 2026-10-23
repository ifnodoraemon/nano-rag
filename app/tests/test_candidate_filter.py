"""Candidate relevance scoring: replaces a lexical score that cannot discriminate.

Measured on the Chinese-law corpus, BM25 top-6 scores are 0.07/0.06/0.06/0.06/0.06/0.06
when the answer is in the pool and 0.06/0.06/0.05/0.05/0.05/0.05 when it is not - no
usable signal. These tests pin the replacement: the model scores candidates, the scores
filter, and a failure to score raises rather than silently keeping the noise.
"""
from __future__ import annotations

import pytest

from app.agentic.candidate_filter import DROP_BELOW, filter_candidates, score_candidates
from app.core.exceptions import ModelOutputError


class _Client:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls = 0

    async def generate(self, messages, **kwargs):  # noqa: ANN001, ANN003
        self.calls += 1
        return {"content": self.payload}


CANDIDATES = [
    ("大气污染防治法", "排放废气应当安装自动监测设备"),
    ("养犬管理条例", "携犬外出应当束犬链"),
    ("统计法", "统计调查对象应当如实提供资料"),
]


@pytest.mark.asyncio
async def test_scores_are_returned_in_candidate_order():
    client = _Client('{"scores": [90, 10, 45]}')
    scores = await score_candidates("厂里排废气要装监测吗", CANDIDATES, client)  # type: ignore[arg-type]
    assert scores == [90.0, 10.0, 45.0]


@pytest.mark.asyncio
async def test_a_relevant_candidate_survives_and_noise_is_dropped():
    client = _Client('{"scores": [90, 10, 45]}')
    kept = await filter_candidates("厂里排废气要装监测吗", CANDIDATES, client)  # type: ignore[arg-type]
    assert kept == [0, 2]
    assert 1 not in kept


@pytest.mark.asyncio
async def test_the_threshold_is_a_real_decision_point():
    client = _Client(f'{{"scores": [{DROP_BELOW - 1}, {DROP_BELOW}, 100]}}')
    kept = await filter_candidates("q", CANDIDATES, client)  # type: ignore[arg-type]
    assert kept == [1, 2], "a score exactly at the bar is kept"


@pytest.mark.asyncio
async def test_a_weak_pool_yields_no_candidates_and_is_reported():
    """Nothing clears the bar means the pool missed the answer - say so, do not force a pick."""
    client = _Client('{"scores": [5, 3, 1]}')
    kept = await filter_candidates("q", CANDIDATES, client)  # type: ignore[arg-type]
    assert kept == []


@pytest.mark.asyncio
async def test_unparseable_output_raises_instead_of_keeping_everything():
    """Silently keeping the full list would present "we could not judge" as "all relevant"."""
    client = _Client("当然都相关")
    with pytest.raises(ModelOutputError, match="unparseable"):
        await filter_candidates("q", CANDIDATES, client)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_missing_score_list_raises():
    client = _Client('{"relevant": [0]}')
    with pytest.raises(ModelOutputError, match="no score array"):
        await filter_candidates("q", CANDIDATES, client)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_score_count_mismatch_raises():
    client = _Client('{"scores": [90]}')
    with pytest.raises(ModelOutputError, match="1 scores for 3 candidates"):
        await filter_candidates("q", CANDIDATES, client)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_a_non_numeric_score_raises():
    client = _Client('{"scores": [90, "high", 10]}')
    with pytest.raises(ModelOutputError, match="non-numeric"):
        await filter_candidates("q", CANDIDATES, client)  # type: ignore[arg-type]
