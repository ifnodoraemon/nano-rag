"""HyDE without embeddings: the passage becomes term weights, never a longer string."""
from __future__ import annotations

import pytest

from app.agentic.hyde import (
    MAX_FEEDBACK_TERMS,
    hypothetical_document,
    rocchio_weights,
)
from app.core.exceptions import ModelOutputError


class _Client:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    async def generate(self, messages, **kwargs):  # noqa: ANN001, ANN003
        return {"content": self.payload}


@pytest.mark.asyncio
async def test_the_hypothetical_passage_is_what_the_model_wrote():
    client = _Client("生产经营单位应当对从业人员进行安全生产教育和培训。")
    text = await hypothetical_document("没培训就上岗行吗", client)  # type: ignore[arg-type]
    assert "安全生产教育和培训" in text


@pytest.mark.asyncio
async def test_an_empty_passage_raises_instead_of_searching_with_nothing():
    client = _Client("   ")
    with pytest.raises(ModelOutputError, match="no content"):
        await hypothetical_document("q", client)  # type: ignore[arg-type]


def test_query_terms_keep_full_weight_and_feedback_terms_are_downweighted():
    weights = rocchio_weights(["培训", "上岗"], ["培训", "生产经营", "教育"])
    # The user's own terms are never weakened by feedback.
    assert weights["培训"] >= 1.0
    assert weights["上岗"] == 1.0
    # A feedback term starts from zero and gains beta.
    assert 0 < weights["生产经营"] < weights["上岗"]


def test_a_feedback_term_already_in_the_query_is_reinforced():
    weights = rocchio_weights(["培训"], ["培训", "培训"])
    assert weights["培训"] > rocchio_weights(["培训"], [])["培训"]


def test_corpus_frequent_terms_are_dropped():
    """A term in most documents discriminates nothing - this is where naive concat fails."""
    feedback = ["的", "应当", "培训"]
    weights = rocchio_weights(
        ["培训"],
        feedback,
        document_frequencies={"的": 990, "应当": 800, "培训": 5},
        n_docs=1000,
    )
    assert "的" not in weights
    assert "应当" not in weights
    assert "培训" in weights


def test_the_cutoff_is_a_ratio_not_an_absolute_count():
    # 50/1000 = 5% -> kept; the same absolute count in a 100-doc corpus would be 50%.
    weights = rocchio_weights(
        [],
        ["term"],
        document_frequencies={"term": 50},
        n_docs=1000,
    )
    assert "term" in weights


def test_feedback_depth_is_bounded():
    feedback = [f"term{i}" for i in range(MAX_FEEDBACK_TERMS * 3)]
    weights = rocchio_weights([], feedback)
    assert len(weights) <= MAX_FEEDBACK_TERMS


def test_the_strongest_feedback_terms_are_the_ones_kept():
    feedback = ["weak"] + ["strong"] * 5
    weights = rocchio_weights([], feedback)
    assert "strong" in weights
    assert weights["strong"] > weights["weak"]


def test_without_any_feedback_the_query_is_unchanged():
    assert rocchio_weights(["a", "b"], []) == {"a": 1.0, "b": 1.0}
