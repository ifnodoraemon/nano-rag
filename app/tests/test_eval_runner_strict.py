"""Eval-runner no-silent-loss semantics.

The old runner projected every record down to 5 builtin fields, so
trace_id/kb_id/conflicting counts vanished from the report rows — the
benchmark then treated every row as a bad case (missing metrics read as
0.0), diagnosis always reported retrieval_gap, and the
conflicting-hit-rate gate could never fail. These tests pin the fixed
behavior.
"""
from __future__ import annotations

import pytest

from app.eval.deepeval_runner import (
    DeepevalRunner,
    _answer_exact_match,
    _reference_context_recall,
)


def _record(**overrides) -> dict:
    record = {
        "sample_id": "sample-1",
        "kb_id": "default",
        "query": "中宁县徐套乡的区片综合地价是多少？",
        "answer": "中宁县徐套乡的区片综合地价为 31800 元/亩。",
        "reference_answer": "中宁县徐套乡的区片综合地价为 31800 元/亩。",
        "reference_contexts": [
            "| 中卫市 | 中宁县 | III | 31800 | 喊叫水乡、徐套乡 |"
        ],
        "retrieved_contexts": [
            "Table row: 地级市=中卫市; 县（市、区）=中宁县; 区片编号=III; "
            "区片综合地价=31800; 区片范围=喊叫水乡、徐套乡"
        ],
        "trace_id": "trace-abc",
        "conflicting_context_count": 1,
        "top_k": 6,
    }
    record.update(overrides)
    return record


def test_run_passes_every_field_through() -> None:
    report = DeepevalRunner().run([_record()])
    row = report["results"][0]
    assert row["trace_id"] == "trace-abc"
    assert row["kb_id"] == "default"
    assert row["conflicting_context_count"] == 1
    assert row["top_k"] == 6
    assert report["mode"] == "deterministic"


def test_run_computes_deterministic_metrics_without_llm() -> None:
    report = DeepevalRunner().run([_record()])
    row = report["results"][0]
    assert row["answer_exact_match"] == 1.0
    assert row["reference_context_recall"] == 1.0
    agg = report["aggregate"]
    assert agg["answer_exact_match"] == 1.0
    assert agg["reference_context_recall"] == 1.0
    assert agg["conflicting_hit_rate"] == 1.0
    assert agg["conflicting_context_count_avg"] == 1.0


def test_run_missing_number_fails_recall_deterministically() -> None:
    report = DeepevalRunner().run(
        [
            _record(
                retrieved_contexts=[
                    "Table row: 地级市=中卫市; 县（市、区）=中宁县; "
                    "区片编号=III; 区片综合地价=31800"
                ],
            )
        ]
    )
    # 喊叫水乡/徐套乡 missing → overlap below threshold AND numbers still
    # present, but the character overlap is 4/8-ish; assert recall is partial
    # or zero — never silently 1.0.
    assert report["results"][0]["reference_context_recall"] < 1.0


def test_run_wrong_number_fails_recall() -> None:
    report = DeepevalRunner().run(
        [_record(retrieved_contexts=["徐套乡区片综合地价为 62000 元/亩"])]
    )
    assert report["results"][0]["reference_context_recall"] == 0.0


def test_run_empty_records() -> None:
    report = DeepevalRunner().run([])
    assert report["records"] == 0
    assert report["results"] == []


def test_answer_exact_match_is_token_based() -> None:
    assert (
        _answer_exact_match(
            "根据规定，中宁县徐套乡的区片综合地价为 31800 元/亩。",
            "中宁县徐套乡的区片综合地价为 31800 元/亩。",
        )
        == 1.0
    )
    # Wrong number → not a match, even though 99% of tokens match.
    assert (
        _answer_exact_match(
            "中宁县徐套乡的区片综合地价为 62000 元/亩。",
            "中宁县徐套乡的区片综合地价为 31800 元/亩。",
        )
        == 0.0
    )
    # Empty reference cannot produce a pass.
    assert _answer_exact_match("anything", "") == 0.0


def test_reference_context_recall_table_row_serialization() -> None:
    # Table references are re-serialized as "header=value" rows; recall must
    # not require byte-exact containment.
    assert (
        _reference_context_recall(
            ["| 中卫市 | 中宁县 | III | 31800 | 喊叫水乡、徐套乡 |"],
            ["Table row: 地级市=中卫市; 区片范围=喊叫水乡、徐套乡; 价格=31800"],
        )
        == 1.0
    )
    assert _reference_context_recall([], ["anything"]) == 0.0
    assert _reference_context_recall(["ref"], []) == 0.0


@pytest.mark.asyncio
async def test_run_async_passes_fields_through_and_computes_builtin() -> None:
    """LLM mode must keep the passthrough + deterministic metrics too."""

    class _FakeGenerationClient:
        alias = "fake-model"
        base_url = "http://localhost:1/v1"
        api_key = "fake-key"

    runner = DeepevalRunner(generation_client=_FakeGenerationClient())

    original_import = __import__

    def fake_import(name, *args, **kwargs):  # noqa: ANN002, ANN003
        if name.startswith("deepeval.metrics"):
            class _M:
                def __init__(self, **kwargs):  # noqa: ARG002
                    self.score = 0.5

                async def a_measure(self, test_case):  # noqa: ARG002
                    self.score = 0.5

            class _Metrics:
                FaithfulnessMetric = _M
                AnswerRelevancyMetric = _M
                ContextualPrecisionMetric = _M
                ContextualRecallMetric = _M

            return _Metrics()
        if name.startswith("deepeval.test_case"):
            class _TestCase:
                def __init__(self, **kwargs):  # noqa: ARG002
                    pass

            class _Cases:
                LLMTestCase = _TestCase

            return _Cases()
        return original_import(name, *args, **kwargs)

    import builtins

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(builtins, "__import__", fake_import)
        report = await runner.run_async([_record()])

    row = report["results"][0]
    assert row["trace_id"] == "trace-abc"
    assert row["answer_exact_match"] == 1.0
    assert row["reference_context_recall"] == 1.0
    assert report["mode"] == "llm-assisted"
