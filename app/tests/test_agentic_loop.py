"""The agentic verification loop: budget, corrective queries, termination.

Audited as completely unexercised: every fake verifier returned sufficient=True
on the first pass, so corrective re-retrieval, query dedupe and the loop bound
were never executed - which is how an off-by-one survived (a configured budget of
2 performed 3 retrievals).
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace


from app.agentic import AgenticReasoningService
from app.agentic.service import EvidenceCheck
from app.core.tracing import TraceStore
from app.generation.answer_formatter import AnswerFormatter
from app.generation.prompt_builder import PromptBuilder


class FakeTracingManager:
    @contextmanager
    def span(self, name, attributes=None):  # noqa: ANN001
        yield


def _service(max_loops: int = 2) -> AgenticReasoningService:
    config = SimpleNamespace(
        parsed_dir=Path("/tmp/nonexistent-loop-test"),
        settings={
            "agent": {
                "max_retrieval_loops": max_loops,
                "max_subqueries": 4,
                "min_coverage_ratio": 0.38,
            },
            "prompt": {"version": "test"},
        },
    )

    class _Client:
        alias = "fake"

        async def generate(self, messages, **kwargs):  # noqa: ANN001, ARG002
            return {"content": '{"subqueries": []}'}

    return AgenticReasoningService(
        config=config,
        generation_client=_Client(),
        prompt_builder=PromptBuilder({"chat": {"system": "system"}}),
        answer_formatter=AnswerFormatter(),
        trace_store=TraceStore(),
        tracing_manager=FakeTracingManager(),
        wiki_searcher=SimpleNamespace(),
    )


def _check(sufficient: bool, follow_ups: list[str] | None = None) -> EvidenceCheck:
    return EvidenceCheck(
        sufficient=sufficient,
        coverage_ratio=0.9 if sufficient else 0.1,
        missing_terms=[] if sufficient else ["x"],
        follow_up_queries=follow_ups or [],
        has_contexts=True,
        has_conflicts=False,
        reason="test",
    )


def _state(queried: list[str], check: EvidenceCheck) -> dict:
    return {
        "retrieval_queries": queried,
        "check": check,
        "subqueries": [],
        "resolved_query": queried[0],
        "payload": SimpleNamespace(query=queried[0]),
    }


def test_sufficient_evidence_goes_straight_to_synthesis() -> None:
    service = _service()
    assert service._route_after_verification(
        _state(["q"], _check(True))
    ) == "answer_synthesis"


def test_insufficient_evidence_triggers_corrective_recall() -> None:
    service = _service(max_loops=2)
    route = service._route_after_verification(
        _state(["q"], _check(False, ["q2"]))
    )
    assert route == "corrective_recall"


def test_the_budget_counts_the_initial_retrieval() -> None:
    """A budget of 2 must mean two retrievals, not three.

    `retrieval_queries` already holds the initial retrieval; comparing it with
    `>` instead of `>=` allowed one extra round.
    """
    service = _service(max_loops=2)
    # One retrieval so far -> a second is still allowed.
    assert service._route_after_verification(
        _state(["q"], _check(False, ["q2"]))
    ) == "corrective_recall"
    # Two retrievals -> the budget is spent.
    assert service._route_after_verification(
        _state(["q", "q2"], _check(False, ["q3"]))
    ) == "answer_synthesis"


def test_a_budget_of_one_permits_a_single_retrieval() -> None:
    service = _service(max_loops=1)
    assert service._route_after_verification(
        _state(["q"], _check(False, ["q2"]))
    ) == "answer_synthesis"


def test_no_unused_follow_up_query_ends_the_loop() -> None:
    service = _service(max_loops=5)
    assert service._route_after_verification(
        _state(["q"], _check(False, []))
    ) == "answer_synthesis"


def test_an_already_asked_follow_up_is_skipped() -> None:
    """Re-searching the same query cannot change the evidence."""
    service = _service(max_loops=5)
    state = _state(["q", "q2"], _check(False, ["q2"]))
    assert service._next_query(state) is None
    assert service._route_after_verification(state) == "answer_synthesis"


def test_the_next_query_is_the_first_never_asked_follow_up() -> None:
    service = _service(max_loops=5)
    state = _state(["q"], _check(False, ["q2", "q3"]))
    assert service._next_query(state) == "q2"


def test_follow_up_queries_are_preferred_over_the_original() -> None:
    """Falling back to the original query first would repeat the same search."""
    service = _service(max_loops=5)
    state = _state(["q"], _check(False, ["q2"]))
    assert service._next_query(state) == "q2"
