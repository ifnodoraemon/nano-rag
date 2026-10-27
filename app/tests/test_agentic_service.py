from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agentic import AgenticReasoningService
from app.core.exceptions import ModelOutputError
from app.core.tracing import TraceStore
from app.generation.answer_formatter import AnswerFormatter
from app.generation.prompt_builder import PromptBuilder


class FakeTracingManager:
    @contextmanager
    def span(self, name, attributes=None):  # noqa: ANN001
        yield


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        parsed_dir=Path("/tmp/nonexistent-agent-test-parsed"),
        settings={
            "agent": {
                "max_retrieval_loops": 2,
                "max_subqueries": 4,
                "min_coverage_ratio": 0.38,
            },
            "prompt": {"version": "test"},
        },
    )


def _service(generation_client) -> AgenticReasoningService:
    # wiki_searcher is now a hard dependency of the agentic engine; the
    # helper tests exercise only _decompose/_verify/_render_structured_answer,
    # which never touch it, so a truthy stub satisfies the constructor's
    # fail-fast.
    return AgenticReasoningService(
        config=_config(),
        generation_client=generation_client,
        prompt_builder=PromptBuilder({"chat": {"system": "system"}}),
        answer_formatter=AnswerFormatter(),
        trace_store=TraceStore(),
        tracing_manager=FakeTracingManager(),
        wiki_searcher=SimpleNamespace(),
    )


class FakeGenerationClient:
    alias = "fake-generator"

    async def generate(self, messages, **kwargs):  # noqa: ANN001
        rendered = str(messages)
        if "Break the user question" in rendered:
            return {
                "content": (
                    '{"subqueries":["组件A属于哪个系统",'
                    '"组件A需要满足什么条件"]}'
                )
            }
        if "evidence auditor" in rendered:
            return {
                "content": (
                    '{"sufficient":true,"coverage_ratio":0.9,'
                    '"missing_terms":[],"follow_up_queries":[],"reason":"closed"}'
                )
            }
        assert "Agent evidence check" in rendered
        return {
            "content": (
                '{"is_answerable":true,"missing_entities":[],'
                '"extracted_answer":"组件A属于系统B，并需要满足3.1节的验收条件。[C1]",'
                '"supporting_claims":["[factual] 组件A与系统B的归属关系和3.1节要求来自同一证据。[C1]"]}'
            ),
            "finish_reason": "stop",
            "usage": {"total_tokens": 12},
        }


class BrokenPlannerGenerationClient:
    async def generate(self, messages, **kwargs):  # noqa: ANN001, ARG002
        raise RuntimeError("planner unavailable")


class StringBooleanVerifierClient:
    alias = "string-bool-verifier"

    async def generate(self, messages, **kwargs):  # noqa: ANN001, ARG002
        return {
            "content": (
                '{"sufficient":"false","coverage_ratio":"0.25",'
                '"missing_terms":["missing"],'
                '"follow_up_queries":["follow up"],"reason":"needs_more"}'
            )
        }


class GarbageJsonClient:
    alias = "garbage"

    async def generate(self, messages, **kwargs):  # noqa: ANN001, ARG002
        return {"content": "this is not json at all"}


@pytest.mark.asyncio
async def test_agent_planner_helpers_fail_loud_on_generation_error() -> None:
    # No degraded fallback: an unavailable planner/verifier raises and fails
    # the request visibly instead of continuing with the raw query or a
    # fabricated "insufficient" verdict.
    service = _service(BrokenPlannerGenerationClient())

    with pytest.raises(RuntimeError, match="planner unavailable"):
        await service._decompose("原始问题")
    with pytest.raises(RuntimeError, match="planner unavailable"):
        await service._verify(
            "原始问题",
            ["原始问题"],
            [{"chunk_id": "c1", "text": "evidence"}],
        )


@pytest.mark.asyncio
async def test_agent_helpers_fail_loud_on_unparseable_json() -> None:
    service = _service(GarbageJsonClient())

    with pytest.raises(ModelOutputError):
        await service._decompose("原始问题")
    with pytest.raises(ModelOutputError):
        await service._verify(
            "原始问题",
            ["原始问题"],
            [{"chunk_id": "c1", "text": "evidence"}],
        )


@pytest.mark.asyncio
async def test_agent_verifier_parses_string_booleans() -> None:
    service = _service(StringBooleanVerifierClient())

    check = await service._verify(
        "原始问题",
        ["原始问题"],
        [{"chunk_id": "c1", "text": "partial evidence"}],
    )

    assert check.sufficient is False
    assert check.coverage_ratio == 0.25
    assert check.follow_up_queries == ["follow up"]


@pytest.mark.asyncio
async def test_agent_verify_without_contexts_is_deterministically_insufficient() -> None:
    service = _service(FakeGenerationClient())

    check = await service._verify("原始问题", ["原始问题"], [])

    assert check.sufficient is False
    assert check.reason == "no_contexts"
    assert check.has_contexts is False


def test_render_structured_answer_answerable() -> None:
    service = _service(FakeGenerationClient())
    answer, answerable = service._render_structured_answer(
        '{"is_answerable":true,"missing_entities":[],'
        '"extracted_answer":"答案 [C1]",'
        '"supporting_claims":["[factual] 支撑点 [C1]"]}'
    )
    assert answer.startswith("Final Answer:\n答案 [C1]")
    assert "[factual] 支撑点 [C1]" in answer


def test_render_structured_answer_unanswerable() -> None:
    """The model writes the refusal, in the question's language.

    A format string here once made every refusal Chinese: a US Code question in English
    was answered "文档中未包含关于..." because the sentence was assembled in code, where no
    prompt could reach it. The system no longer invents refusal wording.
    """
    service = _service(FakeGenerationClient())
    answer, answerable = service._render_structured_answer(
        '{"is_answerable":false,"missing_entities":["term X"],'
        '"extracted_answer":"The provided documents do not mention term X.",'
        '"supporting_claims":[]}'
    )
    assert answerable is False
    assert "The provided documents do not mention term X." in answer
    assert "文档中未包含" not in answer


def test_an_unanswerable_verdict_must_carry_its_own_explanation() -> None:
    """Otherwise the system would have to synthesise wording - which is how it once
    produced Chinese refusals for English questions."""
    service = _service(FakeGenerationClient())
    with pytest.raises(ModelOutputError, match="its own explanation"):
        service._render_structured_answer(
            '{"is_answerable":false,"missing_entities":["term X"],'
            '"extracted_answer":"","supporting_claims":[]}'
        )


def test_render_structured_answer_rejects_contract_violations() -> None:
    service = _service(FakeGenerationClient())
    with pytest.raises(ModelOutputError):
        service._render_structured_answer('{"is_answerable":"yes"}')
    with pytest.raises(ModelOutputError):
        service._render_structured_answer(
            '{"is_answerable":true,"missing_entities":[],"extracted_answer":"",'
            '"supporting_claims":[]}'
        )
    with pytest.raises(ModelOutputError):
        service._render_structured_answer(
            '{"is_answerable":true,"missing_entities":[],"extracted_answer":"ok",'
            '"supporting_claims":"not-a-list"}'
        )


class SequencedAuditorClient:
    """An auditor that answers a fixed sequence, so re-audit paths are testable."""

    alias = "sequenced-auditor"

    def __init__(self, answers: list[dict]) -> None:
        self.answers = list(answers)
        self.calls = 0

    async def generate(self, messages, **kwargs):  # noqa: ANN001, ARG002
        payload = self.answers[min(self.calls, len(self.answers) - 1)]
        self.calls += 1
        return {"content": __import__("json").dumps(payload, ensure_ascii=False)}


@pytest.mark.asyncio
async def test_a_first_pass_sufficient_audit_costs_one_call() -> None:
    client = SequencedAuditorClient(
        [{"sufficient": True, "coverage_ratio": 0.9, "missing_terms": [],
          "follow_up_queries": [], "reason": "ok"}]
    )
    service = _service(client)

    check = await service._verify(
        "问题", ["问题"], [{"chunk_id": "c1", "text": "答案在这里。", "citation_label": "C1"}]
    )

    assert check.sufficient is True
    assert check.initial_sufficient is True
    assert client.calls == 1
    assert len(check.audit_seconds) == 1
    assert check.as_dict()["audit_calls"] == 1
    assert check.as_dict()["redrill_improved"] is False


@pytest.mark.asyncio
async def test_an_insufficient_verdict_costs_exactly_one_audit_call() -> None:
    """A real gap must not be papered over, and must not be paid for twice.

    A second auditor pass over the same documents was removed after measuring
    that it exposed no term the first pass had not: the first pass is already
    term-driven, so a gap it reports is a gap in the retrieved evidence. The
    audit count on the trace has to show one call, which is what makes "fewer
    corrective rounds" separable from "extra audit calls".
    """
    client = SequencedAuditorClient(
        [
            {"sufficient": False, "coverage_ratio": 0.1, "missing_terms": ["缴费年限"],
             "follow_up_queries": ["缴费年限 规定"], "reason": "缺"},
            {"sufficient": True, "coverage_ratio": 0.9, "missing_terms": [],
             "follow_up_queries": [], "reason": "should never be asked"},
        ]
    )
    service = _service(client)

    check = await service._verify(
        "问题", ["问题"], [{"chunk_id": "c1", "text": "无关内容。", "citation_label": "C1"}]
    )

    assert check.sufficient is False
    assert check.initial_sufficient is False
    # The verdict is not re-audited: one call, and the corrector gets the gaps.
    assert client.calls == 1
    assert len(check.audit_seconds) == 1
    assert check.as_dict()["audit_calls"] == 1
    assert check.as_dict()["redrill_improved"] is False
    assert check.follow_up_queries == ["缴费年限 规定"]


