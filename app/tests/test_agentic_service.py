from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agentic import AgenticReasoningService
from app.core.exceptions import ModelGatewayError, ModelOutputError
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
    answer = service._render_structured_answer(
        '{"is_answerable":true,"missing_entities":[],'
        '"extracted_answer":"答案 [C1]",'
        '"supporting_claims":["[factual] 支撑点 [C1]"]}'
    )
    assert answer.startswith("Final Answer:\n答案 [C1]")
    assert "[factual] 支撑点 [C1]" in answer


def test_render_structured_answer_unanswerable() -> None:
    service = _service(FakeGenerationClient())
    answer = service._render_structured_answer(
        '{"is_answerable":false,"missing_entities":["术语X"],'
        '"extracted_answer":"","supporting_claims":[]}'
    )
    assert "术语X" in answer
    assert "[insufficiency]" in answer


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
