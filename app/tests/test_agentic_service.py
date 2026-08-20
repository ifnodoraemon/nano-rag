from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agentic import AgenticReasoningService
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
    # helper tests exercise only _decompose/_verify, which never touch it,
    # so a truthy stub satisfies the constructor's fail-fast.
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
                "Final Answer:\n"
                "组件A属于系统B，并需要满足3.1节的验收条件。[C1]\n\n"
                "Supporting Claims:\n"
                "- [factual] 组件A与系统B的归属关系和3.1节要求来自同一证据。[C1]"
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


@pytest.mark.asyncio
async def test_agent_planner_helpers_degrade_on_generation_error() -> None:
    service = _service(BrokenPlannerGenerationClient())

    assert await service._decompose("原始问题") == ["原始问题"]
    check = await service._verify(
        "原始问题",
        ["原始问题"],
        [{"chunk_id": "c1", "text": "evidence"}],
    )

    # An unavailable verifier must fail closed (insufficient + follow-up),
    # never be treated as "evidence is sufficient".
    assert check.sufficient is False
    assert check.coverage_ratio == 0.0
    assert check.follow_up_queries == ["原始问题"]
    assert check.reason == "verifier_unavailable"


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
