import asyncio

from app.retrieval.query_rewriter import QueryRewriter, QueryRewriterConfig


def test_query_rewriter_config_defaults() -> None:
    config = QueryRewriterConfig.from_env()
    assert config.enable_rewrite is False
    assert config.enable_multi_query is False
    assert config.multi_query_count == 3
    assert config.enable_hyde is False


def test_query_rewriter_returns_original_when_disabled() -> None:
    rewriter = QueryRewriter(config=QueryRewriterConfig(enable_rewrite=False))
    result = asyncio.run(rewriter.rewrite("test query"))
    assert result == "test query"


def test_query_rewriter_multi_query_returns_original_when_disabled() -> None:
    rewriter = QueryRewriter(config=QueryRewriterConfig(enable_multi_query=False))
    result = asyncio.run(rewriter.generate_multi_queries("test query"))
    assert result == ["test query"]


def test_query_rewriter_hyde_returns_original_when_disabled() -> None:
    rewriter = QueryRewriter(config=QueryRewriterConfig(enable_hyde=False))
    result = asyncio.run(rewriter.generate_hyde("test query"))
    assert result == "test query"


class FakeGenerationClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def generate(self, messages: list[dict[str, str]]) -> dict[str, str]:
        prompt = messages[0]["content"]
        self.calls.append(prompt)
        if "Rewritten query:" in prompt:
            return {"content": "vacation carryover"}
        if "Generate 2 queries" in prompt:
            return {"content": "1. PTO carryover\n2. leave rollover"}
        return {"content": "policy about pto carryover and leave rollover"}


def test_query_rewriter_build_plan_applies_rewrite_multi_query_and_hyde() -> None:
    rewriter = QueryRewriter(
        generation_client=FakeGenerationClient(),
        config=QueryRewriterConfig(
            enable_rewrite=True,
            enable_multi_query=True,
            multi_query_count=2,
            enable_hyde=True,
        ),
    )

    plan = asyncio.run(rewriter.build_plan("vacation policy"))

    assert plan.rewritten_query == "vacation carryover"
    assert plan.retrieval_queries == [
        "vacation carryover",
        "PTO carryover",
        "leave rollover",
    ]
    assert plan.hyde_query == "policy about pto carryover and leave rollover"
