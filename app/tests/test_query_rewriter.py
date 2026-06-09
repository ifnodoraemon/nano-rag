import asyncio

from app.retrieval.query_rewriter import QueryRewriter, QueryRewriterConfig


def test_query_rewriter_config_defaults(monkeypatch) -> None:
    monkeypatch.delenv("RAG_QUERY_REWRITE_ENABLED", raising=False)
    monkeypatch.delenv("RAG_MULTI_QUERY_ENABLED", raising=False)
    monkeypatch.delenv("RAG_HYDE_ENABLED", raising=False)
    monkeypatch.delenv("RAG_QUERY_DECOMPOSITION_ENABLED", raising=False)
    config = QueryRewriterConfig.from_env()
    assert config.enable_rewrite is True
    assert config.enable_multi_query is True
    assert config.multi_query_count == 3
    assert config.enable_hyde is True
    assert config.enable_decomposition is True


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
        if "query decomposition assistant" in prompt:
            return {"content": "1. What is the vacation policy?\n2. What is the reimbursement process?"}
        if "Rewritten query:" in prompt:
            if "vacation policy" in prompt:
                return {"content": "vacation carryover"}
            if "reimbursement process" in prompt:
                return {"content": "expense reimbursement steps"}
            return {"content": "vacation carryover"}
        if "Generate 2 queries" in prompt:
            return {"content": "1. PTO carryover\n2. leave rollover"}
        return {"content": "policy about pto carryover and leave rollover"}


class BrokenGenerationClient:
    async def generate(self, messages):  # noqa: ANN001, ARG002
        raise RuntimeError("rewriter unavailable")


def test_query_rewriter_build_plan_applies_rewrite_multi_query_and_hyde() -> None:
    generation_client = FakeGenerationClient()
    rewriter = QueryRewriter(
        generation_client=generation_client,
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
    assert '"query": "vacation policy"' in generation_client.calls[0]


def test_query_rewriter_degrades_on_generation_error() -> None:
    rewriter = QueryRewriter(
        generation_client=BrokenGenerationClient(),
        config=QueryRewriterConfig(
            enable_rewrite=True,
            enable_multi_query=True,
            enable_hyde=True,
        ),
    )

    plan = asyncio.run(rewriter.build_plan("original query"))

    assert plan.rewritten_query is None
    assert plan.retrieval_queries == ["original query"]
    assert plan.hyde_query is None


def test_query_rewriter_build_plan_with_decomposition() -> None:
    generation_client = FakeGenerationClient()
    rewriter = QueryRewriter(
        generation_client=generation_client,
        config=QueryRewriterConfig(
            enable_decomposition=True,
            enable_rewrite=True,
            enable_multi_query=False,
            enable_hyde=False,
        ),
    )

    plan = asyncio.run(rewriter.build_plan("vacation policy and reimbursement process"))

    assert plan.retrieval_queries == [
        "vacation carryover",
        "expense reimbursement steps",
    ]
