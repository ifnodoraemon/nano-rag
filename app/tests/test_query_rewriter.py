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
