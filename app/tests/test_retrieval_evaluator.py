import asyncio

from app.retrieval.retrieval_evaluator import (
    RetrievalEvaluator,
    RetrievalEvaluatorConfig,
)


def test_retrieval_evaluator_config_defaults() -> None:
    config = RetrievalEvaluatorConfig.from_env()
    assert config.enable_evaluation is False
    assert config.relevance_threshold == 0.5


def test_retrieval_evaluator_returns_default_when_disabled() -> None:
    evaluator = RetrievalEvaluator(
        config=RetrievalEvaluatorConfig(enable_evaluation=False)
    )
    score = asyncio.run(evaluator.evaluate_relevance("test query", "test document"))
    assert score == 1.0


def test_retrieval_evaluator_filter_relevant_when_disabled() -> None:
    evaluator = RetrievalEvaluator(
        config=RetrievalEvaluatorConfig(enable_evaluation=False)
    )
    docs = ["doc1", "doc2", "doc3"]
    results = asyncio.run(evaluator.filter_relevant("query", docs, top_k=2))
    assert len(results) == 2
    assert all(score == 1.0 for _, score in results)


def test_retrieval_evaluator_returns_default_when_no_client() -> None:
    evaluator = RetrievalEvaluator(
        generation_client=None,
        config=RetrievalEvaluatorConfig(enable_evaluation=True),
    )
    score = asyncio.run(evaluator.evaluate_relevance("test query", "test document"))
    assert score == 1.0


def test_retrieval_evaluator_short_document_returns_default() -> None:
    evaluator = RetrievalEvaluator(
        generation_client=None,
        config=RetrievalEvaluatorConfig(enable_evaluation=True),
    )
    score = asyncio.run(evaluator.evaluate_relevance("test query", "hi"))
    assert score == 1.0
