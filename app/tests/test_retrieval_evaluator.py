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


class FakeGenerationClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def generate(self, messages):  # noqa: ANN001
        prompt = messages[0]["content"]
        self.prompts.append(prompt)
        return {"content": "0.7"}


def test_retrieval_evaluator_sends_query_and_document_as_json() -> None:
    generation_client = FakeGenerationClient()
    evaluator = RetrievalEvaluator(
        generation_client=generation_client,
        config=RetrievalEvaluatorConfig(enable_evaluation=True),
    )

    score = asyncio.run(
        evaluator.evaluate_relevance(
            'query with {"json": true}',
            "This document is long enough to pass the relevance evaluator length gate.",
        )
    )

    assert score == 0.7
    assert '"query": "query with' in generation_client.prompts[0]
    assert '"document": "This document is long enough' in generation_client.prompts[0]
