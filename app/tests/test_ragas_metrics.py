from app.eval.ragas_metrics import RAGASConfig, RAGASMetrics


def test_ragas_config_defaults() -> None:
    config = RAGASConfig.from_env()
    assert config.enable_llm_judge is True
    assert config.llm_judge_model == "gpt-4o-mini"
    assert config.max_concurrent_judgments == 3


def test_ragas_metrics_defaults() -> None:
    metrics = RAGASMetrics()
    assert metrics.answer_exact_match == 0.0
    assert metrics.reference_context_recall == 0.0
    assert metrics.faithfulness == 0.0
    assert metrics.answer_relevance == 0.0
    assert metrics.context_precision == 0.0


def test_ragas_metrics_custom_values() -> None:
    metrics = RAGASMetrics(
        answer_exact_match=1.0,
        reference_context_recall=0.8,
        faithfulness=0.9,
        answer_relevance=0.85,
        context_precision=0.75,
    )
    assert metrics.answer_exact_match == 1.0
    assert metrics.reference_context_recall == 0.8
    assert metrics.faithfulness == 0.9
    assert metrics.answer_relevance == 0.85
    assert metrics.context_precision == 0.75
