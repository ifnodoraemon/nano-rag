from app.eval.ragas_runner import RagasRunner


def test_ragas_runner_returns_aggregate_metrics() -> None:
    runner = RagasRunner()
    report = runner.run(
        [
            {
                "query": "q1",
                "reference_answer": "a1",
                "answer": "a1",
                "reference_contexts": ["ctx"],
                "retrieved_contexts": ["ctx"],
            }
        ]
    )

    assert report["status"] == "ok"
    assert report["records"] == 1
    assert report["aggregate"]["answer_exact_match"] == 1.0
