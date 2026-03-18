from app.generation.answer_formatter import AnswerFormatter


def test_answer_formatter_adds_citation_if_missing() -> None:
    formatter = AnswerFormatter()
    response = formatter.format(
        answer="测试答案",
        contexts=[{"chunk_id": "c1", "source": "/tmp/doc.md", "score": 1.0, "text": "ctx"}],
        trace_id="t1",
    )

    assert response.answer.endswith("[c1]")
    assert len(response.citations) == 1
