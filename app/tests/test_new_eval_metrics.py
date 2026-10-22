"""Abstention, citation precision, latency and cost metrics."""
from __future__ import annotations


from app.eval.deepeval_runner import DeepevalRunner, _abstention_metrics, _cost_metrics, _latency_metrics
from app.generation.citation_check import check_citations


def _contexts(*pairs):
    return [{"citation_label": label, "text": text} for label, text in pairs]


# ---------------- abstention ----------------

def test_false_refusal_rate_counts_answerable_questions_refused():
    rows = [
        {"answerable": True, "expected_answerable": True},
        {"answerable": False, "expected_answerable": True},
    ]
    metrics = _abstention_metrics(rows)
    assert metrics["false_refusal_rate"] == 0.5


def test_unsupported_answer_rate_counts_fabricated_answers():
    rows = [
        {"answerable": True, "expected_answerable": False},
        {"answerable": False, "expected_answerable": False},
    ]
    metrics = _abstention_metrics(rows)
    assert metrics["correct_abstention_rate"] == 0.5
    assert metrics["unsupported_answer_rate"] == 0.5


def test_abstention_ignores_rows_without_a_verdict():
    rows = [{"expected_answerable": True}, {"answerable": None, "expected_answerable": True}]
    assert _abstention_metrics(rows) == {}


def test_abstention_counts_unlabelled_rows_instead_of_scoring_them():
    rows = [
        {"answerable": True, "expected_answerable": True},
        {"answerable": True},
    ]
    metrics = _abstention_metrics(rows)
    assert metrics["abstention_rows"] == 2
    assert metrics["abstention_rows_unlabelled"] == 1
    assert metrics["false_refusal_rate"] == 0.0


def test_refusal_is_judged_from_the_structured_verdict_not_the_wording():
    """A refusal that never uses the usual wording is still a refusal."""
    rows = [{"answerable": False, "expected_answerable": True}]
    assert _abstention_metrics(rows)["false_refusal_rate"] == 1.0


# ---------------- latency / cost ----------------

def test_latency_percentiles_expose_the_tail():
    rows = [{"latency_seconds": float(value)} for value in range(1, 101)]
    metrics = _latency_metrics(rows)
    assert metrics["latency_max_seconds"] == 100.0
    assert metrics["latency_p50_seconds"] == 51.0
    assert 95.0 <= metrics["latency_p95_seconds"] <= 100.0


def test_latency_absent_when_no_rows_carry_it():
    assert _latency_metrics([{"answer": "x"}]) == {}


def test_cost_uses_generation_usage():
    rows = [
        {"generation_usage": {"total_tokens": 100}},
        {"generation_usage": {"prompt_tokens": 30, "completion_tokens": 20}},
    ]
    metrics = _cost_metrics(rows)
    assert metrics["tokens_per_query_total"] == 150
    assert metrics["tokens_per_query_avg"] == 75.0


def test_cost_absent_when_usage_missing():
    assert _cost_metrics([{"answer": "x"}]) == {}


# ---------------- citation precision ----------------

def test_citation_precision_is_one_when_evidence_matches():
    report = check_citations(
        "员工应在出差结束后 15 个自然日内提交差旅报销申请 [C1]。",
        _contexts(("C1", "员工应在出差结束后 15 个自然日内提交差旅报销申请。")),
    )
    assert report.citation_precision == 1.0


def test_citation_precision_catches_an_unrelated_cited_passage():
    """The label resolves and the number matches, yet the passage is unrelated.

    Integrity alone reports this as perfect; precision is what sees it.
    """
    report = check_citations(
        "员工应在出差结束后 15 个自然日内提交差旅报销申请 [C1]。",
        _contexts(("C1", "公司年会在每年一月举办，15 人参加。")),
    )
    assert report.integrity_score == 1.0
    assert report.citation_precision == 0.0


def test_citation_precision_is_none_without_citations():
    report = check_citations("文档中未包含相关信息。", _contexts(("C1", "x")))
    assert report.citation_precision is None


def test_citation_precision_appears_in_the_dict():
    report = check_citations("答案 [C1]。", _contexts(("C1", "答案")))
    payload = report.as_dict()
    assert "citation_precision" in payload
    assert "sentences_lexically_supported" in payload


# ---------------- end-to-end aggregate ----------------

def test_runner_aggregate_includes_the_new_metrics():
    rows = [
        {
            "sample_id": "a",
            "answer": "答案 [C1]。",
            "retrieved_contexts": [{"citation_label": "C1", "text": "答案"}],
            "answerable": True,
            "expected_answerable": True,
            "latency_seconds": 1.5,
            "generation_usage": {"total_tokens": 42},
        }
    ]
    aggregate = DeepevalRunner(generation_client=None).run(rows)["aggregate"]
    assert aggregate["false_refusal_rate"] == 0.0
    assert aggregate["latency_p95_seconds"] == 1.5
    assert aggregate["tokens_per_query_avg"] == 42.0
    assert aggregate["citation_precision"] == 1.0
