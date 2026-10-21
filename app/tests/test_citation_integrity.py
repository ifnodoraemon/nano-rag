"""Deterministic citation integrity.

The invariant "every answer is traceable to retrieved context" was enforced only
by the prompt. These pin the checks that make it executable: a cited label must
exist, the numbers a cited sentence states must appear in the context it cites,
and "no citations at all" is reported as not-evaluable rather than perfect.
"""

from __future__ import annotations

from app.eval.deepeval_runner import DeepevalRunner, _builtin_metrics
from app.generation.citation_check import check_citations


def _ctx(label: str, text: str) -> dict[str, object]:
    return {"citation_label": label, "chunk_id": f"chunk-{label}", "text": text}


# --------------------------------------------------------------------------- #
# Label resolution                                                            #
# --------------------------------------------------------------------------- #

def test_all_labels_resolve_and_numbers_are_grounded() -> None:
    contexts = [_ctx("C1", "员工应在出差结束后 15 个自然日内提交差旅报销申请。")]
    report = check_citations("差旅报销需在 15 个自然日内提交 [C1]。", contexts)

    assert report.cited_labels == ("C1",)
    assert report.resolved_labels == ("C1",)
    assert report.unresolved_labels == ()
    assert report.grounded_sentences == 1
    assert report.integrity_score == 1.0


def test_fabricated_label_is_reported() -> None:
    # The model cites a label that was never provided: the answer looks
    # traceable but points at nothing.
    contexts = [_ctx("C1", "员工应在出差结束后 15 个自然日内提交差旅报销申请。")]
    report = check_citations("报销需在 15 个自然日内提交 [C9]。", contexts)

    assert report.unresolved_labels == ("C9",)
    assert report.resolved_labels == ()
    assert report.integrity_score == 0.0
    assert any(issue.kind == "unresolved_label" for issue in report.issues)


def test_number_absent_from_cited_context_is_ungrounded() -> None:
    # A real label attached to a number the context does not contain - the
    # failure mode that matters most (prices, deadlines, versions).
    contexts = [_ctx("C1", "员工应在出差结束后 15 个自然日内提交差旅报销申请。")]
    report = check_citations("报销需在 30 个自然日内提交 [C1]。", contexts)

    assert report.ungrounded_number_count == 1
    assert report.integrity_score == 0.0
    assert any(issue.kind == "ungrounded_number" for issue in report.issues)


def test_number_formatting_is_normalized() -> None:
    # 15, 15.0 and 15.00 are the same claim.
    contexts = [_ctx("C1", "上限为 15,000 元。")]
    report = check_citations("上限是 15000 元 [C1]。", contexts)

    assert report.ungrounded_number_count == 0
    assert report.integrity_score == 1.0


def test_grouped_citation_labels_are_all_checked() -> None:
    contexts = [
        _ctx("C1", "第一条：适用范围。"),
        _ctx("C2", "提交时限为 15 个自然日。"),
    ]
    report = check_citations("时限为 15 个自然日 [C1, C2]。", contexts)

    assert set(report.cited_labels) == {"C1", "C2"}
    assert report.unresolved_labels == ()
    assert report.integrity_score == 1.0


def test_number_may_be_grounded_by_any_cited_context() -> None:
    # A sentence citing two labels is supported if the number is in either.
    contexts = [
        _ctx("C1", "适用范围：全体正式员工。"),
        _ctx("C2", "提交时限为 15 个自然日。"),
    ]
    report = check_citations("时限为 15 个自然日 [C1, C2]。", contexts)

    assert report.integrity_score == 1.0


def test_quoted_span_must_appear_in_the_cited_context() -> None:
    contexts = [_ctx("C1", "制度原文：员工须在十五日内提交。")]
    report = check_citations('制度规定“十五日内提交” [C1]。', contexts)
    assert report.ungrounded_quote_count == 0

    wrong = check_citations('制度规定“三十日内提交” [C1]。', contexts)
    assert wrong.ungrounded_quote_count == 1
    assert wrong.integrity_score == 0.0


# --------------------------------------------------------------------------- #
# Not-evaluable vs perfect                                                    #
# --------------------------------------------------------------------------- #

def test_answer_without_citations_is_not_evaluable() -> None:
    contexts = [_ctx("C1", "员工应在出差结束后 15 个自然日内提交差旅报销申请。")]
    report = check_citations("报销时限为 15 个自然日。", contexts)

    assert report.sentences_with_citations == 0
    assert report.sentences_without_citations == 1
    # "No citations" is a different failure; it must not read as a perfect score.
    assert report.integrity_score is None


def test_insufficiency_answer_is_not_evaluable() -> None:
    report = check_citations("文档中未包含相关信息。", [])
    assert report.integrity_score is None
    assert report.sentences_without_citations == 1


def test_partial_grounding_scores_by_sentence() -> None:
    contexts = [_ctx("C1", "时限为 15 个自然日。")]
    # First sentence grounded, second cites a number that is not in the context.
    report = check_citations("时限为 15 个自然日 [C1]。金额为 500 元 [C1]。", contexts)

    assert report.sentences_with_citations == 2
    assert report.grounded_sentences == 1
    assert report.integrity_score == 0.5


def test_report_is_json_serializable() -> None:
    contexts = [_ctx("C1", "时限为 15 个自然日。")]
    payload = check_citations("时限为 15 个自然日 [C1]。", contexts).as_dict()
    assert payload["integrity_score"] == 1.0
    assert payload["cited_labels"] == ["C1"]
    assert isinstance(payload["issues"], list)


# --------------------------------------------------------------------------- #
# Eval integration                                                            #
# --------------------------------------------------------------------------- #

def test_builtin_metrics_include_citation_integrity() -> None:
    metrics = _builtin_metrics(
        {
            "answer": "时限为 15 个自然日 [C1]。",
            "reference_answer": "15 个自然日",
            "retrieved_contexts": ["提交时限为 15 个自然日。"],
            "reference_contexts": ["提交时限为 15 个自然日。"],
        }
    )
    assert metrics["citation_integrity"] == 1.0


def test_report_aggregates_citation_integrity_and_its_coverage() -> None:
    report = DeepevalRunner().run(
        [
            {
                "answer": "时限为 15 个自然日 [C1]。",
                "reference_answer": "15 个自然日",
                "retrieved_contexts": ["提交时限为 15 个自然日。"],
                "reference_contexts": ["提交时限为 15 个自然日。"],
            },
            {
                # No citation -> not evaluable -> excluded from the mean, and
                # visible in the coverage figure.
                "answer": "时限为 15 个自然日。",
                "reference_answer": "15 个自然日",
                "retrieved_contexts": ["提交时限为 15 个自然日。"],
                "reference_contexts": ["提交时限为 15 个自然日。"],
            },
        ]
    )
    aggregate = report["aggregate"]
    assert aggregate["citation_integrity"] == 1.0
    assert aggregate["metric_coverage"]["citation_integrity"] == 0.5
    assert aggregate["records_without_metric"]["citation_integrity"] == 1
