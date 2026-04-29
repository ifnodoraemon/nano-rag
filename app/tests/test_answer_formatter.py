from app.generation.answer_formatter import AnswerFormatter


def test_answer_formatter_adds_citation_if_missing() -> None:
    formatter = AnswerFormatter()
    response = formatter.format(
        answer="Test answer",
        contexts=[{"chunk_id": "c1", "citation_label": "C1", "source": "/tmp/doc.md", "score": 1.0, "text": "ctx"}],
        trace_id="t1",
    )

    assert "[C1]" in response.answer
    assert len(response.citations) == 1
    assert response.citations[0].citation_label == "C1"
    assert response.citations[0].span_text == "ctx"
    assert response.citations[0].span_start == 0
    assert response.citations[0].span_end == 3


def test_answer_formatter_prioritizes_primary_citations_and_reorders_contexts() -> None:
    formatter = AnswerFormatter()
    response = formatter.format(
        answer="Test answer",
        contexts=[
            {
                "chunk_id": "c-conflict",
                "citation_label": "C3",
                "source": "/tmp/conflict.md",
                "score": 0.7,
                "text": "conflict",
                "evidence_role": "conflicting",
                "wiki_status": "conflicting",
            },
            {
                "chunk_id": "c-primary",
                "citation_label": "C1",
                "source": "/tmp/primary.md",
                "score": 0.6,
                "text": "primary",
                "evidence_role": "primary",
            },
            {
                "chunk_id": "c-support",
                "citation_label": "C2",
                "source": "/tmp/support.md",
                "score": 0.9,
                "text": "support",
                "evidence_role": "supporting",
            },
        ],
        trace_id="t1",
    )

    assert response.citations[0].chunk_id == "c-primary"
    assert len(response.citations) == 1
    assert [context["chunk_id"] for context in response.contexts] == [
        "c-primary",
        "c-support",
        "c-conflict",
    ]
    assert "[C1]" in response.answer
    assert response.citations[0].evidence_role == "primary"
    assert response.citations[0].span_text == "primary"


def test_answer_formatter_adds_conflict_notice_when_conflicting_evidence_exists() -> None:
    formatter = AnswerFormatter()
    response = formatter.format(
        answer="Test answer",
        contexts=[
            {
                "chunk_id": "c-primary",
                "citation_label": "C1",
                "source": "/tmp/primary.md",
                "score": 1.0,
                "text": "primary",
                "evidence_role": "primary",
            },
            {
                "chunk_id": "c-conflict",
                "citation_label": "C2",
                "source": "/tmp/conflict.md",
                "score": 0.8,
                "text": "conflict",
                "evidence_role": "conflicting",
                "wiki_status": "conflicting",
            },
        ],
        trace_id="t1",
    )

    assert "available evidence is conflicting" in response.answer
    assert "[C1]" in response.answer
    assert "Conflicting evidence" not in response.answer


def test_answer_formatter_does_not_duplicate_conflict_notice() -> None:
    formatter = AnswerFormatter()
    response = formatter.format(
        answer="The sources are inconsistent, so this should be interpreted carefully.",
        contexts=[
            {
                "chunk_id": "c-conflict",
                "citation_label": "C1",
                "source": "/tmp/conflict.md",
                "score": 1.0,
                "text": "conflict",
                "evidence_role": "conflicting",
                "wiki_status": "conflicting",
            }
        ],
        trace_id="t1",
    )

    assert response.answer.count("inconsistent") == 1


def test_answer_formatter_does_not_duplicate_evidence_summary() -> None:
    formatter = AnswerFormatter()
    response = formatter.format(
        answer="Test answer [C1]\n\nPrimary evidence: [C1]",
        contexts=[
            {
                "chunk_id": "c-primary",
                "citation_label": "C1",
                "source": "/tmp/primary.md",
                "score": 1.0,
                "text": "primary",
                "evidence_role": "primary",
            }
        ],
        trace_id="t1",
    )

    assert response.answer.count("Primary evidence: [C1]") == 1


def test_answer_formatter_extracts_best_matching_span_from_context() -> None:
    formatter = AnswerFormatter()
    response = formatter.format(
        answer="Carryover is allowed up to 5 days.",
        contexts=[
            {
                "chunk_id": "c-primary",
                "citation_label": "C1",
                "source": "/tmp/primary.md",
                "score": 1.0,
                "text": (
                    "Leave rules overview. "
                    "Carryover is allowed up to 5 days with manager approval. "
                    "Unused leave expires after one year."
                ),
                "evidence_role": "primary",
            }
        ],
        trace_id="t1",
    )

    assert (
        response.citations[0].span_text
        == "Carryover is allowed up to 5 days with manager approval."
    )
    assert response.citations[0].span_start is not None
    assert response.citations[0].span_end is not None


def test_answer_formatter_respects_model_citation_label_order() -> None:
    formatter = AnswerFormatter()
    response = formatter.format(
        answer="Use supporting evidence first [C2], then the primary citation [C1].",
        contexts=[
            {
                "chunk_id": "c-primary",
                "citation_label": "C1",
                "source": "/tmp/primary.md",
                "score": 0.9,
                "text": "primary",
                "evidence_role": "primary",
            },
            {
                "chunk_id": "c-support",
                "citation_label": "C2",
                "source": "/tmp/support.md",
                "score": 0.7,
                "text": "support",
                "evidence_role": "supporting",
            },
        ],
        trace_id="t1",
    )

    assert [citation.citation_label for citation in response.citations] == ["C2", "C1"]


def test_answer_formatter_only_returns_explicitly_cited_labels() -> None:
    formatter = AnswerFormatter()
    response = formatter.format(
        answer="The reimbursement request must be submitted within 15 days. [C1]",
        contexts=[
            {
                "chunk_id": "c-primary",
                "citation_label": "C1",
                "source": "/tmp/primary.md",
                "score": 0.9,
                "text": "Submit reimbursement within 15 days.",
                "evidence_role": "primary",
            },
            {
                "chunk_id": "c-support",
                "citation_label": "C2",
                "source": "/tmp/support.md",
                "score": 0.7,
                "text": "Another unrelated policy section.",
                "evidence_role": "primary",
            },
        ],
        trace_id="t1",
    )

    assert [citation.citation_label for citation in response.citations] == ["C1"]
    assert "[C2]" not in response.answer


def test_answer_formatter_drops_numeric_citations_without_answer_value() -> None:
    formatter = AnswerFormatter()
    response = formatter.format(
        answer="中宁县徐套乡的价格为 31,800 元/亩 [C1, C2, C3]。",
        contexts=[
            {
                "chunk_id": "fragment",
                "citation_label": "C1",
                "source": "/tmp/doc.md",
                "score": 0.9,
                "text": "| 0 | 喊叫水乡、徐套乡 |",
                "evidence_role": "primary",
            },
            {
                "chunk_id": "unit",
                "citation_label": "C2",
                "source": "/tmp/doc.md",
                "score": 0.8,
                "text": "单位：元/亩",
                "evidence_role": "primary",
            },
            {
                "chunk_id": "row",
                "citation_label": "C3",
                "source": "/tmp/doc.md",
                "score": 0.7,
                "text": "| 中卫市 | 中宁县 | III | 31800 | 喊叫水乡、徐套乡 |",
                "evidence_role": "primary",
            },
        ],
        trace_id="t1",
    )

    assert response.answer == "中宁县徐套乡的价格为 31,800 元/亩 [C3]。"
    assert [citation.citation_label for citation in response.citations] == ["C3"]


def test_answer_formatter_prefers_matching_table_row_for_citation_span() -> None:
    formatter = AnswerFormatter()
    response = formatter.format(
        answer="中宁县新鲜一级西红柿的今日价格和上一周价格均为 4.98元/500克 [C1]。",
        contexts=[
            {
                "chunk_id": "price-row",
                "citation_label": "C1",
                "source": "/tmp/prices.pdf",
                "score": 0.9,
                "text": (
                    "| 品种 | 规格等级 | 计量单位 | 今日价格 | 上一周价格 | 同期比（±%） | 备注 |\n"
                    "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
                    "| 茄子 | 新鲜一级 | 元/500克 | 5.18 | 4.98 | 4.0 | |\n"
                    "| 西红柿 | 新鲜一级 | 元/500克 | 4.98 | 4.98 | 0.0 | |\n"
                ),
                "evidence_role": "primary",
            }
        ],
        trace_id="t1",
    )

    assert response.citations[0].span_text == "| 西红柿 | 新鲜一级 | 元/500克 | 4.98 | 4.98 | 0.0 | |"


def test_answer_formatter_ignores_citation_label_number_for_span_scoring() -> None:
    formatter = AnswerFormatter()
    response = formatter.format(
        answer="徐套乡属于中宁县 [C4]。",
        contexts=[
            {
                "chunk_id": "scope-row",
                "citation_label": "C4",
                "source": "/tmp/scope.pdf",
                "score": 0.9,
                "text": (
                    "| 中卫市 | 中宁县 | Ⅰ | 41200 | 宁安镇、新堡镇、石空镇 |\n"
                    "| 中卫市 | 中宁县 | Ⅲ | 31800 | 喊叫水乡、徐套乡 |\n"
                ),
                "evidence_role": "primary",
            }
        ],
        trace_id="t1",
    )

    assert response.citations[0].span_text == "| 中卫市 | 中宁县 | Ⅲ | 31800 | 喊叫水乡、徐套乡 |"


def test_answer_formatter_extracts_structured_answer_plan_and_supporting_claims() -> None:
    formatter = AnswerFormatter()
    response = formatter.format(
        answer=(
            "Final Answer:\n"
            "Carryover is allowed up to 5 days with manager approval. [C1]\n\n"
            "Supporting Claims:\n"
            "- [factual] The current leave policy sets the carryover limit at 5 days. [C1]\n"
            "- [conditional] Manager approval is required before rollover is finalized. [C2]\n"
        ),
        contexts=[
            {
                "chunk_id": "c-primary",
                "citation_label": "C1",
                "source": "/tmp/primary.md",
                "score": 0.9,
                "text": "Carryover is allowed up to 5 days.",
                "evidence_role": "primary",
            },
            {
                "chunk_id": "c-support",
                "citation_label": "C2",
                "source": "/tmp/support.md",
                "score": 0.7,
                "text": "Manager approval is required.",
                "evidence_role": "supporting",
            },
        ],
        trace_id="t1",
    )

    assert response.answer.startswith(
        "Carryover is allowed up to 5 days with manager approval. [C1]"
    )
    assert len(response.supporting_claims) == 2
    assert response.supporting_claims[0].text == (
        "The current leave policy sets the carryover limit at 5 days."
    )
    assert response.supporting_claims[0].claim_type == "factual"
    assert response.supporting_claims[0].citation_labels == ["C1"]
    assert response.supporting_claims[1].claim_type == "conditional"
    assert response.supporting_claims[1].citation_labels == ["C2"]


def test_answer_formatter_ignores_none_supporting_claims_placeholder() -> None:
    formatter = AnswerFormatter()
    response = formatter.format(
        answer="Final Answer:\nEvidence is insufficient. [C1]\n\nSupporting Claims:\n- None",
        contexts=[
            {
                "chunk_id": "c1",
                "citation_label": "C1",
                "source": "/tmp/doc.md",
                "score": 1.0,
                "text": "Evidence is insufficient.",
                "evidence_role": "supporting",
            }
        ],
        trace_id="t1",
    )

    assert response.supporting_claims == []


def test_answer_formatter_infers_claim_types_when_prefix_is_missing() -> None:
    formatter = AnswerFormatter()
    response = formatter.format(
        answer=(
            "Final Answer:\n"
            "The answer depends on manager approval. [C1]\n\n"
            "Supporting Claims:\n"
            "- Approval applies only if the employee has unused leave. [C1]\n"
            "- Sources are inconsistent about whether carryover is capped. [C2]\n"
            "- Evidence is insufficient to determine the contractor policy. [C3]\n"
            "- Carryover is capped at 5 days. [C4]\n"
        ),
        contexts=[
            {
                "chunk_id": "c1",
                "citation_label": "C1",
                "source": "/tmp/1.md",
                "score": 1.0,
                "text": "Approval applies only if the employee has unused leave.",
            },
            {
                "chunk_id": "c2",
                "citation_label": "C2",
                "source": "/tmp/2.md",
                "score": 0.9,
                "text": "Sources disagree on the cap.",
            },
            {
                "chunk_id": "c3",
                "citation_label": "C3",
                "source": "/tmp/3.md",
                "score": 0.8,
                "text": "Evidence is insufficient.",
            },
            {
                "chunk_id": "c4",
                "citation_label": "C4",
                "source": "/tmp/4.md",
                "score": 0.7,
                "text": "Carryover is capped at 5 days.",
            },
        ],
        trace_id="t1",
    )

    assert [claim.claim_type for claim in response.supporting_claims] == [
        "conditional",
        "conflict",
        "insufficiency",
        "factual",
    ]
