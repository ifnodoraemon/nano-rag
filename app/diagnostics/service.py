from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.schemas.diagnosis import DiagnosisFinding, DiagnosisResponse
from app.schemas.trace import TraceRecord

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from app.model_client.generation import GenerationClient


def _looks_like_refusal(answer: str) -> bool:
    refusal_markers = (
        "cannot confirm",
        "insufficient information",
        "insufficient evidence",
        "no available context",
        "please provide",
        "无法确认",
        "信息不足",
        "证据不足",
        "没有可用上下文",
        "请提供",
    )
    lowered = answer.lower()
    return any(marker in answer or marker in lowered for marker in refusal_markers)


def _mentions_conflict(answer: str) -> bool:
    conflict_markers = (
        "存在冲突",
        "说法不一致",
        "来源不一致",
        "无法确定",
        "不能确定",
        "inconsistent",
        "conflict",
    )
    lowered = answer.lower()
    return any(marker in answer or marker in lowered for marker in conflict_markers)


def _count_claim_types(claims: object) -> dict[str, int]:
    counts = {
        "factual": 0,
        "conditional": 0,
        "conflict": 0,
        "insufficiency": 0,
    }
    if not isinstance(claims, list):
        return counts
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        claim_type = str(claim.get("claim_type", "")).strip().lower()
        if claim_type in counts:
            counts[claim_type] += 1
    return counts


@dataclass
class DiagnosisService:
    generation_client: GenerationClient | None = None

    def diagnose_trace(self, trace: TraceRecord) -> DiagnosisResponse:
        findings: list[DiagnosisFinding] = []
        contexts = trace.contexts or []
        retrieved = trace.retrieved or []
        answer = str(trace.answer or "")
        claim_type_counts = _count_claim_types(trace.supporting_claims)
        conflicting_contexts = [
            context
            for context in contexts
            if isinstance(context, dict) and context.get("wiki_status") == "conflicting"
        ]

        if conflicting_contexts:
            findings.append(
                DiagnosisFinding(
                    category="wiki_conflict_detected",
                    severity="medium",
                    rationale="A conflicting wiki topic was retrieved, which means the knowledge layer has already detected disagreement across sources.",
                    suggested_actions=[
                        "Review the topic page's Potential Conflicts section and the linked source pages first.",
                        "If the conflict is real, add effective scope, date, or version metadata to the knowledge page.",
                        "Require the generation layer to explain the conflict before giving a conditional answer.",
                    ],
                    evidence={
                        "conflicting_context_count": len(conflicting_contexts),
                        "conflicting_chunk_ids": [
                            context.get("chunk_id") for context in conflicting_contexts
                        ],
                        "conflicting_sources": [
                            context.get("source") for context in conflicting_contexts
                        ],
                    },
                )
            )
            if claim_type_counts["conflict"] == 0:
                findings.append(
                    DiagnosisFinding(
                        category="conflict_claim_missing",
                        severity="medium",
                        rationale="Conflicting evidence was retrieved, but no structured conflict claim was produced.",
                        suggested_actions=[
                            "Require the generation layer to emit at least one [conflict] supporting claim when conflicting evidence is present.",
                            "Keep conflict claims short and bind them to explicit citation labels.",
                            "Use claim-level checks in regression to prevent conflict omissions.",
                        ],
                        evidence={
                            "conflicting_context_count": len(conflicting_contexts),
                            "claim_type_counts": claim_type_counts,
                        },
                    )
                )
            if answer and not _mentions_conflict(answer):
                findings.append(
                    DiagnosisFinding(
                        category="conflict_not_reflected_in_answer",
                        severity="high",
                        rationale="A conflicting topic was retrieved, but the final answer did not reflect that conflict, creating overconfidence risk.",
                        suggested_actions=[
                            "Tighten the prompt so conflicting topics must be acknowledged explicitly.",
                            "Add conflict safeguards in the answer formatter or post-generation processing.",
                            "When necessary, downgrade conflicting topics to a clarification-style answer.",
                        ],
                        evidence={
                            "answer_preview": answer[:200],
                            "conflicting_chunk_ids": [
                                context.get("chunk_id")
                                for context in conflicting_contexts
                            ],
                        },
                    )
                )

        if not retrieved:
            findings.append(
                DiagnosisFinding(
                    category="retrieval_empty",
                    severity="high",
                    rationale="The retrieval stage returned no candidate chunks, so the likely issue is ingestion, embeddings, or retrieval configuration.",
                    suggested_actions=[
                        "Confirm that documents were ingested successfully and the vector store contains data.",
                        "Check that the embedding model and collection dimensions are aligned.",
                        "Increase retrieval top_k first, then inspect the retrieve/debug output.",
                    ],
                    evidence={"retrieved_count": 0, "context_count": len(contexts)},
                )
            )

        elif contexts and not answer:
            findings.append(
                DiagnosisFinding(
                    category="generation_empty",
                    severity="high",
                    rationale="Relevant context was available, but generation returned no answer.",
                    suggested_actions=[
                        "Inspect the generation provider response body and finish_reason.",
                        "Confirm that prompt construction and the answer formatter did not clear the content.",
                    ],
                    evidence={
                        "context_count": len(contexts),
                        "generation_finish_reason": trace.generation_finish_reason,
                    },
                )
            )

        elif contexts and _looks_like_refusal(answer):
            findings.append(
                DiagnosisFinding(
                    category="generation_refusal_with_context",
                    severity="medium",
                    rationale="Relevant context was retrieved, but the model still produced a refusal or insufficient-information answer, which points more to prompting or model stability.",
                    suggested_actions=[
                        "Tighten the system prompt so retrieved evidence should lead to a direct answer first.",
                        "Compare different generation models or temperature settings on the same bad case.",
                        "Move the key supporting sentences earlier in the prompt to reduce conservative refusals.",
                    ],
                    evidence={
                        "context_count": len(contexts),
                        "answer_preview": answer[:200],
                        "finish_reason": trace.generation_finish_reason,
                    },
                )
            )
            if claim_type_counts["insufficiency"] == 0:
                findings.append(
                    DiagnosisFinding(
                        category="insufficiency_claim_missing",
                        severity="medium",
                        rationale="The answer behaves like an insufficiency or refusal response, but no structured insufficiency claim was produced.",
                        suggested_actions=[
                            "Require the generation layer to emit an [insufficiency] claim whenever the answer says evidence is insufficient.",
                            "Bind insufficiency claims to the citation labels that justify the limitation.",
                            "Track insufficiency-claim coverage in eval to avoid silent refusals.",
                        ],
                        evidence={
                            "answer_preview": answer[:200],
                            "claim_type_counts": claim_type_counts,
                        },
                    )
                )

        if retrieved and trace.reranked_chunk_ids and trace.retrieved_chunk_ids:
            if set(trace.reranked_chunk_ids) - set(trace.retrieved_chunk_ids):
                findings.append(
                    DiagnosisFinding(
                        category="rerank_inconsistent",
                        severity="high",
                        rationale="The rerank result includes chunk_ids that were not present in retrieval, which indicates inconsistent pipeline state.",
                        suggested_actions=[
                            "Check the mapping between rerank result indexes and the original hits.",
                            "Confirm that the trace record was not overwritten by a later request.",
                        ],
                        evidence={
                            "retrieved_chunk_ids": trace.retrieved_chunk_ids,
                            "reranked_chunk_ids": trace.reranked_chunk_ids,
                        },
                    )
                )

        if contexts and retrieved and len(contexts) < min(len(retrieved), 1):
            findings.append(
                DiagnosisFinding(
                    category="context_trimmed",
                    severity="low",
                    rationale="Retrieval produced results, but only a small number of final contexts remained, likely because final_contexts is too restrictive.",
                    suggested_actions=[
                        "Check whether retrieval.final_contexts is set too low.",
                    ],
                    evidence={
                        "retrieved_count": len(retrieved),
                        "context_count": len(contexts),
                        "retrieval_params": trace.retrieval_params,
                    },
                )
            )

        if not findings:
            findings.append(
                DiagnosisFinding(
                    category="no_obvious_issue",
                    severity="info",
                    rationale="No obvious structural issue is visible in this trace. If the answer is still unstable, compare it against similar eval bad cases in batch.",
                    suggested_actions=[
                        "Review historical performance for the same question in the eval set.",
                        "Run an A/B comparison across prompt versions and model versions.",
                    ],
                    evidence={
                        "retrieved_count": len(retrieved),
                        "context_count": len(contexts),
                        "finish_reason": trace.generation_finish_reason,
                    },
                )
            )

        summary = "; ".join(finding.rationale for finding in findings[:2])
        return DiagnosisResponse(
            target_type="trace",
            trace_id=trace.trace_id,
            sample_id=trace.sample_id,
            summary=summary,
            findings=findings,
        )

    def diagnose_eval_result(
        self, report: dict[str, object], result_index: int
    ) -> DiagnosisResponse:
        results = report.get("results", []) if isinstance(report, dict) else []
        if result_index < 0 or result_index >= len(results):
            raise IndexError(f"eval result index out of range: {result_index}")
        result = results[result_index]
        findings: list[DiagnosisFinding] = []

        exact_match = float(result.get("answer_exact_match", 0.0) or 0.0)
        context_recall = float(result.get("reference_context_recall", 0.0) or 0.0)
        answer = str(result.get("answer", "") or "")
        reference_answer = str(result.get("reference_answer", "") or "")

        if context_recall < 1.0:
            findings.append(
                DiagnosisFinding(
                    category="retrieval_gap",
                    severity="high",
                    rationale="The reference evidence did not appear completely in retrieved_contexts, so the primary issue is in retrieval.",
                    suggested_actions=[
                        "Check whether chunking is too coarse or too fine.",
                        "Increase retrieval top_k and inspect the rank of the correct chunk.",
                        "Introduce rerank or change the embedding model if needed.",
                    ],
                    evidence={
                        "reference_context_recall": context_recall,
                        "answer_exact_match": exact_match,
                    },
                )
            )

        if context_recall >= 1.0 and exact_match < 1.0:
            findings.append(
                DiagnosisFinding(
                    category="generation_misalignment",
                    severity="medium",
                    rationale="The retrieval evidence was complete, but the final answer still missed the reference answer, so the issue is more likely generation or evaluation criteria.",
                    suggested_actions=[
                        "Check whether the system prompt is too conservative or too verbose.",
                        "Manually inspect whether exact_match is overstating harmless formatting differences.",
                        "Consider adding semantic similarity or an LLM-judge metric instead of relying only on exact match.",
                    ],
                    evidence={
                        "reference_context_recall": context_recall,
                        "answer_exact_match": exact_match,
                        "answer_preview": answer[:200],
                        "reference_preview": reference_answer[:200],
                    },
                )
            )

        if _looks_like_refusal(answer) and context_recall >= 1.0:
            findings.append(
                DiagnosisFinding(
                    category="refusal_bad_case",
                    severity="medium",
                    rationale="This bad case is a refusal despite sufficient evidence, which usually points to prompt constraints or model instability.",
                    suggested_actions=[
                        "Add this sample to the regression set to track refusal behavior explicitly.",
                        "Strengthen the prompt instruction that matched evidence should lead to a direct answer.",
                    ],
                    evidence={"answer_preview": answer[:200]},
                )
            )

        if not findings:
            findings.append(
                DiagnosisFinding(
                    category="no_obvious_issue",
                    severity="info",
                    rationale="This eval sample does not expose an obvious structural issue.",
                    suggested_actions=["Keep analyzing clustered bad cases instead of this single sample in isolation."],
                    evidence={
                        "reference_context_recall": context_recall,
                        "answer_exact_match": exact_match,
                    },
                )
            )

        summary = "; ".join(finding.rationale for finding in findings[:2])
        return DiagnosisResponse(
            target_type="eval_result",
            trace_id=result.get("trace_id"),
            sample_id=result.get("sample_id"),
            summary=summary,
            findings=findings,
        )

    async def add_ai_suggestion(
        self, diagnosis: DiagnosisResponse, payload: dict[str, object]
    ) -> DiagnosisResponse:
        if self.generation_client is None:
            return diagnosis

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a RAG diagnosis assistant. Based on the given trace or eval diagnosis result, "
                    "produce 3 high-priority improvement suggestions that are short, concrete, and actionable."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "diagnosis": diagnosis.model_dump(),
                        "payload": payload,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            result = await self.generation_client.generate(messages)
            suggestion = str(result.get("content", "")).strip() or None
        except Exception as exc:  # pragma: no cover
            logger.warning("ai diagnosis suggestion failed: %s", exc)
            suggestion = f"AI diagnosis unavailable: {exc}"
        return diagnosis.model_copy(update={"ai_suggestion": suggestion})
