from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.core.exceptions import DiagnosisError, StoreError
from app.schemas.diagnosis import DiagnosisFinding, DiagnosisResponse
from app.schemas.trace import TraceRecord

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from app.model_client.generation import GenerationClient


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
                # One root cause, one finding. These used to be two separate
                # checks on the same predicate, so every conflicting retrieval
                # was counted twice in diagnosis_counts and the report showed
                # two symptoms for a single defect.
                findings.append(
                    DiagnosisFinding(
                        category="conflict_not_reflected_in_answer",
                        severity="high",
                        rationale=(
                            "Conflicting evidence was retrieved, but the answer carried no "
                            "structured [conflict] claim, so the disagreement is invisible to "
                            "the reader and the answer reads as more certain than the evidence."
                        ),
                        suggested_actions=[
                            "Require the generation layer to emit at least one [conflict] supporting claim when conflicting evidence is present.",
                            "Tighten the prompt so conflicting topics must be acknowledged explicitly.",
                            "When conflicts cannot be resolved, downgrade to a clarification-style answer.",
                        ],
                        evidence={
                            "conflicting_context_count": len(conflicting_contexts),
                            "conflicting_chunk_ids": [
                                context.get("chunk_id")
                                for context in conflicting_contexts
                            ],
                            "claim_type_counts": claim_type_counts,
                            "answer_preview": answer[:200],
                        },
                    )
                )

        if not retrieved:
            findings.append(
                DiagnosisFinding(
                    category="retrieval_empty",
                    severity="high",
                    rationale="The retrieval stage returned no candidate documents, so the likely issue is ingestion or discovery (wiki/BM25) configuration.",
                    suggested_actions=[
                        "Confirm that documents were ingested successfully and wiki source pages exist.",
                        "Check that the BM25 discovery index has indexed the expected kb_id.",
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

        elif contexts and claim_type_counts["insufficiency"] > 0:
            findings.append(
                DiagnosisFinding(
                    category="generation_refusal_with_context",
                    severity="medium",
                    rationale="Relevant context was retrieved, but the model emitted a structured insufficiency claim, which points more to prompting, model stability, or missing evidence binding.",
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
        raw_results: object = report.get("results", []) if isinstance(report, dict) else []
        if not isinstance(raw_results, list):
            raise StoreError(
                "eval report has a malformed 'results' field (expected a list)"
            )
        results = raw_results
        if result_index < 0 or result_index >= len(results):
            raise IndexError(f"eval result index out of range: {result_index}")
        result = results[result_index]
        if not isinstance(result, dict):
            raise StoreError("eval report row is not an object")
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
                        "Increase retrieval top_k and inspect the rank of the correct document.",
                        "Review the wiki discovery layer: page summaries, topic pages, and version metadata.",
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
        except Exception as exc:  # pragma: no cover - provider failure
            # Log the detail server-side; never echo raw upstream error text to
            # the caller (provider 4xx bodies can contain the key or URL), and
            # never present a failed AI call as a diagnosis result.
            logger.exception("ai diagnosis suggestion failed")
            raise DiagnosisError(
                "AI diagnosis suggestion failed; retry with include_ai disabled "
                "or check the generation provider configuration."
            ) from exc
        return diagnosis.model_copy(update={"ai_suggestion": suggestion})
