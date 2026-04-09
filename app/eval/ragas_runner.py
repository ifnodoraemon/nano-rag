from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from app.eval.ragas_metrics import (
    RAGASConfig,
    evaluate_ragas_metrics,
)
from app.utils.constants import MIN_CONTEXT_MATCH_LENGTH
from app.utils.text import normalize_text, normalize_for_comparison

if TYPE_CHECKING:
    from app.model_client.generation import GenerationClient


class RagasRunner:
    def __init__(
        self,
        generation_client: GenerationClient | None = None,
        config: RAGASConfig | None = None,
    ) -> None:
        self.generation_client = generation_client
        self.config = config or RAGASConfig.from_env()

    def run(self, records: list[dict]) -> dict:
        return self._run_sync(records)

    def _run_sync(self, records: list[dict]) -> dict:
        if not records:
            return self._empty_result()

        results: list[dict] = []
        exact_matches = 0.0
        context_recalls = 0.0
        retrieved_counts = 0.0
        conflicting_counts = 0.0
        conflicting_hits = 0.0
        conflict_claim_counts = 0.0
        insufficiency_claim_counts = 0.0
        conflict_claim_hits = 0.0
        insufficiency_claim_hits = 0.0

        for record in records:
            answer = normalize_text(record.get("answer", ""))
            reference_answer = normalize_text(record.get("reference_answer", ""))
            contexts = record.get("retrieved_contexts", []) or []
            reference_contexts = record.get("reference_contexts", []) or []
            query = str(record.get("query", ""))
            conflicting_context_count = int(
                record.get("conflicting_context_count", 0) or 0
            )
            conflict_claim_count = int(record.get("conflict_claim_count", 0) or 0)
            insufficiency_claim_count = int(
                record.get("insufficiency_claim_count", 0) or 0
            )

            normalized_answer = normalize_for_comparison(answer)
            normalized_reference = normalize_for_comparison(reference_answer)
            exact_match = (
                1.0
                if normalized_answer and normalized_answer == normalized_reference
                else 0.0
            )
            if reference_contexts:
                matched = 0
                merged_contexts = "\n".join(str(item) for item in contexts)
                for reference_context in reference_contexts:
                    ref_text = str(reference_context).strip()
                    if (
                        len(ref_text) >= MIN_CONTEXT_MATCH_LENGTH
                        and ref_text in merged_contexts
                    ):
                        matched += 1
                context_recall = matched / len(reference_contexts)
            else:
                context_recall = 0.0

            result_entry = {
                "sample_id": record.get("sample_id"),
                "trace_id": record.get("trace_id"),
                "query": query,
                "answer_exact_match": exact_match,
                "reference_context_recall": round(context_recall, 4),
                "retrieved_context_count": len(contexts),
                "conflicting_context_count": conflicting_context_count,
                "conflict_claim_count": conflict_claim_count,
                "insufficiency_claim_count": insufficiency_claim_count,
                "answer": record.get("answer", ""),
                "reference_answer": record.get("reference_answer", ""),
            }
            results.append(result_entry)
            exact_matches += exact_match
            context_recalls += context_recall
            retrieved_counts += len(contexts)
            conflicting_counts += conflicting_context_count
            conflicting_hits += 1.0 if conflicting_context_count > 0 else 0.0
            conflict_claim_counts += conflict_claim_count
            insufficiency_claim_counts += insufficiency_claim_count
            conflict_claim_hits += 1.0 if conflict_claim_count > 0 else 0.0
            insufficiency_claim_hits += (
                1.0 if insufficiency_claim_count > 0 else 0.0
            )

        total = len(records)
        aggregate = {
            "answer_exact_match": round(exact_matches / total, 4),
            "reference_context_recall": round(context_recalls / total, 4),
            "retrieved_context_count_avg": round(retrieved_counts / total, 4),
            "conflicting_context_count_avg": round(conflicting_counts / total, 4),
            "conflicting_hit_rate": round(conflicting_hits / total, 4),
            "conflict_claim_count_avg": round(conflict_claim_counts / total, 4),
            "conflict_claim_hit_rate": round(conflict_claim_hits / total, 4),
            "insufficiency_claim_count_avg": round(
                insufficiency_claim_counts / total, 4
            ),
            "insufficiency_claim_hit_rate": round(
                insufficiency_claim_hits / total, 4
            ),
        }

        return {
            "status": "ok",
            "records": total,
            "aggregate": aggregate,
            "results": results,
        }

    async def run_async(self, records: list[dict]) -> dict:
        if not records:
            return self._empty_result()

        if not self.generation_client or not self.config.enable_llm_judge:
            return self._run_sync(records)

        semaphore = asyncio.Semaphore(self.config.max_concurrent_judgments)

        async def evaluate_record(record: dict) -> dict:
            query = str(record.get("query", ""))
            answer = normalize_text(record.get("answer", ""))
            reference_answer = normalize_text(record.get("reference_answer", ""))
            contexts = record.get("retrieved_contexts", []) or []
            reference_contexts = record.get("reference_contexts", []) or []

            metrics = await evaluate_ragas_metrics(
                self.generation_client,
                query,
                answer,
                contexts,
                reference_contexts,
                reference_answer,
                self.config,
            )

            return {
                "sample_id": record.get("sample_id"),
                "trace_id": record.get("trace_id"),
                "query": query,
                "answer_exact_match": metrics.answer_exact_match,
                "reference_context_recall": round(metrics.reference_context_recall, 4),
                "faithfulness": round(metrics.faithfulness, 4),
                "answer_relevance": round(metrics.answer_relevance, 4),
                "context_precision": round(metrics.context_precision, 4),
                "retrieved_context_count": len(contexts),
                "conflicting_context_count": int(
                    record.get("conflicting_context_count", 0) or 0
                ),
                "conflict_claim_count": int(record.get("conflict_claim_count", 0) or 0),
                "insufficiency_claim_count": int(
                    record.get("insufficiency_claim_count", 0) or 0
                ),
                "answer": record.get("answer", ""),
                "reference_answer": record.get("reference_answer", ""),
            }

        async def bounded_evaluate(record: dict) -> dict:
            async with semaphore:
                return await evaluate_record(record)

        results = await asyncio.gather(*[bounded_evaluate(r) for r in records])
        total = len(results)
        aggregate = {
            "answer_exact_match": round(
                sum(r["answer_exact_match"] for r in results) / total, 4
            ),
            "reference_context_recall": round(
                sum(r["reference_context_recall"] for r in results) / total, 4
            ),
            "retrieved_context_count_avg": round(
                sum(r["retrieved_context_count"] for r in results) / total, 4
            ),
            "conflicting_context_count_avg": round(
                sum(r.get("conflicting_context_count", 0) for r in results) / total, 4
            ),
            "conflicting_hit_rate": round(
                sum(1 for r in results if r.get("conflicting_context_count", 0) > 0)
                / total,
                4,
            ),
            "conflict_claim_count_avg": round(
                sum(r.get("conflict_claim_count", 0) for r in results) / total, 4
            ),
            "conflict_claim_hit_rate": round(
                sum(1 for r in results if r.get("conflict_claim_count", 0) > 0)
                / total,
                4,
            ),
            "insufficiency_claim_count_avg": round(
                sum(r.get("insufficiency_claim_count", 0) for r in results) / total, 4
            ),
            "insufficiency_claim_hit_rate": round(
                sum(1 for r in results if r.get("insufficiency_claim_count", 0) > 0)
                / total,
                4,
            ),
            "faithfulness_avg": round(
                sum(r.get("faithfulness", 0) for r in results) / total, 4
            ),
            "answer_relevance_avg": round(
                sum(r.get("answer_relevance", 0) for r in results) / total, 4
            ),
            "context_precision_avg": round(
                sum(r.get("context_precision", 0) for r in results) / total, 4
            ),
        }

        return {
            "status": "ok",
            "records": total,
            "aggregate": aggregate,
            "results": results,
        }

    def _empty_result(self) -> dict:
        return {
            "status": "ok",
            "records": 0,
            "aggregate": {
                "answer_exact_match": 0.0,
                "reference_context_recall": 0.0,
                "retrieved_context_count_avg": 0.0,
                "conflicting_context_count_avg": 0.0,
                "conflicting_hit_rate": 0.0,
                "conflict_claim_count_avg": 0.0,
                "conflict_claim_hit_rate": 0.0,
                "insufficiency_claim_count_avg": 0.0,
                "insufficiency_claim_hit_rate": 0.0,
            },
            "results": [],
        }
