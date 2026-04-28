from __future__ import annotations

import asyncio
import logging
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.utils.constants import MIN_CONTEXT_MATCH_LENGTH
from app.utils.text import normalize_text, normalize_for_comparison

if TYPE_CHECKING:
    from app.model_client.generation import GenerationClient

logger = logging.getLogger(__name__)

_METRIC_CLASS_NAMES = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "AnswerRelevancy",
    "context_precision": "ContextPrecision",
    "context_recall": "ContextRecall",
}


def _load_metric_classes() -> dict[str, type]:
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    return {
        "AnswerRelevancy": AnswerRelevancy,
        "ContextPrecision": ContextPrecision,
        "ContextRecall": ContextRecall,
        "Faithfulness": Faithfulness,
    }


@dataclass
class RAGASConfig:
    enable_llm_judge: bool = True
    llm_judge_model: str = "gemini-2.5-flash"
    max_concurrent_judgments: int = 3
    lib_metrics: list[str] = field(default_factory=lambda: [
        "faithfulness", "answer_relevancy", "context_precision",
    ])
    lib_llm_model: str = "gemini-2.5-flash"
    lib_llm_base_url: str | None = None
    lib_llm_api_key: str | None = None

    @classmethod
    def from_env(cls) -> "RAGASConfig":
        raw_metrics = os.getenv(
            "RAG_RAGAS_LIB_METRICS",
            "faithfulness,answer_relevancy,context_precision",
        )
        metrics = [m.strip() for m in raw_metrics.split(",") if m.strip()]
        return cls(
            enable_llm_judge=os.getenv("RAG_RAGAS_LLM_JUDGE", "true").lower()
            in ("true", "1", "yes"),
            llm_judge_model=os.getenv("RAG_RAGAS_JUDGE_MODEL", "gemini-2.5-flash"),
            max_concurrent_judgments=int(os.getenv("RAG_RAGAS_MAX_CONCURRENT", "3")),
            lib_metrics=metrics,
            lib_llm_model=os.getenv("RAG_RAGAS_LIB_LLM_MODEL", "gemini-2.5-flash"),
            lib_llm_base_url=os.getenv("RAG_RAGAS_LIB_LLM_BASE_URL"),
            lib_llm_api_key=os.getenv("RAG_RAGAS_LIB_LLM_API_KEY"),
        )


def _resolve_metrics(requested: list[str], llm) -> list:
    metric_classes = _load_metric_classes()
    resolved = []
    for name in requested:
        key = name.lower().strip()
        class_name = _METRIC_CLASS_NAMES.get(key)
        cls = metric_classes.get(class_name or "")
        if cls is None:
            logger.warning("unknown ragas metric: %s, skipping", name)
        else:
            resolved.append(cls(llm=llm))
    return resolved


def _build_llm(config: RAGASConfig, generation_client: GenerationClient | None):
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    base_url = config.lib_llm_base_url
    api_key = config.lib_llm_api_key
    model = config.lib_llm_model

    if base_url and api_key:
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        return llm_factory(model, client=client)

    if generation_client:
        client = AsyncOpenAI(
            base_url=generation_client.base_url,
            api_key=generation_client.api_key,
        )
        return llm_factory(generation_client.alias, client=client)

    api_key = os.getenv("RAG_RAGAS_LIB_DEFAULT_API_KEY", "")
    if not api_key:
        raise ValueError(
            "RAGAS lib requires an OpenAI-compatible client. "
            "Set RAG_RAGAS_LIB_LLM_BASE_URL + RAG_RAGAS_LIB_LLM_API_KEY, "
            "or provide a generation_client, or set RAG_RAGAS_LIB_DEFAULT_API_KEY."
        )
    client = AsyncOpenAI(api_key=api_key)
    return llm_factory(model, client=client)


class RagasRunner:
    def __init__(
        self,
        generation_client: GenerationClient | None = None,
        config: RAGASConfig | None = None,
    ) -> None:
        self.generation_client = generation_client
        self.config = config or RAGASConfig.from_env()

    def run(self, records: list[dict]) -> dict:
        if not records:
            return self._empty_result()
        return self._compute(records)

    async def run_async(self, records: list[dict]) -> dict:
        if not records:
            return self._empty_result()
        return await self._run_async(records)

    def _compute(self, records: list[dict]) -> dict:
        results = [self._builtin_fields(r) for r in records]
        total = len(results)
        return {
            "status": "ok",
            "records": total,
            "aggregate": self._aggregate(results, total),
            "results": results,
        }

    async def _run_async(self, records: list[dict]) -> dict:
        from ragas import evaluate
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

        samples: list[SingleTurnSample] = []
        for record in records:
            contexts = record.get("retrieved_contexts", []) or []
            if contexts and isinstance(contexts[0], dict):
                contexts = [str(c.get("text", c)) for c in contexts]
            else:
                contexts = [str(c) for c in contexts]

            reference_contexts = record.get("reference_contexts", []) or []
            samples.append(
                SingleTurnSample(
                    user_input=str(record.get("query", "")),
                    response=str(record.get("answer", "")),
                    reference=record.get("reference_answer") or None,
                    retrieved_contexts=contexts,
                    reference_contexts=[str(c) for c in reference_contexts] if reference_contexts else None,
                )
            )

        dataset = EvaluationDataset(samples=samples)
        llm = _build_llm(self.config, self.generation_client)
        metrics = _resolve_metrics(self.config.lib_metrics, llm)
        if not metrics:
            logger.error("no valid ragas metrics resolved from: %s", self.config.lib_metrics)
            return self._empty_result()

        result = evaluate(dataset=dataset, metrics=metrics)
        return self._format_result(records, result)

    def _format_result(self, records: list[dict], eval_result) -> dict:
        try:
            if hasattr(eval_result, "to_pandas"):
                per_record = eval_result.to_pandas().to_dict("records")
            else:
                scores = eval_result.scores if hasattr(eval_result, "scores") else None
                if isinstance(scores, list):
                    per_record = scores
                elif scores is not None and hasattr(scores, "to_pandas"):
                    per_record = scores.to_pandas().to_dict("records")
                elif scores is not None and hasattr(scores, "to_dict"):
                    per_record = scores.to_dict("records")
                else:
                    per_record = [{} for _ in records]
        except Exception:
            per_record = [{} for _ in records]

        if not isinstance(per_record, list):
            per_record = [{} for _ in records]

        results: list[dict] = []
        lib_sums: dict[str, float] = {}

        for index, record in enumerate(records):
            row = per_record[index] if index < len(per_record) else {}
            if not isinstance(row, dict):
                row = {}
            row_metrics = {}
            for key, value in row.items():
                score = self._coerce_metric_score(value)
                if score is None:
                    continue
                row_metrics[key] = round(score, 4)
                lib_sums[key] = lib_sums.get(key, 0.0) + score

            builtin = self._builtin_fields(record)
            entry = {**builtin, **row_metrics}
            results.append(entry)

        total = len(records)
        aggregate = self._aggregate(results, total)
        for key, total_score in lib_sums.items():
            if key not in aggregate:
                aggregate[key] = round(total_score / total, 4)

        return {
            "status": "ok",
            "records": total,
            "aggregate": aggregate,
            "results": results,
        }

    def _coerce_metric_score(self, value) -> float | None:  # noqa: ANN001
        if value is None or isinstance(value, bool):
            return None
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(score):
            return None
        return score

    def _builtin_fields(self, record: dict) -> dict:
        answer = normalize_text(record.get("answer", ""))
        reference_answer = normalize_text(record.get("reference_answer", ""))
        contexts = record.get("retrieved_contexts", []) or []
        reference_contexts = record.get("reference_contexts", []) or []
        na = normalize_for_comparison(answer)
        nr = normalize_for_comparison(reference_answer)
        exact_match = 1.0 if na and na == nr else 0.0
        if reference_contexts:
            merged = "\n".join(str(item) for item in contexts)
            matched = sum(
                1
                for ref in reference_contexts
                if len(str(ref).strip()) >= MIN_CONTEXT_MATCH_LENGTH and str(ref).strip() in merged
            )
            context_recall = matched / len(reference_contexts)
        else:
            context_recall = 0.0
        return {
            "sample_id": record.get("sample_id"),
            "trace_id": record.get("trace_id"),
            "query": str(record.get("query", "")),
            "answer_exact_match": exact_match,
            "reference_context_recall": round(context_recall, 4),
            "retrieved_context_count": len(contexts),
            "conflicting_context_count": int(record.get("conflicting_context_count", 0) or 0),
            "conflict_claim_count": int(record.get("conflict_claim_count", 0) or 0),
            "insufficiency_claim_count": int(record.get("insufficiency_claim_count", 0) or 0),
            "answer": record.get("answer", ""),
            "reference_answer": record.get("reference_answer", ""),
        }

    def _aggregate(self, results: list[dict], total: int) -> dict:
        if total == 0:
            return {}
        return {
            "answer_exact_match": round(sum(r["answer_exact_match"] for r in results) / total, 4),
            "reference_context_recall": round(sum(r["reference_context_recall"] for r in results) / total, 4),
            "retrieved_context_count_avg": round(sum(r["retrieved_context_count"] for r in results) / total, 4),
            "conflicting_context_count_avg": round(sum(r.get("conflicting_context_count", 0) for r in results) / total, 4),
            "conflicting_hit_rate": round(sum(1 for r in results if r.get("conflicting_context_count", 0) > 0) / total, 4),
            "conflict_claim_count_avg": round(sum(r.get("conflict_claim_count", 0) for r in results) / total, 4),
            "conflict_claim_hit_rate": round(sum(1 for r in results if r.get("conflict_claim_count", 0) > 0) / total, 4),
            "insufficiency_claim_count_avg": round(sum(r.get("insufficiency_claim_count", 0) for r in results) / total, 4),
            "insufficiency_claim_hit_rate": round(sum(1 for r in results if r.get("insufficiency_claim_count", 0) > 0) / total, 4),
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
