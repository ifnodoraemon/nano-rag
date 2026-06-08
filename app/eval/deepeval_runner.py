import asyncio
import logging
import math
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.utils.text import normalize_text, normalize_for_comparison

if TYPE_CHECKING:
    from app.model_client.generation import GenerationClient

logger = logging.getLogger(__name__)

def _build_llm(generation_client):
    from deepeval.models.base_model import DeepEvalBaseLLM
    from openai import AsyncOpenAI
    
    class CustomOpenAI(DeepEvalBaseLLM):
        def __init__(self, gc):
            self.model = gc.alias
            self.client = AsyncOpenAI(
                base_url=gc.base_url,
                api_key=gc.api_key,
            )

        def load_model(self):
            return self.client

        def generate(self, prompt: str) -> str:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
            return loop.run_until_complete(self.a_generate(prompt))

        async def a_generate(self, prompt: str) -> str:
            res = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return res.choices[0].message.content

        def get_model_name(self):
            return self.model

    return CustomOpenAI(generation_client)


class DeepevalRunner:
    def __init__(
        self,
        generation_client=None,
    ) -> None:
        self.generation_client = generation_client

    def run(self, records: list[dict]) -> dict:
        if not records:
            return self._empty_result()
        return asyncio.run(self.run_async(records))

    async def run_async(self, records: list[dict]) -> dict:
        from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric
        from deepeval.test_case import LLMTestCase
        
        if not records:
            return self._empty_result()

        llm = _build_llm(self.generation_client)
        
        metrics = [
            FaithfulnessMetric(model=llm, include_reason=False),
            AnswerRelevancyMetric(model=llm, include_reason=False),
            ContextualPrecisionMetric(model=llm, include_reason=False),
            ContextualRecallMetric(model=llm, include_reason=False)
        ]

        results = []
        for record in records:
            contexts = record.get("retrieved_contexts", []) or []
            if contexts and isinstance(contexts[0], dict):
                contexts = [str(c.get("text", c)) for c in contexts]
            else:
                contexts = [str(c) for c in contexts]

            reference_contexts = record.get("reference_contexts", []) or []
            reference_contexts_strs = [str(c) for c in reference_contexts] if reference_contexts else None

            test_case = LLMTestCase(
                input=str(record.get("query", "")),
                actual_output=str(record.get("answer", "")),
                expected_output=str(record.get("reference_answer", "")) if record.get("reference_answer") else None,
                retrieval_context=contexts,
            )
            
            row_metrics = {}
            for metric in metrics:
                try:
                    await metric.a_measure(test_case)
                    name = metric.__class__.__name__.replace("Metric", "").lower()
                    if name == "contextualprecision":
                        name = "context_precision"
                    elif name == "contextualrecall":
                        name = "context_recall"
                    elif name == "answerrelevancy":
                        name = "answer_relevancy"
                    row_metrics[name] = round(metric.score, 4)
                except Exception as e:
                    logger.error(f"Error computing metric {metric.__class__.__name__}: {e}")
            
            entry = {**self._builtin_fields(record), **row_metrics}
            results.append(entry)

        total = len(results)
        aggregate = self._aggregate(results, total)

        return {
            "status": "ok",
            "records": total,
            "aggregate": aggregate,
            "results": results,
        }

    def _builtin_fields(self, record: dict) -> dict:
        return {
            "sample_id": record.get("sample_id"),
            "query": str(record.get("query", "")),
            "answer": record.get("answer", ""),
            "reference_answer": record.get("reference_answer", ""),
            "retrieved_context_count": len(record.get("retrieved_contexts", []) or [])
        }

    def _aggregate(self, results: list[dict], total: int) -> dict:
        if total == 0:
            return {}
        agg = {}
        for key in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
            scores = [r.get(key) for r in results if r.get(key) is not None]
            if scores:
                agg[key] = round(sum(scores) / len(scores), 4)
        agg["retrieved_context_count_avg"] = round(sum(r.get("retrieved_context_count", 0) for r in results) / total, 4)
        return agg

    def _empty_result(self) -> dict:
        return {
            "status": "ok",
            "records": 0,
            "aggregate": {},
            "results": [],
        }
