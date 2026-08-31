import asyncio
import logging
import re
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Deterministic match thresholds: a reference context counts as recalled when
# every numeric token of the reference appears in a retrieved context and the
# character overlap is at least this ratio (table rows are re-serialized into
# "header=value" chunks, so byte-exact containment would always fail).
_CONTEXT_MATCH_OVERLAP = 0.75


def _normalize_tokens(text: str) -> list[str]:
    """Deterministic tokenization shared by the exact-match and recall
    metrics: CJK characters as single tokens plus lowercase alphanumeric runs
    (numbers stay whole so 31800 does not fuzz-match 3180)."""
    if not text:
        return []
    tokens: list[str] = []
    for match in re.finditer(r"[0-9a-zA-Z]+|[\u4e00-\u9fff]", str(text).casefold()):
        tokens.append(match.group(0))
    return tokens


def _token_set(text: str) -> set[str]:
    return set(_normalize_tokens(text))


def _numeric_tokens(text: str) -> set[str]:
    return {token for token in _token_set(text) if token.isdigit()}


def _answer_exact_match(answer: str, reference_answer: str) -> float:
    """1.0 when every token of the reference answer (numbers included)
    appears in the generated answer, else 0.0. A strict byte-equality check
    would score every harmless paraphrase as a failure."""
    if not reference_answer:
        return 0.0
    reference = _token_set(reference_answer)
    if not reference:
        return 0.0
    answer = _token_set(answer)
    return 1.0 if reference <= answer else 0.0


def _context_matched(reference: str, retrieved: list[str]) -> bool:
    ref_tokens = _token_set(reference)
    if not ref_tokens:
        return False
    ref_numbers = _numeric_tokens(reference)
    for context in retrieved:
        ctx_tokens = _token_set(context)
        if not ref_numbers <= ctx_tokens:
            continue
        overlap = len(ref_tokens & ctx_tokens) / len(ref_tokens)
        if overlap >= _CONTEXT_MATCH_OVERLAP:
            return True
    return False


def _reference_context_recall(reference_contexts: list[str], retrieved: list[str]) -> float:
    """Fraction of reference contexts recalled by the retrieved contexts."""
    if not reference_contexts:
        return 0.0
    matched = sum(
        1 for reference in reference_contexts if _context_matched(reference, retrieved)
    )
    return round(matched / len(reference_contexts), 4)


def _builtin_metrics(record: dict) -> dict:
    """Deterministic, LLM-free metrics computed for every record in both
    modes. They are the inputs of the benchmark bad-case count, the
    diagnosis service and the run_eval gates, so they must exist in the
    no-LLM mode too."""
    retrieved = [
        str(context.get("text", context)) if isinstance(context, dict) else str(context)
        for context in (record.get("retrieved_contexts") or [])
    ]
    reference_contexts = [
        str(context)
        for context in (record.get("reference_contexts") or [])
    ]
    return {
        "answer_exact_match": _answer_exact_match(
            str(record.get("answer", "") or ""),
            str(record.get("reference_answer", "") or ""),
        ),
        "reference_context_recall": _reference_context_recall(
            reference_contexts, retrieved
        ),
    }


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
    """Two evaluation modes over the same record pipeline:

    - ``run`` (deterministic): builtin metrics only, no LLM calls, no
      deepeval import. This is what ``--no-ragas-lib`` runs, and it is a real
      mode — the previous implementation made the same LLM-metric calls as
      ``run_async``.
    - ``run_async`` (LLM-assisted): builtin metrics plus deepeval
      faithfulness/relevancy/precision/recall.

    Both modes pass the whole input record through to the results (the
    previous 5-field projection dropped trace_id/kb_id, which made the
    benchmark's latency/model lookups, the kb filter and the bad-case/diagno
    sis logic silently wrong for every row).
    """

    def __init__(
        self,
        generation_client=None,
    ) -> None:
        self.generation_client = generation_client

    def run(self, records: list[dict]) -> dict:
        """Deterministic builtin metrics only (no LLM, no deepeval)."""
        if not records:
            return self._empty_result()
        results: list[dict] = []
        for record in records:
            # Full passthrough: trace_id, kb_id, conflicting counts, claim
            # stats and everything else materialized upstream must survive
            # into the report rows.
            entry = {**record, **_builtin_metrics(record)}
            results.append(entry)
        return {
            "status": "ok",
            "mode": "deterministic",
            "records": len(results),
            "aggregate": self._aggregate(results),
            "results": results,
        }

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
        for i, record in enumerate(records):
            logger.info(f"Evaluating record {i+1}/{len(records)}: {record.get('query')}")
            contexts = record.get("retrieved_contexts", []) or []
            if contexts and isinstance(contexts[0], dict):
                contexts = [str(c.get("text", c)) for c in contexts]
            else:
                contexts = [str(c) for c in contexts]

            test_case = LLMTestCase(
                input=str(record.get("query", "")),
                actual_output=str(record.get("answer", "")),
                expected_output=str(record.get("reference_answer", "")) if record.get("reference_answer") else None,
                retrieval_context=contexts,
            )

            row_metrics = {}

            async def evaluate_metric(metric):
                try:
                    await metric.a_measure(test_case)
                    name = metric.__class__.__name__.replace("Metric", "").lower()
                    if name == "contextualprecision":
                        name = "context_precision"
                    elif name == "contextualrecall":
                        name = "context_recall"
                    elif name == "answerrelevancy":
                        name = "answer_relevancy"
                    return name, round(metric.score, 4)
                except Exception as e:
                    logger.error(f"Error computing metric {metric.__class__.__name__}: {e}")
                    return None, None

            metric_results = await asyncio.gather(*(evaluate_metric(m) for m in metrics))
            for name, score in metric_results:
                if name:
                    row_metrics[name] = score

            entry = {**record, **_builtin_metrics(record), **row_metrics}
            results.append(entry)

        return {
            "status": "ok",
            "mode": "llm-assisted",
            "records": len(results),
            "aggregate": self._aggregate(results),
            "results": results,
        }

    def _aggregate(self, results: list[dict]) -> dict:
        total = len(results)
        if total == 0:
            return {}
        agg = {}
        for key in [
            "answer_exact_match",
            "reference_context_recall",
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]:
            scores = [r.get(key) for r in results if r.get(key) is not None]
            if scores:
                agg[key] = round(sum(scores) / len(scores), 4)
        agg["retrieved_context_count_avg"] = round(
            sum(len(r.get("retrieved_contexts") or []) for r in results) / total, 4
        )
        # Conflict stats from the materialized records so the
        # --max-conflicting-hit-rate gate works in deterministic mode too.
        conflicting_counts = [
            int(r.get("conflicting_context_count", 0) or 0) for r in results
        ]
        agg["conflicting_context_count_avg"] = round(
            sum(conflicting_counts) / total, 4
        )
        agg["conflicting_hit_rate"] = round(
            sum(1 for value in conflicting_counts if value > 0) / total, 4
        )
        return agg

    def _empty_result(self) -> dict:
        return {
            "status": "ok",
            "mode": "deterministic",
            "records": 0,
            "aggregate": {},
            "results": [],
        }
