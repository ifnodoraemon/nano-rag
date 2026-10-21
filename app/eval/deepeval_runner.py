import asyncio
import logging
import os
import re
from collections import Counter

from app.utils.coerce import as_float as _as_float
from app.utils.coerce import as_int as _as_int
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Deterministic match thresholds: a reference context counts as recalled when
# every numeric token of the reference appears in a retrieved context and the
# character overlap is at least this ratio (table rows are re-serialized into
# "header=value" chunks, so byte-exact containment would always fail).
_CONTEXT_MATCH_OVERLAP = 0.75

# Bounds for the deepeval judge model (one call per metric per record).
_EVAL_LLM_TIMEOUT_SECONDS = float(os.getenv("RAG_EVAL_LLM_TIMEOUT_SECONDS", "120"))
_EVAL_LLM_MAX_RETRIES = int(os.getenv("RAG_EVAL_LLM_MAX_RETRIES", "1"))


# Provenance markers that carry no answer content: the formatter appends
# "[C1]" style labels to every cited sentence. Comparing them would make every
# real answer "wrong".
_CITATION_MARKER_RE = re.compile(r"\[\s*C\d+\s*\]|\[\s*C\d+(?:\s*,\s*C\d+)*\s*\]", re.IGNORECASE)


def _strip_citation_markers(text: str) -> str:
    return _CITATION_MARKER_RE.sub(" ", str(text))


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


def _answer_exact_match(answer: str, reference_answer: str) -> float | None:
    """1.0 when the answer and the reference carry the same token multiset.

    Multiset equality, not set containment. Containment scored a *contradicting*
    answer 1.0 — "员工 不 报销 15 天 内 无需 提交" contains every token of the
    reference "15 天 内 提交" — which silently inflated the primary quality
    metric. Extra tokens (negations, hedges) now make it a non-match; semantic
    paraphrase is what the LLM-assisted deepeval metrics are for.

    Returns None when the record carries no reference answer: "no reference"
    is not "wrong answer", and folding it into the mean hides missing data.
    """
    if not str(reference_answer or "").strip():
        return None
    reference = Counter(_normalize_tokens(_strip_citation_markers(reference_answer)))
    if not reference:
        return None
    answer_tokens = Counter(_normalize_tokens(_strip_citation_markers(answer)))
    return 1.0 if reference == answer_tokens else 0.0


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


def _reference_context_recall(
    reference_contexts: list[str], retrieved: list[str]
) -> float | None:
    """Fraction of reference contexts recalled by the retrieved contexts.

    Returns None (not 0.0) when the record has no reference contexts: a dataset
    that omits references must not be scored as "recall totally failed", and a
    gate on it must fail as *missing data* rather than pass on a fabricated
    number.
    """
    if not reference_contexts:
        return None
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
            # An eval of hundreds of records must not hang forever on one
            # upstream call: bound it and let the per-metric error path record
            # the failure (with partial coverage now failing the gate).
            self.client = AsyncOpenAI(
                base_url=gc.base_url,
                api_key=gc.api_key,
                timeout=_EVAL_LLM_TIMEOUT_SECONDS,
                max_retries=_EVAL_LLM_MAX_RETRIES,
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

            row_metrics: dict[str, float] = {}
            row_errors: dict[str, str] = {}

            async def evaluate_metric(metric, case):
                # Bind the case explicitly: the closure over the loop variable
                # would measure whichever record the loop had reached.
                try:
                    await metric.a_measure(case)
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
                    return metric.__class__.__name__.replace("Metric", "").lower(), None

            metric_results = await asyncio.gather(
                *(evaluate_metric(m, test_case) for m in metrics)
            )
            for name, score in metric_results:
                if score is None:
                    # Keep the failure visible on the row (and countable by the
                    # gate) instead of dropping the key and letting the mean
                    # over survivors look healthy.
                    row_errors[name] = "metric_error"
                else:
                    row_metrics[name] = score

            entry = {**record, **_builtin_metrics(record), **row_metrics}
            if row_errors:
                entry["metric_errors"] = row_errors
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
        agg: dict[str, object] = {}
        coverage: dict[str, float] = {}
        for key in [
            "answer_exact_match",
            "reference_context_recall",
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]:
            scores = [
                value
                for r in results
                if (value := r.get(key)) is not None
            ]
            if scores:
                numeric = [_as_float(value) for value in scores]
                agg[key] = round(sum(numeric) / len(numeric), 4)
                # Fraction of rows that actually produced this metric. A gate
                # that only looks at the mean of survivors passes when 9/10
                # rows errored out; the gate reads this to fail instead.
                coverage[key] = round(len(scores) / total, 4)
        agg["metric_coverage"] = coverage
        agg["records_without_metric"] = {
            key: total - int(round(value * total)) for key, value in coverage.items()
        }
        agg["retrieved_context_count_avg"] = round(
            sum(len(r.get("retrieved_contexts") or []) for r in results) / total, 4
        )
        # Conflict stats from the materialized records so the
        # --max-conflicting-hit-rate gate works in deterministic mode too.
        conflicting_counts = [
            _as_int(r.get("conflicting_context_count")) for r in results
        ]
        agg["conflicting_context_count_avg"] = round(
            sum(conflicting_counts) / total, 4
        )
        agg["conflicting_hit_rate"] = round(
            sum(1 for value in conflicting_counts if _as_int(value) > 0) / total, 4
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
