#!/bin/bash
# Continuous evaluation entry point.
#
# Two tiers, deliberately separate because they cost very different amounts:
#
#   1. Deterministic gates (scripts/run_all_gates.py). No model calls: retrieval
#      recall / MRR / nDCG, read-plan chunk selection, and the builtin answer
#      metrics (exact match, context recall, citation integrity & precision,
#      abstention, latency, cost). Cheap enough for every commit.
#   2. Archived end-to-end report gates (scripts/gate_e2e_report.py). Computed from
#      one report by rules: citation integrity / precision, answered-rate band,
#      error rate, latency. No judge is involved, on purpose - three versions of an
#      LLM judge scored the same report 67%, 67% and 54%, and sampling the disputed
#      cases showed the judge was the unreliable part.
#   3. LLM-assisted answer quality (deepeval). One model call per metric per
#      record, so it runs on the regression dataset only.
#
# A gate whose dataset is missing is SKIPPED loudly, never silently passed - and
# --strict turns that skip into a failure, which is what a release check uses.
set -e

# deepeval ships telemetry by default; eval runs must not phone out.
export DEEPEVAL_TELEMETRY_OPTOUT=true

DATASET="data/eval/rag_quality_regression.jsonl"

echo "=== 1/3 deterministic gates ==="
python -u scripts/run_all_gates.py

echo
echo "=== 2/3 archived end-to-end report gates ==="
E2E_REPORT="${E2E_REPORT:-data/eval/cnlaw/e2e270_final.json}"
if [ -f "$E2E_REPORT" ]; then
    python -u scripts/gate_e2e_report.py --report "$E2E_REPORT"
else
    echo "SKIPPED: no archived end-to-end report at $E2E_REPORT" >&2
    echo "  (run the pipeline over data/eval/cnlaw/title_blind_queries.jsonl to produce one)" >&2
fi

echo
echo "=== 3/3 LLM-assisted answer quality ==="
# Ensure the dataset exists; generate it from negative feedback if missing.
if [ ! -f "$DATASET" ]; then
    echo "Regression dataset not found. Generating from negative feedback..."
    python scripts/generate_regression_dataset.py
fi

# Thresholds map to metrics emitted by app/eval/deepeval_runner.py's aggregate:
# faithfulness, answer_relevancy, context_precision, context_recall, plus the
# deterministic ones.
# No silent dataset substitution: swapping in a different dataset made the gate
# meaningless ("CI Evaluation passed" on data it was not asked to evaluate).
if [ ! -f "$DATASET" ]; then
    echo "ERROR: regression dataset $DATASET could not be produced." >&2
    echo "Record negative feedback (or commit a dataset) before running the gate." >&2
    exit 1
fi

python -u scripts/run_eval.py \
    --dataset "$DATASET" \
    --mode deepeval \
    --min-answer-relevancy 0.7 \
    --min-context-recall 0.6 \
    --min-citation-integrity 0.9

echo
echo "CI evaluation passed."
