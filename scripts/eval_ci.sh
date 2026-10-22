#!/bin/bash
# Continuous evaluation entry point.
#
# Two tiers, deliberately separate because they cost very different amounts:
#
#   1. Deterministic gates (scripts/run_all_gates.py). No model calls: retrieval
#      recall / MRR / nDCG, read-plan chunk selection, and the builtin answer
#      metrics (exact match, context recall, citation integrity & precision,
#      abstention, latency, cost). Cheap enough for every commit.
#   2. LLM-assisted answer quality (deepeval). One model call per metric per
#      record, so it runs on the regression dataset only.
#
# A gate whose dataset is missing is SKIPPED loudly, never silently passed - and
# --strict turns that skip into a failure, which is what a release check uses.
set -e

# deepeval ships telemetry by default; eval runs must not phone out.
export DEEPEVAL_TELEMETRY_OPTOUT=true

DATASET="data/eval/rag_quality_regression.jsonl"

echo "=== 1/2 deterministic gates ==="
python -u scripts/run_all_gates.py

echo
echo "=== 2/2 LLM-assisted answer quality ==="
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
