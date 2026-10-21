#!/bin/bash
set -e

# deepeval ships telemetry by default; eval runs must not phone out.
export DEEPEVAL_TELEMETRY_OPTOUT=true

echo "Running Continuous Evaluation (CI/CD) with DeepEval..."

# Ensure dataset exists; generate it from negative feedback if missing.
DATASET="data/eval/rag_quality_regression.jsonl"
if [ ! -f "$DATASET" ]; then
    echo "Regression dataset not found. Generating from negative feedback..."
    python scripts/generate_regression_dataset.py
fi

# Thresholds map to metrics emitted by app/eval/deepeval_runner.py's aggregate:
# faithfulness, answer_relevancy, context_precision, context_recall.
# No silent dataset substitution: swapping in a different dataset made the gate
# meaningless ("CI Evaluation passed" on data it was not asked to evaluate).
if [ ! -f "$DATASET" ]; then
    echo "ERROR: regression dataset $DATASET could not be produced." >&2
    echo "Record negative feedback (or commit a dataset) before running the gate." >&2
    exit 1
fi

echo "Running DeepEval on regression dataset..."
python -u scripts/run_eval.py --dataset "$DATASET" --min-answer-relevancy 0.7 --min-context-recall 0.6

echo "CI Evaluation passed!"
