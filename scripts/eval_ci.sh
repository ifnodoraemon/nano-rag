#!/bin/bash
set -e

# deepeval ships telemetry by default; eval runs must not phone out.
export DEEPEVAL_TELEMETRY_OPTOUT=true

echo "Running Continuous Evaluation (CI/CD) with DeepEval..."

# Ensure dataset exists, if not, generate from regression or fallback to manual
DATASET="data/eval/rag_quality_regression.jsonl"
if [ ! -f "$DATASET" ]; then
    echo "Regression dataset not found. Generating from negative feedback..."
    python scripts/generate_regression_dataset.py
fi

# Thresholds map to metrics emitted by app/eval/deepeval_runner.py's aggregate:
# faithfulness, answer_relevancy, context_precision, context_recall.
if [ -f "$DATASET" ]; then
    echo "Running DeepEval on regression dataset..."
    python -u scripts/run_eval.py --dataset "$DATASET" --min-answer-relevancy 0.7 --min-context-recall 0.6
else
    echo "No negative feedback available to form regression dataset. Running manual dataset..."
    python -u scripts/run_eval.py --dataset data/eval/employee_handbook_eval.jsonl --min-answer-relevancy 0.7 --min-context-recall 0.6
fi

echo "CI Evaluation passed!"
