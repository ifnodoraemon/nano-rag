#!/bin/bash
set -e

echo "Running Continuous Evaluation (CI/CD) with RAGAS..."

# Ensure dataset exists, if not, generate from regression or fallback to manual
DATASET="data/eval/rag_quality_regression.jsonl"
if [ ! -f "$DATASET" ]; then
    echo "Regression dataset not found. Generating from negative feedback..."
    python scripts/generate_regression_dataset.py
fi

if [ -f "$DATASET" ]; then
    echo "Running RAGAS evaluation on regression dataset..."
    python -u scripts/run_eval.py --dataset "$DATASET" --min-answer-relevancy 0.7
else
    echo "No negative feedback available to form regression dataset. Running manual dataset..."
    python -u scripts/run_eval.py --dataset data/eval/employee_handbook_eval.jsonl --min-answer-relevancy 0.7
fi

echo "CI Evaluation passed!"
