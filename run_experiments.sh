#!/bin/bash

# Emotion-aware LLM Scheduling - Batch Experiment Runner
# Usage: ./run_experiments.sh

set -e  # Exit on error

rm -f results/cache/job_configs.json
rm -f results/cache/responses.json

rm -rf results/llm_runs

# Configuration
OUTPUT_DIR="results/llm_runs"
SCHEDULERS=("FCFS" "SSJF-Emotion")
NUM_JOBS=50
RANDOM_SEED=42

# Parameter sweeps (uncomment/modify as needed)
SYSTEM_LOADS=(1.2)
ALPHAS=(0.5)

echo "=== Starting Batch Experiments ==="
echo "Output directory: $OUTPUT_DIR"
echo "Schedulers: ${SCHEDULERS[*]}"
echo "System loads: ${SYSTEM_LOADS[*]}"
echo ""

# Run experiments
for load in "${SYSTEM_LOADS[@]}"; do
    echo ">>> System Load: $load"

    for scheduler in "${SCHEDULERS[@]}"; do
        echo "  Running $scheduler..."

        uv run python run_simulation.py \
            --scheduler "$scheduler" \
            --num_jobs $NUM_JOBS \
            --system_load "$load" \
            --random_seed $RANDOM_SEED \
            --output_dir "$OUTPUT_DIR" \
            --verbose

        echo "  $scheduler completed."
    done

    echo ""
done

uv run python analysis/plot_emotion_results.py

echo "=== All Experiments Completed ==="
echo "Results saved to: $OUTPUT_DIR"
