#!/bin/bash

# Emotion-aware LLM Scheduling - Batch Experiment Runner
# Usage: ./run_experiments.sh

set -e  # Exit on error

rm -f results/cache/job_configs.json
rm -f results/cache/responses.json

rm -rf results/llm_runs

# Configuration
OUTPUT_DIR="results"
SCHEDULERS=("FCFS" "SSJF-Emotion")
NUM_JOBS=50
RANDOM_SEED=42

# Parameter sweeps (uncomment/modify as needed)
SYSTEM_LOADS=(1 1.2)
ALPHAS=(1.5 2 4)

echo "=== Starting Batch Experiments ==="
echo "Output directory: $OUTPUT_DIR"
echo "Schedulers: ${SCHEDULERS[*]}"
echo "System loads: ${SYSTEM_LOADS[*]}"
echo ""

# Run experiments
for load in "${SYSTEM_LOADS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
        echo ">>> System Load: $load, Alpha: $alpha"

        # Create unique output directory for this parameter combination
        RUN_DIR="${OUTPUT_DIR}/load${load}_alpha${alpha}"
        CACHE_DIR="${RUN_DIR}/cache"
        mkdir -p "$RUN_DIR" "$CACHE_DIR"

        # Set cache directory for this parameter combination
        export LLM_CACHE_CACHE_DIR="$CACHE_DIR"

        for scheduler in "${SCHEDULERS[@]}"; do
            echo "  Running $scheduler..."

            uv run python run_simulation.py \
                --scheduler "$scheduler" \
                --num_jobs $NUM_JOBS \
                --system_load "$load" \
                --alpha "$alpha" \
                --random_seed $RANDOM_SEED \
                --output_dir "$RUN_DIR" \
                --verbose

            echo "  $scheduler completed."
        done

        # Generate plots for this parameter combination
        echo "  Generating plots for load=$load, alpha=$alpha..."
        uv run python analysis/plot_emotion_results.py --runs-dir "$RUN_DIR"

        echo ""
    done
done

echo "=== All Experiments Completed ==="
echo "Results saved to: $OUTPUT_DIR"
