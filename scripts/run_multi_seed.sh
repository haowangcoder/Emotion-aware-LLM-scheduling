#!/bin/bash

# ============================================================================
# Emotion-aware LLM Scheduling - Multi-Seed Experiments
# ============================================================================
# This script runs experiments with multiple random seeds to calculate
# confidence intervals for all metrics.
#
# Usage: ./run_multi_seed.sh [num_seeds]
#        ./run_multi_seed.sh 20  # Run with 20 seeds (default)
# ============================================================================

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================
NUM_SEEDS=${1:-20}  # Default to 20 seeds if not specified
BASE_OUTPUT_DIR="results/multi_seed_runs"
MODE="fixed_jobs"
NUM_JOBS=180
SYSTEM_LOAD=0.9

# Schedulers to compare
SCHEDULERS=("FCFS" "SSJF-Emotion" "SSJF-Valence" "SSJF-Combined")

# Random seeds array (20 seeds for statistical significance)
SEEDS=(42 123 456 789 1000 1234 2345 3456 4567 5678
       6789 7890 8901 9012 1357 2468 3579 4680 5791 6802)

# Use only the requested number of seeds
SEEDS=("${SEEDS[@]:0:$NUM_SEEDS}")

echo ""
echo "========================================================================"
echo "  Emotion-aware LLM Scheduling - Multi-Seed Experiments"
echo "========================================================================"
echo "Mode:           $MODE"
echo "Number of jobs: $NUM_JOBS"
echo "System load:    $SYSTEM_LOAD"
echo "Number of seeds: ${#SEEDS[@]}"
echo "Seeds:          ${SEEDS[*]}"
echo "Schedulers:     ${SCHEDULERS[*]}"
echo "Output dir:     $BASE_OUTPUT_DIR"
echo "========================================================================"
echo ""

# ============================================================================
# RUN EXPERIMENTS FOR EACH SEED
# ============================================================================
for seed in "${SEEDS[@]}"; do
    SEED_DIR="${BASE_OUTPUT_DIR}/seed_${seed}"
    echo ""
    echo "========================================================================"
    echo "  Running experiments with seed=$seed"
    echo "========================================================================"

    for scheduler in "${SCHEDULERS[@]}"; do
        echo ">>> Running $scheduler with seed=$seed..."

        if [[ "$scheduler" == "FCFS" ]]; then
            # First scheduler: generate new trace
            sed -i 's/force_new_job_config:.*/force_new_job_config: true/'  model-serving/config/default.yaml
            sed -i 's/use_saved_job_config:.*/use_saved_job_config: false/' model-serving/config/default.yaml
        else
            # Other schedulers: reuse the trace from FCFS
            sed -i 's/force_new_job_config:.*/force_new_job_config: false/' model-serving/config/default.yaml
            sed -i 's/use_saved_job_config:.*/use_saved_job_config: true/'  model-serving/config/default.yaml
        fi

        uv run python run_simulation.py \
            --mode "$MODE" \
            --scheduler "$scheduler" \
            --num_jobs "$NUM_JOBS" \
            --system_load "$SYSTEM_LOAD" \
            --random_seed "$seed" \
            --output_dir "$SEED_DIR" \
            --verbose

        echo "✓ $scheduler (seed=$seed) completed"
    done
done

# ============================================================================
# AGGREGATE RESULTS
# ============================================================================
echo ""
echo "========================================================================"
echo "  Aggregating Results Across Seeds"
echo "========================================================================"
echo ""

uv run python analysis/aggregate_results.py \
    --input-dir "$BASE_OUTPUT_DIR" \
    --output-file "${BASE_OUTPUT_DIR}/aggregated_results.json"

# ============================================================================
# COMPLETION
# ============================================================================
echo ""
echo "========================================================================"
echo "  Multi-Seed Experiments Completed!"
echo "========================================================================"
echo "Individual results: ${BASE_OUTPUT_DIR}/seed_*/"
echo "Aggregated results: ${BASE_OUTPUT_DIR}/aggregated_results.json"
echo ""
echo "Use the aggregated results to report metrics with confidence intervals."
echo "========================================================================"
