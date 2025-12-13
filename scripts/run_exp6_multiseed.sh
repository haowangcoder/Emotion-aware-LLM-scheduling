#!/bin/bash
# Exp-6: Multi-seed Statistical Validation
#
# Purpose: Run multiple seeds to get statistically reliable results
# Seeds: 42-51 (10 seeds)
#
# Usage:
#   bash scripts/run_exp6_multiseed.sh

set -e

OUTPUT_BASE="results/experiments/exp6_multiseed"
SEEDS=(42 43 44 45 46 47 48 49 50 51)
NUM_JOBS=80
SYSTEM_LOAD=0.9
W_MAX=2.0
K=4
SCHEDULERS=("FCFS" "SJF" "AW-SSJF" "Weight-Only")

mkdir -p "$OUTPUT_BASE"

echo "=========================================="
echo "Exp-6: Multi-seed Statistical Validation"
echo "=========================================="
echo "Parameters:"
echo "  seeds: ${SEEDS[*]}"
echo "  schedulers: ${SCHEDULERS[*]}"
echo "  num_jobs: $NUM_JOBS"
echo "  system_load: $SYSTEM_LOAD"
echo "=========================================="

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "=========================================="
    echo ">>> Seed = $seed"
    echo "=========================================="

    SEED_DIR="${OUTPUT_BASE}/seed${seed}"
    mkdir -p "$SEED_DIR"

    FIRST=true
    for sched in "${SCHEDULERS[@]}"; do
        echo ">>> Running $sched with seed=$seed"

        if [ "$FIRST" = true ]; then
            JOB_CONFIG_FLAG="--force_new_job_config"
            FIRST=false
        else
            JOB_CONFIG_FLAG="--use_saved_job_config"
        fi

        # Build command based on scheduler
        if [ "$sched" = "AW-SSJF" ]; then
            uv run python run_simulation.py \
                --scheduler "$sched" \
                --weight_exponent "$K" \
                --w_max "$W_MAX" \
                --system_load "$SYSTEM_LOAD" \
                --num_jobs "$NUM_JOBS" \
                --random_seed "$seed" \
                --mode fixed_jobs \
                --output_dir "$SEED_DIR" \
                $JOB_CONFIG_FLAG
        else
            uv run python run_simulation.py \
                --scheduler "$sched" \
                --system_load "$SYSTEM_LOAD" \
                --num_jobs "$NUM_JOBS" \
                --random_seed "$seed" \
                --mode fixed_jobs \
                --output_dir "$SEED_DIR" \
                $JOB_CONFIG_FLAG
        fi
    done

    echo ">>> Completed seed=$seed"
done

echo ""
echo "=========================================="
echo "Multi-seed simulations complete! Running analysis..."
echo "=========================================="

uv run python experiments/exp6_multi_seed.py --input_dir "$OUTPUT_BASE"

echo ""
echo "=========================================="
echo "Exp-6 complete!"
echo "=========================================="
echo "Results: $OUTPUT_BASE"
