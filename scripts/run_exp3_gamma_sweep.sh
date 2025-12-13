#!/bin/bash
# Exp-3: gamma_panic Sweep (DUAL_CHANNEL)
#
# Purpose: Demonstrate gamma_panic controls Panic channel weight
# Parameters: gamma_panic ∈ {0.0, 0.15, 0.3, 0.5}
# Fixed: k=4.0, w_max=2.0, weight_mode=dual
#
# Usage:
#   bash scripts/run_exp3_gamma_sweep.sh

set -e

OUTPUT_BASE="results/experiments/exp3_gamma_sweep"
GAMMA_VALUES=(0.0 0.15 0.3 0.5)
NUM_JOBS=80
SYSTEM_LOAD=0.9
W_MAX=2.0
K=4
RANDOM_SEED=42

mkdir -p "$OUTPUT_BASE"

echo "=========================================="
echo "Exp-3: gamma_panic Sweep"
echo "=========================================="
echo "Parameters:"
echo "  gamma_panic values: ${GAMMA_VALUES[*]}"
echo "  k: $K, w_max: $W_MAX, load: $SYSTEM_LOAD"
echo "=========================================="

FIRST=true
for gamma in "${GAMMA_VALUES[@]}"; do
    echo ""
    echo ">>> Running with gamma_panic=$gamma"
    OUTPUT_DIR="${OUTPUT_BASE}/gamma${gamma}"
    mkdir -p "$OUTPUT_DIR"

    if [ "$FIRST" = true ]; then
        JOB_CONFIG_FLAG="--force_new_job_config"
        FIRST=false
    else
        JOB_CONFIG_FLAG="--use_saved_job_config"
    fi

    uv run python run_simulation.py \
        --scheduler AW-SSJF \
        --weight_exponent "$K" \
        --w_max "$W_MAX" \
        --weight_mode dual \
        --gamma_panic "$gamma" \
        --system_load "$SYSTEM_LOAD" \
        --num_jobs "$NUM_JOBS" \
        --random_seed "$RANDOM_SEED" \
        --mode fixed_jobs \
        --output_dir "$OUTPUT_DIR" \
        $JOB_CONFIG_FLAG

    echo ">>> Completed gamma_panic=$gamma"
done

echo ""
echo "=========================================="
echo "gamma_panic sweep simulations complete! Running analysis..."
echo "=========================================="

uv run python experiments/exp3_gamma_sweep.py --input_dir "$OUTPUT_BASE"

echo ""
echo "=========================================="
echo "Exp-3 complete!"
echo "=========================================="
echo "Results: $OUTPUT_BASE"
