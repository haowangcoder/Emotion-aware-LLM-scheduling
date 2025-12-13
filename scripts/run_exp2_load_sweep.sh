#!/bin/bash
# Exp-2: Load Sweep (ρ Scan)
#
# Purpose: Demonstrate results across different load levels
# Parameters: ρ ∈ {0.5, 0.7, 0.9, 1.0, 1.2}
# Schedulers: SJF, AW-SSJF(k=2), AW-SSJF(k=4)
#
# Usage:
#   bash scripts/run_exp2_load_sweep.sh

set -e

# Configuration
OUTPUT_BASE="results/experiments/exp2_load_sweep"
LOAD_VALUES=(0.5 0.7 0.9 1.0 1.2)
NUM_JOBS=80
W_MAX=2.0
RANDOM_SEED=42

# Create output directory
mkdir -p "$OUTPUT_BASE"

echo "=========================================="
echo "Exp-2: Load Sweep"
echo "=========================================="
echo "Parameters:"
echo "  load values: ${LOAD_VALUES[*]}"
echo "  num_jobs: $NUM_JOBS"
echo "  w_max: $W_MAX"
echo "  output: $OUTPUT_BASE"
echo "=========================================="

# Run for each load value
for load in "${LOAD_VALUES[@]}"; do
    echo ""
    echo "=========================================="
    echo ">>> Load = $load"
    echo "=========================================="

    LOAD_DIR="${OUTPUT_BASE}/load${load}"
    mkdir -p "$LOAD_DIR"

    FIRST=true

    # SJF baseline
    echo ">>> Running SJF at load=$load"
    if [ "$FIRST" = true ]; then
        JOB_CONFIG_FLAG="--force_new_job_config"
        FIRST=false
    else
        JOB_CONFIG_FLAG="--use_saved_job_config"
    fi

    uv run python run_simulation.py \
        --scheduler SJF \
        --system_load "$load" \
        --num_jobs "$NUM_JOBS" \
        --random_seed "$RANDOM_SEED" \
        --mode fixed_jobs \
        --output_dir "$LOAD_DIR" \
        $JOB_CONFIG_FLAG

    # AW-SSJF k=2
    echo ">>> Running AW-SSJF (k=2) at load=$load"
    uv run python run_simulation.py \
        --scheduler AW-SSJF \
        --weight_exponent 2 \
        --w_max "$W_MAX" \
        --system_load "$load" \
        --num_jobs "$NUM_JOBS" \
        --random_seed "$RANDOM_SEED" \
        --mode fixed_jobs \
        --output_dir "$LOAD_DIR" \
        --use_saved_job_config

    # AW-SSJF k=4
    echo ">>> Running AW-SSJF (k=4) at load=$load"
    uv run python run_simulation.py \
        --scheduler AW-SSJF \
        --weight_exponent 4 \
        --w_max "$W_MAX" \
        --system_load "$load" \
        --num_jobs "$NUM_JOBS" \
        --random_seed "$RANDOM_SEED" \
        --mode fixed_jobs \
        --output_dir "$LOAD_DIR" \
        --use_saved_job_config

    echo ">>> Completed load=$load"

    # Clear cache for next load level (to regenerate job configs with new load)
    rm -f "${LOAD_DIR}/cache/job_configs.json" 2>/dev/null || true
done

echo ""
echo "=========================================="
echo "Load sweep simulations complete! Running analysis..."
echo "=========================================="

uv run python experiments/exp2_load_sweep.py --input_dir "$OUTPUT_BASE"

echo ""
echo "=========================================="
echo "Exp-2 complete!"
echo "=========================================="
echo "Results: $OUTPUT_BASE"
