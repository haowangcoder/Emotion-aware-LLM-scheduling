#!/bin/bash
# Exp-1: k (weight_exponent) Sweep
#
# Purpose: Demonstrate controllable trade-off via k parameter
# Parameters: k ∈ {1, 2, 3, 4}
# Fixed: w_max=2.0, system_load=0.9, num_jobs=80
#
# Usage:
#   bash scripts/run_exp1_k_sweep.sh
#   # Or via SLURM:
#   sbatch slurm/run_exp1_k_sweep.slurm

set -e

# Configuration
OUTPUT_BASE="results/experiments/exp1_k_sweep"
K_VALUES=(1 2 3 4)
NUM_JOBS=80
SYSTEM_LOAD=0.9
W_MAX=2.0
RANDOM_SEED=42
SCHEDULER="AW-SSJF"

# Create output directory
mkdir -p "$OUTPUT_BASE"

echo "=========================================="
echo "Exp-1: k (weight_exponent) Sweep"
echo "=========================================="
echo "Parameters:"
echo "  k values: ${K_VALUES[*]}"
echo "  num_jobs: $NUM_JOBS"
echo "  system_load: $SYSTEM_LOAD"
echo "  w_max: $W_MAX"
echo "  random_seed: $RANDOM_SEED"
echo "  output: $OUTPUT_BASE"
echo "=========================================="

# Run for each k value
FIRST=true
for k in "${K_VALUES[@]}"; do
    echo ""
    echo ">>> Running with k=$k"
    OUTPUT_DIR="${OUTPUT_BASE}/k${k}"
    mkdir -p "$OUTPUT_DIR"

    # First run generates job config, subsequent runs reuse it
    if [ "$FIRST" = true ]; then
        JOB_CONFIG_FLAG="--force_new_job_config"
        FIRST=false
    else
        JOB_CONFIG_FLAG="--use_saved_job_config"
    fi

    uv run python run_simulation.py \
        --scheduler "$SCHEDULER" \
        --weight_exponent "$k" \
        --w_max "$W_MAX" \
        --system_load "$SYSTEM_LOAD" \
        --num_jobs "$NUM_JOBS" \
        --random_seed "$RANDOM_SEED" \
        --mode fixed_jobs \
        --output_dir "$OUTPUT_DIR" \
        $JOB_CONFIG_FLAG

    echo ">>> Completed k=$k"
done

echo ""
echo "=========================================="
echo "k-sweep simulations complete! Running analysis..."
echo "=========================================="

uv run python experiments/exp1_k_sweep.py --input_dir "$OUTPUT_BASE"

echo ""
echo "=========================================="
echo "Exp-1 complete!"
echo "=========================================="
echo "Results: $OUTPUT_BASE"
