#!/bin/bash
# Exp-4: Online Control (Adaptive k)
#
# Purpose: Demonstrate adaptive k control based on queue length
# Parameters:
#   - Online: adaptive_k with k_min=1, k_max=4
#   - Static comparisons: k=2, k=4
#   - Baseline: SJF
#
# Uses time_window mode for continuous arrivals
#
# Usage:
#   bash scripts/run_exp4_online.sh
#   # Or via SLURM:
#   sbatch slurm/run_exp4_online.slurm

set -e

# Configuration
OUTPUT_BASE="results/experiments/exp4_online"
NUM_JOBS=80
SYSTEM_LOAD=0.9
SIMULATION_DURATION=300
W_MAX=2.0
RANDOM_SEED=42

# Create output directory
mkdir -p "$OUTPUT_BASE"

echo "=========================================="
echo "Exp-4: Online Control (Adaptive k)"
echo "=========================================="
echo "Parameters:"
echo "  num_jobs: $NUM_JOBS"
echo "  system_load: $SYSTEM_LOAD"
echo "  simulation_duration: ${SIMULATION_DURATION}s"
echo "  w_max: $W_MAX"
echo "  random_seed: $RANDOM_SEED"
echo "  output: $OUTPUT_BASE"
echo "=========================================="

# 1. Online Adaptive strategy (first run generates job config)
echo ""
echo ">>> Running Online Adaptive (k=1..4)"
OUTPUT_DIR="${OUTPUT_BASE}/online"
mkdir -p "$OUTPUT_DIR"

uv run python run_simulation.py \
    --scheduler AW-SSJF \
    --adaptive_k \
    --adaptive_k_min 1 \
    --adaptive_k_max 4 \
    --adaptive_k_high_threshold 10 \
    --adaptive_k_low_threshold 3 \
    --w_max "$W_MAX" \
    --system_load "$SYSTEM_LOAD" \
    --num_jobs "$NUM_JOBS" \
    --random_seed "$RANDOM_SEED" \
    --mode time_window \
    --simulation_duration "$SIMULATION_DURATION" \
    --output_dir "$OUTPUT_DIR" \
    --force_new_job_config

echo ">>> Completed Online Adaptive"

# 2. Static k=2 comparison
echo ""
echo ">>> Running Static k=2"
OUTPUT_DIR="${OUTPUT_BASE}/static_k2"
mkdir -p "$OUTPUT_DIR"

uv run python run_simulation.py \
    --scheduler AW-SSJF \
    --weight_exponent 2 \
    --w_max "$W_MAX" \
    --system_load "$SYSTEM_LOAD" \
    --num_jobs "$NUM_JOBS" \
    --random_seed "$RANDOM_SEED" \
    --mode time_window \
    --simulation_duration "$SIMULATION_DURATION" \
    --output_dir "$OUTPUT_DIR" \
    --use_saved_job_config

echo ">>> Completed Static k=2"

# 3. Static k=4 comparison
echo ""
echo ">>> Running Static k=4"
OUTPUT_DIR="${OUTPUT_BASE}/static_k4"
mkdir -p "$OUTPUT_DIR"

uv run python run_simulation.py \
    --scheduler AW-SSJF \
    --weight_exponent 4 \
    --w_max "$W_MAX" \
    --system_load "$SYSTEM_LOAD" \
    --num_jobs "$NUM_JOBS" \
    --random_seed "$RANDOM_SEED" \
    --mode time_window \
    --simulation_duration "$SIMULATION_DURATION" \
    --output_dir "$OUTPUT_DIR" \
    --use_saved_job_config

echo ">>> Completed Static k=4"

# 4. SJF baseline
echo ""
echo ">>> Running SJF (baseline)"
OUTPUT_DIR="${OUTPUT_BASE}/sjf"
mkdir -p "$OUTPUT_DIR"

uv run python run_simulation.py \
    --scheduler SJF \
    --system_load "$SYSTEM_LOAD" \
    --num_jobs "$NUM_JOBS" \
    --random_seed "$RANDOM_SEED" \
    --mode time_window \
    --simulation_duration "$SIMULATION_DURATION" \
    --output_dir "$OUTPUT_DIR" \
    --use_saved_job_config

echo ">>> Completed SJF"

echo ""
echo "=========================================="
echo "Online control simulations complete! Running analysis..."
echo "=========================================="

uv run python experiments/exp4_online_control.py --input_dir "$OUTPUT_BASE"

echo ""
echo "=========================================="
echo "Exp-4 complete!"
echo "=========================================="
echo "Results: $OUTPUT_BASE"
