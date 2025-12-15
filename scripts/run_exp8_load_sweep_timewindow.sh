#!/bin/bash
# Exp-8: Load Sweep (Time Window Mode with Queue Drain)
#
# Purpose: Same as Exp-2, but using time_window mode
# Parameters: rho in {0.5, 0.7, 0.9, 1.0, 1.2}
# Schedulers: SJF, AW-SSJF(k=2), AW-SSJF(k=4)
# Mode: time_window (fixed duration for arrivals, then drain queue)
#
# IMPORTANT: This experiment uses a UNIFIED job trace across all load levels.
# The same jobs (emotion, service_time, etc.) are used, only arrival_rate differs.
#
# NOTE: The time_window mode now includes a "drain" phase that completes all
# jobs that arrived within the window. This ensures correct tail latency measurement.
#
# Usage:
#   bash scripts/run_exp8_load_sweep_timewindow.sh

set -e

# Configuration
OUTPUT_BASE="results/experiments/exp8_load_sweep_timewindow"
LOAD_VALUES=(0.5 0.7 0.9 1.0 1.2)
NUM_JOBS=80
SIMULATION_DURATION=100  # Duration for accepting arrivals
W_MAX=2.0
RANDOM_SEED=42

# Create output directory
mkdir -p "$OUTPUT_BASE"

echo "=========================================="
echo "Exp-8: Load Sweep (Time Window Mode)"
echo "=========================================="
echo "Parameters:"
echo "  load values: ${LOAD_VALUES[*]}"
echo "  num_jobs (trace size): $NUM_JOBS"
echo "  simulation_duration: ${SIMULATION_DURATION}s"
echo "  mode: time_window (with queue drain)"
echo "  w_max: $W_MAX"
echo "  output: $OUTPUT_BASE"
echo "=========================================="

# Step 1: Generate unified job trace for all load levels
echo ""
echo "=========================================="
echo "Step 1: Generating unified job trace..."
echo "=========================================="
uv run python scripts/generate_unified_trace.py \
    --output_dir "$OUTPUT_BASE" \
    --num_jobs 160 \
    --random_seed "$RANDOM_SEED" \
    --load_values "$(IFS=,; echo "${LOAD_VALUES[*]}")"

# Step 2: Run experiments for each load value
echo ""
echo "=========================================="
echo "Step 2: Running experiments..."
echo "=========================================="

for load in "${LOAD_VALUES[@]}"; do
    echo ""
    echo "=========================================="
    echo ">>> Load = $load"
    echo "=========================================="

    LOAD_DIR="${OUTPUT_BASE}/load${load}"
    # Note: DO NOT clear cache - we want to reuse the unified trace!

    # All schedulers use the same pre-generated trace
    # SJF baseline
    echo ">>> Running SJF at load=$load"
    uv run python run_simulation.py \
        --scheduler SJF \
        --system_load "$load" \
        --num_jobs "$NUM_JOBS" \
        --simulation_duration "$SIMULATION_DURATION" \
        --random_seed "$RANDOM_SEED" \
        --mode time_window \
        --output_dir "$LOAD_DIR" \
        --use_saved_job_config

    # AW-SSJF k=2
    echo ">>> Running AW-SSJF (k=2) at load=$load"
    uv run python run_simulation.py \
        --scheduler AW-SSJF \
        --weight_exponent 2 \
        --w_max "$W_MAX" \
        --system_load "$load" \
        --num_jobs "$NUM_JOBS" \
        --simulation_duration "$SIMULATION_DURATION" \
        --random_seed "$RANDOM_SEED" \
        --mode time_window \
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
        --simulation_duration "$SIMULATION_DURATION" \
        --random_seed "$RANDOM_SEED" \
        --mode time_window \
        --output_dir "$LOAD_DIR" \
        --use_saved_job_config

    echo ">>> Completed load=$load"
done

echo ""
echo "=========================================="
echo "Load sweep simulations complete! Running analysis..."
echo "=========================================="

uv run python experiments/exp8_load_sweep_timewindow.py --input_dir "$OUTPUT_BASE"

echo ""
echo "=========================================="
echo "Exp-8 complete!"
echo "=========================================="
echo "Results: $OUTPUT_BASE"
echo "Key difference from Exp-2: Uses time_window mode (fixed duration) instead of fixed_jobs mode"
