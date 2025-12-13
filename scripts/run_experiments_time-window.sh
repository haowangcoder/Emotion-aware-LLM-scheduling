#!/bin/bash

# ============================================================================
# Affect-Aware LLM Scheduling - Time-Window Mode Experiments
# ============================================================================
# This script runs experiments in Time-Window mode where jobs arrive
# continuously and we count how many are completed within a fixed duration.
# Focus: throughput differences between schedulers.
#
# Schedulers:
#   - FCFS: First-Come-First-Serve (baseline)
#   - SJF: Shortest-Job-First (BERT-predicted service time)
#   - AW-SSJF: Affect-Weighted SSJF (main algorithm)
#   - Weight-Only: Pure affect-based priority (ablation baseline)
#
# Usage: ./run_experiments_time-window.sh
# ============================================================================

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================
MODE="time_window"
SIMULATION_DURATION=93  # Simulation duration in seconds
NUM_JOBS=54             # Trace size for job generation
RANDOM_SEED=42

# Schedulers to compare (updated to new naming convention)
SCHEDULERS=("FCFS" "SJF" "AW-SSJF" "Weight-Only")

# Parameter sweeps
SYSTEM_LOADS=(1.2)
# SYSTEM_LOADS=(0.8 1.0 1.2 1.5 1.8 2.0)  # Uncomment for load sweep

# AW-SSJF parameters (Depression-First strategy)
W_MAX=2.0  # Maximum affect weight

# Output directory
OUTPUT_DIR="results/llm_runs_time${SIMULATION_DURATION}_job${NUM_JOBS}_load${SYSTEM_LOADS}"
echo "OUTPUT_DIR = $OUTPUT_DIR"

# ============================================================================
# CLEANUP
# ============================================================================
echo "Cleaning up cache for ${OUTPUT_DIR}..."
rm -rf "${OUTPUT_DIR}/cache/"

# ============================================================================
# EXPERIMENT INFO
# ============================================================================
echo ""
echo "========================================================================"
echo "  Affect-Aware LLM Scheduling - Time-Window Mode"
echo "========================================================================"
echo "Mode:                $MODE"
echo "Simulation duration: ${SIMULATION_DURATION}s"
echo "Trace size:          $NUM_JOBS jobs"
echo "Schedulers:          ${SCHEDULERS[*]}"
echo "System loads:        ${SYSTEM_LOADS[*]}"
echo "Random seed:         $RANDOM_SEED"
echo "W_max:               $W_MAX"
echo "Output dir:          $OUTPUT_DIR"
echo "========================================================================"
echo ""

# ============================================================================
# RUN EXPERIMENTS
# ============================================================================
for load in "${SYSTEM_LOADS[@]}"; do
  for scheduler in "${SCHEDULERS[@]}"; do
    echo ">>> Running $scheduler at load=$load ..."

    if [[ "$scheduler" == "FCFS" ]]; then
      # First scheduler: Generate new job configs
      JOB_CONFIG_ARGS="--force_new_job_config"
    else
      # Subsequent schedulers: Reuse the same job configs for fair comparison
      JOB_CONFIG_ARGS="--use_saved_job_config"
    fi

    uv run python run_simulation.py \
      --mode "$MODE" \
      --scheduler "$scheduler" \
      --simulation_duration "$SIMULATION_DURATION" \
      --num_jobs "$NUM_JOBS" \
      --system_load "$load" \
      --random_seed "$RANDOM_SEED" \
      --w_max "$W_MAX" \
      --output_dir "$OUTPUT_DIR" \
      $JOB_CONFIG_ARGS \
      --verbose

    echo ">>> $scheduler completed"
  done
done

# ============================================================================
# GENERATE PLOTS
# ============================================================================
echo "========================================================================"
echo "  Generating Visualization Plots"
echo "========================================================================"
echo ""

uv run python analysis/plot_emotion_results.py --runs-dir "$OUTPUT_DIR"

echo ""
echo "========================================================================"
echo "  All Experiments Completed Successfully!"
echo "========================================================================"
echo "Results saved to: $OUTPUT_DIR"
echo "Plots saved to:   $OUTPUT_DIR/plots"
echo ""
echo "Key Metrics (Time-Window Mode):"
echo "  - Throughput (jobs/sec)"
echo "  - Number of completed jobs within ${SIMULATION_DURATION}s"
echo "  - Average JCT"
echo "  - Average Waiting Time"
echo ""
echo "Expected: AW-SSJF provides better fairness for depression-quadrant users"
echo "          while maintaining similar throughput to SJF."
echo "========================================================================"
