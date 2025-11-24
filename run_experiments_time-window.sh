#!/bin/bash

# ============================================================================
# Emotion-aware LLM Scheduling - Time-Window Mode Experiments
# ============================================================================
# This script runs experiments in Time-Window mode where jobs arrive 
# continuously and we count how many are completed within a fixed duration.
# Focus: throughput differences between schedulers.
#
# Usage: ./run_experiments_time-window.sh
# ============================================================================

set -e  # Exit on error

# ============================================================================
# CLEANUP
# ============================================================================
echo "Cleaning up cache and previous results..."
rm -rf results/cache/
rm -rf results/llm_runs_time*/cache/


# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_DIR="results/llm_runs"
MODE="time_window"
SIMULATION_DURATION=80  # 5 minutes (300 seconds)
NUM_JOBS=50             # Trace size (actual will be 200 for sufficient coverage)
RANDOM_SEED=42

# Schedulers to compare
SCHEDULERS=("FCFS" "SSJF-Emotion")

# Parameter sweeps
SYSTEM_LOADS=(1.2)
# SYSTEM_LOADS=(0.8 1.0 1.2 1.5 1.8 2.0)  # Uncomment for load sweep

# Update OUTPUT_DIR
OUTPUT_DIR="${OUTPUT_DIR}_time${SIMULATION_DURATION}_job${NUM_JOBS}_load${SYSTEM_LOADS}"
echo "OUTPUT_DIR = $OUTPUT_DIR"

# ============================================================================
# EXPERIMENT INFO
# ============================================================================
echo ""
echo "========================================================================"
echo "  Emotion-aware LLM Scheduling - Time-Window Mode"
echo "========================================================================"
echo "Mode:                $MODE"
echo "Simulation duration: ${SIMULATION_DURATION}s"
echo "Trace size:          $NUM_JOBS jobs (will generate 2x)"
echo "Schedulers:          ${SCHEDULERS[*]}"
echo "System loads:        ${SYSTEM_LOADS[*]}"
echo "Random seed:         $RANDOM_SEED"
echo "Output dir:          $OUTPUT_DIR"
echo "========================================================================"
echo ""

# ============================================================================
# RUN EXPERIMENTS
# ============================================================================
for load in "${SYSTEM_LOADS[@]}"; do
  for scheduler in "${SCHEDULERS[@]}"; do
    echo ">>> Running $scheduler ..."

    if [[ "$scheduler" == "FCFS" ]]; then
      # FCFS：生成 time-window 用的 trace
      sed -i 's/force_new_job_config:.*/force_new_job_config: false/'  model-serving/config/default.yaml
      sed -i 's/use_saved_job_config:.*/use_saved_job_config: false/' model-serving/config/default.yaml
    else
      # SSJF：复用 FCFS 刚刚生成的那条 trace
      sed -i 's/force_new_job_config:.*/force_new_job_config: false/' model-serving/config/default.yaml
      sed -i 's/use_saved_job_config:.*/use_saved_job_config: false/'  model-serving/config/default.yaml
    fi

    uv run python run_simulation.py \
      --mode "$MODE" \
      --scheduler "$scheduler" \
      --simulation_duration "$SIMULATION_DURATION" \
      --num_jobs "$NUM_JOBS" \
      --system_load "$load" \
      --random_seed "$RANDOM_SEED" \
      --output_dir "$OUTPUT_DIR" \
      --verbose

    echo "✓ $scheduler completed"
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
echo "  - Throughput (jobs/sec) - SHOULD DIFFER significantly!"
echo "  - Number of completed jobs within ${SIMULATION_DURATION}s"
echo "  - Average JCT"
echo "  - Average Waiting Time"
echo ""
echo "Expected: SSJF-Emotion completes MORE jobs than FCFS in same time."
echo "========================================================================"
