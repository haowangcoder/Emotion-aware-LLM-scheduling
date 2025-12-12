#!/bin/bash

# ============================================================================
# Affect-Aware LLM Scheduling - Fixed-Jobs Mode Experiments
# ============================================================================
# This script runs experiments in Fixed-Jobs mode where all jobs must be
# completed. Focus: latency metrics (JCT, waiting time).
#
# Schedulers:
#   - FCFS: First-Come-First-Serve (baseline)
#   - SJF: Shortest-Job-First (BERT-predicted service time)
#   - AW-SSJF: Affect-Weighted SSJF (main algorithm)
#   - Weight-Only: Pure affect-based priority (ablation)
#
# Usage: ./run_experiments_fixed.sh
# ============================================================================

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================
MODE="fixed_jobs"
NUM_JOBS=80
RANDOM_SEED=42

# Schedulers to compare (updated to new naming convention)
# FCFS = baseline, SJF = service-time only, AW-SSJF = full affect-aware
SCHEDULERS=("FCFS" "SJF" "AW-SSJF" "Weight-Only")

# Parameter sweeps
SYSTEM_LOADS=(0.9)
# SYSTEM_LOADS=(0.6 0.8 1.0 1.2 1.5)  # Uncomment for load sweep

# AW-SSJF parameters (Depression-First strategy)
W_MAX=2.0  # Maximum affect weight

# Output directory
OUTPUT_DIR="results/llm_runs_job${NUM_JOBS}_load${SYSTEM_LOADS}"
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
echo "  Affect-Aware LLM Scheduling - Fixed-Jobs Mode"
echo "========================================================================"
echo "Mode:           $MODE"
echo "Number of jobs: $NUM_JOBS"
echo "Schedulers:     ${SCHEDULERS[*]}"
echo "System loads:   ${SYSTEM_LOADS[*]}"
echo "Random seed:    $RANDOM_SEED"
echo "W_max:          $W_MAX"
echo "Output dir:     $OUTPUT_DIR"
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
      sed -i 's/force_new_job_config:.*/force_new_job_config: true/'  model-serving/config/default.yaml
      sed -i 's/use_saved_job_config:.*/use_saved_job_config: false/' model-serving/config/default.yaml
    else
      # Subsequent schedulers: Reuse the same job configs for fair comparison
      sed -i 's/force_new_job_config:.*/force_new_job_config: false/' model-serving/config/default.yaml
      sed -i 's/use_saved_job_config:.*/use_saved_job_config: true/'  model-serving/config/default.yaml
    fi

    uv run python run_simulation.py \
      --mode "$MODE" \
      --scheduler "$scheduler" \
      --num_jobs "$NUM_JOBS" \
      --system_load "$load" \
      --random_seed "$RANDOM_SEED" \
      --w_max "$W_MAX" \
      --output_dir "$OUTPUT_DIR" \
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
echo "Key Metrics (Fixed-Jobs Mode):"
echo "  - Average JCT (Job Completion Time)"
echo "  - P99 JCT"
echo "  - Average Waiting Time"
echo "  - Fairness across emotion quadrants (Russell model)"
echo ""
echo "Note: Throughput should be similar across schedulers in this mode"
echo "      (all jobs must be completed)."
echo "========================================================================"
