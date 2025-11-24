#!/bin/bash

# ============================================================================
# Emotion-aware LLM Scheduling - Fixed-Jobs Mode Experiments
# ============================================================================
# This script runs experiments in Fixed-Jobs mode where all jobs must be
# completed. Focus: latency metrics (JCT, waiting time).
#
# Usage: ./run_experiments_fixed.sh
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
MODE="fixed_jobs"
NUM_JOBS=50
RANDOM_SEED=42

# Schedulers to compare
SCHEDULERS=("FCFS" "SSJF-Emotion")

# Parameter sweeps
SYSTEM_LOADS=(1.2)
# ALPHAS=(1.5)
# SYSTEM_LOADS=(0.8 1.0 1.2 1.5)  # Uncomment for load sweep

# Update OUTPUT_DIR
OUTPUT_DIR="${OUTPUT_DIR}_job${NUM_JOBS}_load${SYSTEM_LOADS}"
echo "OUTPUT_DIR = $OUTPUT_DIR"

# ============================================================================
# EXPERIMENT INFO
# ============================================================================
echo ""
echo "========================================================================"
echo "  Emotion-aware LLM Scheduling - Fixed-Jobs Mode"
echo "========================================================================"
echo "Mode:           $MODE"
echo "Number of jobs: $NUM_JOBS"
echo "Schedulers:     ${SCHEDULERS[*]}"
echo "System loads:   ${SYSTEM_LOADS[*]}"
echo "Random seed:    $RANDOM_SEED"
echo "Output dir:     $OUTPUT_DIR"
echo "========================================================================"
echo ""

# ============================================================================
# RUN EXPERIMENTS
# ============================================================================
for load in "${SYSTEM_LOADS[@]}"; do
  for scheduler in "${SCHEDULERS[@]}"; do
    echo ">>> Running $scheduler ..."

    if [[ "$scheduler" == "FCFS" ]]; then
      # 第一个：强制生成一批新的 trace，并写入 job_configs.json
      sed -i 's/force_new_job_config:.*/force_new_job_config: true/'  model-serving/config/default.yaml
      sed -i 's/use_saved_job_config:.*/use_saved_job_config: false/' model-serving/config/default.yaml
    else
      # 第二个：严格复用刚刚生成的 trace（绝不再采样）
      sed -i 's/force_new_job_config:.*/force_new_job_config: false/' model-serving/config/default.yaml
      sed -i 's/use_saved_job_config:.*/use_saved_job_config: true/'  model-serving/config/default.yaml
    fi

    uv run python run_simulation.py \
      --mode "$MODE" \
      --scheduler "$scheduler" \
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
echo "Key Metrics (Fixed-Jobs Mode):"
echo "  - Average JCT (Job Completion Time)"
echo "  - P99 JCT"
echo "  - Average Waiting Time"
echo "  - Fairness across emotion classes"
echo ""
echo "Note: Throughput should be similar across schedulers in this mode"
echo "      (all jobs must be completed)."
echo "========================================================================"
