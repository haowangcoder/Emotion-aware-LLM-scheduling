#!/bin/bash

# ============================================================================
# Emotion-aware LLM Scheduling - Starvation Coefficient Sweep
# ============================================================================
# This script runs experiments with different starvation prevention
# coefficients to analyze the trade-off between efficiency and fairness.
#
# Usage: ./run_starvation_sweep.sh
# ============================================================================

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_OUTPUT_DIR="results/starvation_sweep"
MODE="fixed_jobs"
NUM_JOBS=54
SYSTEM_LOAD=0.9
RANDOM_SEED=42

# Schedulers that use starvation prevention
SCHEDULERS=("SSJF-Emotion" "SSJF-Combined")

# Starvation coefficients to sweep
# Higher values = more lenient (jobs wait longer before being rescued)
# Lower values = more aggressive prevention (fairer but less efficient)
STARVATION_COEFFICIENTS=(2 5 10 20)

echo ""
echo "========================================================================"
echo "  Emotion-aware LLM Scheduling - Starvation Coefficient Sweep"
echo "========================================================================"
echo "Mode:                   $MODE"
echo "Number of jobs:         $NUM_JOBS"
echo "System load:            $SYSTEM_LOAD"
echo "Schedulers:             ${SCHEDULERS[*]}"
echo "Starvation coefficients: ${STARVATION_COEFFICIENTS[*]}"
echo "Output dir:             $BASE_OUTPUT_DIR"
echo "========================================================================"
echo ""

# ============================================================================
# RUN BASELINE FCFS (for comparison)
# ============================================================================
echo ">>> Running baseline FCFS..."
BASELINE_DIR="${BASE_OUTPUT_DIR}/baseline"

# Generate fresh trace
sed -i 's/force_new_job_config:.*/force_new_job_config: true/'  model-serving/config/default.yaml
sed -i 's/use_saved_job_config:.*/use_saved_job_config: false/' model-serving/config/default.yaml

uv run python run_simulation.py \
    --mode "$MODE" \
    --scheduler "FCFS" \
    --num_jobs "$NUM_JOBS" \
    --system_load "$SYSTEM_LOAD" \
    --random_seed "$RANDOM_SEED" \
    --output_dir "$BASELINE_DIR" \
    --verbose

echo "✓ FCFS baseline completed"

# ============================================================================
# RUN EXPERIMENTS FOR EACH COEFFICIENT
# ============================================================================
for coeff in "${STARVATION_COEFFICIENTS[@]}"; do
    COEFF_DIR="${BASE_OUTPUT_DIR}/coeff_${coeff}"
    echo ""
    echo "========================================================================"
    echo "  Running experiments with starvation_coefficient=${coeff}"
    echo "========================================================================"

    for scheduler in "${SCHEDULERS[@]}"; do
        echo ">>> Running $scheduler with starvation_coefficient=$coeff..."

        # Reuse the trace from FCFS baseline
        sed -i 's/force_new_job_config:.*/force_new_job_config: false/' model-serving/config/default.yaml
        sed -i 's/use_saved_job_config:.*/use_saved_job_config: true/'  model-serving/config/default.yaml

        # Copy the job config from baseline to this directory
        mkdir -p "${COEFF_DIR}/cache"
        cp "${BASELINE_DIR}/cache/job_configs.json" "${COEFF_DIR}/cache/" 2>/dev/null || true
        cp "${BASELINE_DIR}/cache/responses.json" "${COEFF_DIR}/cache/" 2>/dev/null || true

        uv run python run_simulation.py \
            --mode "$MODE" \
            --scheduler "$scheduler" \
            --num_jobs "$NUM_JOBS" \
            --system_load "$SYSTEM_LOAD" \
            --random_seed "$RANDOM_SEED" \
            --starvation_coefficient "$coeff" \
            --output_dir "$COEFF_DIR" \
            --verbose

        echo "✓ $scheduler (coefficient=$coeff) completed"
    done
done

# ============================================================================
# GENERATE COMPARISON REPORT
# ============================================================================
echo ""
echo "========================================================================"
echo "  Generating Starvation Trade-off Analysis"
echo "========================================================================"

# Generate Pareto tradeoff plot using the plotting module
uv run python -c "
import sys
sys.path.insert(0, 'analysis')
from plotting import load_starvation_sweep_results, generate_starvation_sweep_plots
from plotting.utils import setup_publication_style

setup_publication_style(dpi=300)

sweep_data = load_starvation_sweep_results('results/starvation_sweep')

# Print summary table
print()
print('=' * 80)
print('Starvation Coefficient Trade-off Analysis')
print('=' * 80)

if sweep_data['baseline']:
    b = sweep_data['baseline']
    print(f\"\\nBaseline (FCFS):\")
    print(f\"  Avg Wait: {b['avg_wait']:.2f}s\")
    print(f\"  P99 Wait: {b['p99']:.2f}s\")
    print(f\"  Jain Index: {b['jain']:.4f}\")

print()
print('-' * 80)
print(f\"{'Scheduler':<15} {'Coeff':<8} {'Avg Wait':<12} {'P99 Wait':<12} {'Jain Index':<12}\")
print('-' * 80)

for sched, coeff_data in sweep_data['schedulers'].items():
    for coeff in sorted(coeff_data.keys()):
        d = coeff_data[coeff]
        print(f\"{sched:<15} {coeff:<8} {d['avg_wait']:<12.2f} {d['p99']:<12.2f} {d['jain']:<12.4f}\")

print('-' * 80)
print()
print('Analysis:')
print('- Lower coefficient = more aggressive starvation prevention = higher fairness')
print('- Higher coefficient = more lenient = better average performance, worse tail latency')
print('=' * 80)

# Generate Pareto plot
generate_starvation_sweep_plots(
    sweep_data,
    'results/starvation_sweep/plots',
    formats=['pdf', 'png']
)
print()
print('Pareto plot saved to: results/starvation_sweep/plots/')
"

# ============================================================================
# COMPLETION
# ============================================================================
echo ""
echo "========================================================================"
echo "  Starvation Sweep Experiments Completed!"
echo "========================================================================"
echo "Results saved to: $BASE_OUTPUT_DIR"
echo ""
echo "Trade-off analysis:"
echo "  - coeff=2:  Aggressive prevention, high fairness, longer avg wait"
echo "  - coeff=5:  Moderate prevention"
echo "  - coeff=10: Default setting"
echo "  - coeff=20: Lenient, better efficiency, worse tail latency"
echo "========================================================================"
