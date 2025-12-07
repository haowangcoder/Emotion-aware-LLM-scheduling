#!/bin/bash

# ============================================================================
# Emotion-aware LLM Scheduling - Parameter Sweep (α and β)
# ============================================================================
# This script runs experiments sweeping two key parameters:
#   - α (alpha): arousal → service time strength (L = L_0 * (1 + α * arousal))
#   - β (beta): valence weight strength (W = 1 + β * (-valence))
#
# Usage: ./run_param_sweep.sh
# ============================================================================

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_OUTPUT_DIR="results/param_sweep"
MODE="fixed_jobs"
NUM_JOBS=54
SYSTEM_LOAD=0.9
RANDOM_SEED=42

# Scheduler to test (uses both α and β)
SCHEDULER="SSJF-Combined"

# Parameter sweep values
# α: arousal impact on service time (0 = no effect, 1 = full effect)
ALPHA_VALUES=(0.0 0.25 0.5 0.75 1.0)

# β: valence weight strength (0 = no differentiation, 1 = strong differentiation)
BETA_VALUES=(0.0 0.25 0.5 0.75 1.0)

echo ""
echo "========================================================================"
echo "  Emotion-aware LLM Scheduling - Parameter Sweep (α × β)"
echo "========================================================================"
echo "Mode:           $MODE"
echo "Number of jobs: $NUM_JOBS"
echo "System load:    $SYSTEM_LOAD"
echo "Scheduler:      $SCHEDULER"
echo "α values:       ${ALPHA_VALUES[*]}"
echo "β values:       ${BETA_VALUES[*]}"
echo "Output dir:     $BASE_OUTPUT_DIR"
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
# RUN 2D PARAMETER SWEEP
# ============================================================================
TOTAL_RUNS=$((${#ALPHA_VALUES[@]} * ${#BETA_VALUES[@]}))
RUN_COUNT=0

for alpha in "${ALPHA_VALUES[@]}"; do
    for beta in "${BETA_VALUES[@]}"; do
        RUN_COUNT=$((RUN_COUNT + 1))

        # Create directory name with parameter values
        PARAM_DIR="${BASE_OUTPUT_DIR}/alpha_${alpha}_beta_${beta}"

        echo ""
        echo "========================================================================"
        echo "  [$RUN_COUNT/$TOTAL_RUNS] Running α=${alpha}, β=${beta}"
        echo "========================================================================"

        # Reuse the trace from FCFS baseline
        sed -i 's/force_new_job_config:.*/force_new_job_config: false/' model-serving/config/default.yaml
        sed -i 's/use_saved_job_config:.*/use_saved_job_config: true/'  model-serving/config/default.yaml

        # Copy the job config from baseline
        mkdir -p "${PARAM_DIR}/cache"
        cp "${BASELINE_DIR}/cache/job_configs.json" "${PARAM_DIR}/cache/" 2>/dev/null || true
        cp "${BASELINE_DIR}/cache/responses.json" "${PARAM_DIR}/cache/" 2>/dev/null || true

        uv run python run_simulation.py \
            --mode "$MODE" \
            --scheduler "$SCHEDULER" \
            --num_jobs "$NUM_JOBS" \
            --system_load "$SYSTEM_LOAD" \
            --random_seed "$RANDOM_SEED" \
            --alpha "$alpha" \
            --beta "$beta" \
            --output_dir "$PARAM_DIR" \
            --verbose

        echo "✓ α=${alpha}, β=${beta} completed"
    done
done

# ============================================================================
# GENERATE ANALYSIS AND PLOTS
# ============================================================================
echo ""
echo "========================================================================"
echo "  Generating Parameter Sweep Analysis"
echo "========================================================================"

uv run python -c "
import sys
sys.path.insert(0, 'analysis')
from plotting import load_param_sweep_results, generate_param_sweep_plots
from plotting.utils import setup_publication_style

setup_publication_style(dpi=300)

sweep_data = load_param_sweep_results('results/param_sweep')

# Print summary
print()
print('=' * 80)
print('Parameter Sweep Results (α × β)')
print('=' * 80)

if sweep_data['baseline']:
    b = sweep_data['baseline']
    print(f\"\\nBaseline (FCFS):\")
    print(f\"  Avg Wait: {b['avg_wait']:.2f}s, P99: {b['p99']:.2f}s, Jain: {b['jain']:.4f}\")

print()
print('α (arousal→length) × β (valence weight) Grid:')
print('-' * 80)

# Print as table
alphas = sweep_data['alphas']
betas = sweep_data['betas']

print(f\"{'':>8}\", end='')
for beta in betas:
    print(f\"  β={beta:<6}\", end='')
print()

for alpha in alphas:
    print(f\"α={alpha:<5}\", end='')
    for beta in betas:
        key = (alpha, beta)
        if key in sweep_data['grid']:
            d = sweep_data['grid'][key]
            print(f\"  {d['avg_wait']:>6.1f}s\", end='')
        else:
            print(f\"  {'N/A':>7}\", end='')
    print()

print('-' * 80)
print()
print('Legend: Values show avg_waiting_time (s)')
print('=' * 80)

# Generate plots
generate_param_sweep_plots(
    sweep_data,
    'results/param_sweep/plots',
    formats=['pdf', 'png']
)
print()
print('Plots saved to: results/param_sweep/plots/')
"

# ============================================================================
# COMPLETION
# ============================================================================
echo ""
echo "========================================================================"
echo "  Parameter Sweep Experiments Completed!"
echo "========================================================================"
echo "Results saved to: $BASE_OUTPUT_DIR"
echo ""
echo "Parameter effects:"
echo "  - α (alpha): Controls arousal → service time mapping strength"
echo "    α=0: No emotion effect on length"
echo "    α=1: Full arousal-based length variation"
echo ""
echo "  - β (beta): Controls valence-based priority differentiation"
echo "    β=0: Equal priority for all valence classes"
echo "    β=1: Strong preference for negative-valence jobs"
echo "========================================================================"
