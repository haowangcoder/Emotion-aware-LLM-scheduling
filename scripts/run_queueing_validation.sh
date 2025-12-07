#!/bin/bash
# =============================================================================
# M/G/1 Queueing Theory Validation Experiments
# =============================================================================
#
# This script runs load sweep experiments to validate M/G/1 queueing theory
# predictions against simulation results.
#
# Usage:
#   ./scripts/run_queueing_validation.sh
#
# Output:
#   results/queueing_validation/
#   ├── load_0.5/
#   │   ├── FCFS_*_summary.json
#   │   ├── FCFS_*_jobs.csv
#   │   ├── SSJF-Emotion_*_summary.json
#   │   └── SSJF-Emotion_*_jobs.csv
#   ├── load_0.6/
#   │   └── ...
#   └── validation_results/
#       ├── theory_predictions.csv
#       ├── validation_report.md
#       └── plots/
# =============================================================================

set -eo pipefail  # Exit on error and propagate failures through pipelines

# =============================================================================
# Configuration
# =============================================================================

BASE_DIR="results/queueing_validation"
SCHEDULERS=("FCFS" "SSJF-Emotion")

# Experiment parameters (user confirmed: 200 jobs, single seed)
NUM_JOBS=200
RANDOM_SEED=42

# System parameters
ALPHA=0.8
BASE_SERVICE_TIME=2.0

# Load levels to test
LOADS=(0.5 0.6 0.7 0.8 0.9)

# =============================================================================
# Functions
# =============================================================================

run_simulation() {
    local scheduler=$1
    local load=$2
    local output_dir=$3

    echo "  Running $scheduler at load=$load..."

    uv run python run_simulation.py \
        --scheduler "$scheduler" \
        --num_jobs $NUM_JOBS \
        --system_load "$load" \
        --base_service_time $BASE_SERVICE_TIME \
        --alpha $ALPHA \
        --random_seed $RANDOM_SEED \
        --output_dir "$output_dir" \
        --mode fixed_jobs \
        2>&1 | tail -5
}

# =============================================================================
# Main
# =============================================================================

echo "=========================================="
echo "M/G/1 Queueing Theory Validation"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Jobs per experiment: $NUM_JOBS"
echo "  Random seed: $RANDOM_SEED"
echo "  Alpha (α): $ALPHA"
echo "  Base service time (L₀): $BASE_SERVICE_TIME s"
echo "  Load levels: ${LOADS[*]}"
echo ""

# Create base directory
mkdir -p "$BASE_DIR"

# =============================================================================
# Phase 1: Run Simulations
# =============================================================================

echo "Phase 1: Running Simulations"
echo "----------------------------"

for LOAD in "${LOADS[@]}"; do
    LOAD_DIR="$BASE_DIR/load_$LOAD"
    mkdir -p "$LOAD_DIR"

    echo ""
    echo "Load ρ = $LOAD"
    echo "--------------"

    for SCHEDULER in "${SCHEDULERS[@]}"; do
        run_simulation "$SCHEDULER" "$LOAD" "$LOAD_DIR"
    done
done

echo ""
echo "Simulations complete!"
echo ""

# =============================================================================
# Phase 2: Run Validation Analysis
# =============================================================================

echo "Phase 2: Running Validation Analysis"
echo "------------------------------------"

VALIDATION_DIR="$BASE_DIR/validation_results"
mkdir -p "$VALIDATION_DIR"
mkdir -p "$VALIDATION_DIR/plots"

# Build comma-separated load levels for Python
LOADS_STR=$(IFS=,; echo "${LOADS[*]}")

# Run Python validation script
uv run python -c "
import sys
sys.path.insert(0, '.')

from analysis.queueing_theory.load_sweep import predict_both_schedulers
from analysis.queueing_theory.validation import validate_load_sweep, generate_validation_report
from analysis.queueing_theory.plotting import generate_all_validation_plots
import json

# Parameters
base_service_time = $BASE_SERVICE_TIME
alpha = $ALPHA
load_levels = [$LOADS_STR]

print('Generating theory predictions...')
theory_df = predict_both_schedulers(
    base_service_time=base_service_time,
    alpha=alpha,
    load_levels=load_levels
)
theory_df.to_csv('$VALIDATION_DIR/theory_predictions.csv', index=False)
print(f'  Saved to $VALIDATION_DIR/theory_predictions.csv')

print('Running validation...')
validation_results = validate_load_sweep(
    results_dir='$BASE_DIR',
    load_levels=load_levels,
    schedulers=['FCFS', 'SSJF-Emotion'],
    apply_transient_removal=True
)

# Save validation results as JSON (convert numpy types)
import numpy as np
def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj

with open('$VALIDATION_DIR/validation_results.json', 'w') as f:
    json.dump(convert_numpy(validation_results), f, indent=2, default=str)
print(f'  Saved to $VALIDATION_DIR/validation_results.json')

print('Generating validation report...')
report = generate_validation_report(
    validation_results,
    output_path='$VALIDATION_DIR/validation_report.md'
)
print(f'  Saved to $VALIDATION_DIR/validation_report.md')

print('Generating plots...')
generate_all_validation_plots(
    validation_results,
    output_dir='$VALIDATION_DIR/plots'
)

print('')
print('Validation complete!')
"

echo ""
echo "=========================================="
echo "Results saved to: $BASE_DIR"
echo "=========================================="
echo ""
echo "Key outputs:"
echo "  - Theory predictions: $VALIDATION_DIR/theory_predictions.csv"
echo "  - Validation results: $VALIDATION_DIR/validation_results.json"
echo "  - Validation report:  $VALIDATION_DIR/validation_report.md"
echo "  - Plots:              $VALIDATION_DIR/plots/"
echo ""
