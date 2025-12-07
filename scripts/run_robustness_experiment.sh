#!/bin/bash

# ============================================================================
# Emotion-aware LLM Scheduling - Workload Robustness Experiment
# ============================================================================
# This experiment tests scheduler robustness across different workload types:
#
# 2x2 Design:
#   - Emotion Distribution: Uniform (stratified) vs Real (EmpatheticDialogues)
#   - Arrival Pattern: Poisson (smooth) vs Bursty (ON/OFF)
#
# Key Questions:
#   1. Does the scheduler advantage hold with realistic emotion skew?
#   2. How do bursty arrivals affect tail latency (P99)?
#
# Usage: ./run_robustness_experiment.sh
# ============================================================================

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_OUTPUT_DIR="results/robustness_experiment"
MODE="fixed_jobs"
NUM_JOBS=54
SYSTEM_LOAD=0.9
RANDOM_SEED=42

# Bursty arrival parameters
BURST_FACTOR=5.0      # 5x rate during ON state
ON_DURATION=10.0      # 10s mean ON period
OFF_DURATION=30.0     # 30s mean OFF period

# Schedulers to compare
SCHEDULERS=("FCFS" "SSJF-Emotion" "SSJF-Combined")

# Experiment conditions
declare -A CONDITIONS
CONDITIONS["uniform_poisson"]="Uniform distribution + Poisson arrivals"
CONDITIONS["real_poisson"]="Real distribution + Poisson arrivals"
CONDITIONS["uniform_bursty"]="Uniform distribution + Bursty arrivals"
CONDITIONS["real_bursty"]="Real distribution + Bursty arrivals"

echo ""
echo "========================================================================"
echo "  Emotion-aware LLM Scheduling - Workload Robustness Experiment"
echo "========================================================================"
echo "Mode:           $MODE"
echo "Number of jobs: $NUM_JOBS"
echo "System load:    $SYSTEM_LOAD"
echo "Schedulers:     ${SCHEDULERS[*]}"
echo "Output dir:     $BASE_OUTPUT_DIR"
echo ""
echo "Conditions:"
for cond in "${!CONDITIONS[@]}"; do
    echo "  - $cond: ${CONDITIONS[$cond]}"
done
echo "========================================================================"
echo ""

# Clean up any existing results
rm -rf "$BASE_OUTPUT_DIR"
mkdir -p "$BASE_OUTPUT_DIR"

# ============================================================================
# HELPER FUNCTION: Generate trace with specific configuration
# ============================================================================
generate_trace() {
    local output_dir=$1
    local use_real_dist=$2      # "true" or "false"
    local arrival_pattern=$3    # "poisson" or "bursty"

    mkdir -p "${output_dir}/cache"

    uv run python << EOF
import sys
import json
sys.path.insert(0, 'model-serving')

from core.emotion import EmotionConfig, EMPATHETIC_DIALOGUES_PROBS
from workload.service_time_mapper import ServiceTimeConfig
from workload.task_generator import generate_job_trace

# Configure emotion distribution
use_real = ${use_real_dist}
if use_real:
    emotion_config = EmotionConfig(emotion_probs=EMPATHETIC_DIALOGUES_PROBS)
    print("Using REAL EmpatheticDialogues distribution")
else:
    emotion_config = EmotionConfig()  # Default uniform
    print("Using UNIFORM emotion distribution")

service_time_config = ServiceTimeConfig()

# Generate trace
trace = generate_job_trace(
    num_jobs=${NUM_JOBS},
    arrival_rate=${SYSTEM_LOAD} / service_time_config.base_service_time,
    emotion_config=emotion_config,
    service_time_config=service_time_config,
    enable_emotion=True,
    random_seed=${RANDOM_SEED},
    use_stratified_sampling=not use_real,  # Only stratify if uniform
    arrival_pattern='${arrival_pattern}',
    burst_factor=${BURST_FACTOR},
    on_duration_mean=${ON_DURATION},
    off_duration_mean=${OFF_DURATION},
)

# Save trace in expected format
output = {
    "metadata": {
        "num_jobs": ${NUM_JOBS},
        "random_seed": ${RANDOM_SEED},
        "emotion_distribution": "real" if use_real else "uniform",
        "arrival_pattern": "${arrival_pattern}"
    },
    "jobs": trace
}

with open('${output_dir}/cache/job_configs.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"Generated {len(trace)} jobs -> ${output_dir}/cache/job_configs.json")

# Print arrival time statistics
import numpy as np
arrival_times = [j['arrival_time'] for j in trace]
intervals = np.diff(arrival_times)
print(f"  Arrival intervals: mean={np.mean(intervals):.2f}s, std={np.std(intervals):.2f}s")
print(f"  Arrival burstiness (CoV): {np.std(intervals)/np.mean(intervals):.2f}")
EOF
}

# ============================================================================
# HELPER FUNCTION: Run all schedulers on a trace
# ============================================================================
run_schedulers() {
    local condition_dir=$1

    # Configure to use saved trace
    sed -i 's/force_new_job_config:.*/force_new_job_config: false/' model-serving/config/default.yaml
    sed -i 's/use_saved_job_config:.*/use_saved_job_config: true/'  model-serving/config/default.yaml

    for scheduler in "${SCHEDULERS[@]}"; do
        echo "    Running $scheduler..."

        uv run python run_simulation.py \
            --mode "$MODE" \
            --scheduler "$scheduler" \
            --num_jobs "$NUM_JOBS" \
            --system_load "$SYSTEM_LOAD" \
            --random_seed "$RANDOM_SEED" \
            --output_dir "$condition_dir" \
            --verbose 2>&1 | tail -5

        echo "    Done: $scheduler"
    done
}

# ============================================================================
# CONDITION 1: Uniform + Poisson (Baseline)
# ============================================================================
echo ""
echo "========================================================================"
echo "  Condition 1/4: Uniform Distribution + Poisson Arrivals"
echo "========================================================================"
COND_DIR="${BASE_OUTPUT_DIR}/uniform_poisson"
generate_trace "$COND_DIR" "False" "poisson"
run_schedulers "$COND_DIR"

# ============================================================================
# CONDITION 2: Real + Poisson
# ============================================================================
echo ""
echo "========================================================================"
echo "  Condition 2/4: Real Distribution + Poisson Arrivals"
echo "========================================================================"
COND_DIR="${BASE_OUTPUT_DIR}/real_poisson"
generate_trace "$COND_DIR" "True" "poisson"
run_schedulers "$COND_DIR"

# ============================================================================
# CONDITION 3: Uniform + Bursty
# ============================================================================
echo ""
echo "========================================================================"
echo "  Condition 3/4: Uniform Distribution + Bursty Arrivals"
echo "========================================================================"
COND_DIR="${BASE_OUTPUT_DIR}/uniform_bursty"
generate_trace "$COND_DIR" "False" "bursty"
run_schedulers "$COND_DIR"

# ============================================================================
# CONDITION 4: Real + Bursty
# ============================================================================
echo ""
echo "========================================================================"
echo "  Condition 4/4: Real Distribution + Bursty Arrivals"
echo "========================================================================"
COND_DIR="${BASE_OUTPUT_DIR}/real_bursty"
generate_trace "$COND_DIR" "True" "bursty"
run_schedulers "$COND_DIR"

# ============================================================================
# PHASE 5: Generate Analysis and Plots
# ============================================================================
echo ""
echo "========================================================================"
echo "  Generating Robustness Analysis"
echo "========================================================================"

uv run python -c "
import sys
import json
from pathlib import Path
sys.path.insert(0, 'analysis')
from plotting import load_robustness_experiment_results, generate_robustness_experiment_plots
from plotting.utils import setup_publication_style

setup_publication_style(dpi=300)

# Load results
results = load_robustness_experiment_results('results/robustness_experiment')

# Print comparison table
print()
print('=' * 90)
print('Robustness Experiment Results: 2x2 Design')
print('=' * 90)
print()
print('Emotion Distribution: Uniform (stratified 9-class) vs Real (EmpatheticDialogues)')
print('Arrival Pattern: Poisson (memoryless) vs Bursty (ON/OFF model)')
print()
print('-' * 90)
print(f\"{'Condition':<20} {'Scheduler':<15} {'Avg Wait':<12} {'P99 Wait':<12} {'Jain':<10}\")
print('-' * 90)

for condition in ['uniform_poisson', 'real_poisson', 'uniform_bursty', 'real_bursty']:
    if condition in results:
        for sched, data in results[condition].items():
            print(f\"{condition:<20} {sched:<15} {data['avg_wait']:>10.2f}s {data['p99']:>10.2f}s {data['jain']:>9.4f}\")
        print()

print('-' * 90)
print()

# Key insights
print('KEY INSIGHTS:')
print('  1. Real distribution has more negative emotions -> potentially higher arousal variance')
print('  2. Bursty arrivals create queue buildups -> higher tail latency')
print('  3. SSJF-* should maintain advantage across conditions (validates robustness)')
print('=' * 90)

# Generate plots
generate_robustness_experiment_plots(
    results,
    'results/robustness_experiment/plots',
    formats=['pdf', 'png']
)
print()
print('Plots saved to: results/robustness_experiment/plots/')
"

# ============================================================================
# COMPLETION
# ============================================================================
echo ""
echo "========================================================================"
echo "  Robustness Experiment Completed!"
echo "========================================================================"
echo "Results saved to: $BASE_OUTPUT_DIR"
echo ""
echo "Conditions tested:"
echo "  1. uniform_poisson - Baseline (stratified + smooth arrivals)"
echo "  2. real_poisson    - Realistic emotion skew + smooth arrivals"
echo "  3. uniform_bursty  - Stratified + bursty arrivals (stress test)"
echo "  4. real_bursty     - Most realistic scenario"
echo ""
echo "If SSJF-* maintains advantage across all conditions, the algorithm"
echo "is robust to workload variations."
echo "========================================================================"
