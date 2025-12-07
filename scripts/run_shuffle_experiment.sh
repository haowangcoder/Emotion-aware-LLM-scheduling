#!/bin/bash

# ============================================================================
# Emotion-aware LLM Scheduling - Shuffle Experiment
# ============================================================================
# This experiment shuffles arousal/emotion labels while preserving service
# times to verify that SSJF-Emotion benefits come from the emotion→length
# correlation, not from data distribution artifacts.
#
# If the correlation is real:
#   - Original: SSJF-Emotion should outperform FCFS
#   - Shuffled: SSJF-Emotion should perform similar to or worse than FCFS
#
# Usage: ./run_shuffle_experiment.sh
# ============================================================================

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_OUTPUT_DIR="results/shuffle_experiment"
MODE="fixed_jobs"
NUM_JOBS=54
SYSTEM_LOAD=0.9
RANDOM_SEED=42
SHUFFLE_SEED=12345  # Different seed for shuffling

# Schedulers to compare
SCHEDULERS=("FCFS" "SSJF-Emotion" "SSJF-Combined")

echo ""
echo "========================================================================"
echo "  Emotion-aware LLM Scheduling - Shuffle Experiment"
echo "========================================================================"
echo "Mode:           $MODE"
echo "Number of jobs: $NUM_JOBS"
echo "System load:    $SYSTEM_LOAD"
echo "Schedulers:     ${SCHEDULERS[*]}"
echo "Output dir:     $BASE_OUTPUT_DIR"
echo "========================================================================"
echo ""
echo "This experiment tests whether SSJF-Emotion's advantage comes from"
echo "the emotion→length correlation or from data distribution artifacts."
echo ""

# ============================================================================
# PHASE 1: Run with ORIGINAL trace (emotion→length correlation intact)
# ============================================================================
echo "========================================================================"
echo "  PHASE 1: Original Trace (with emotion→length correlation)"
echo "========================================================================"

ORIGINAL_DIR="${BASE_OUTPUT_DIR}/original"

# Generate fresh trace for first scheduler
sed -i 's/force_new_job_config:.*/force_new_job_config: true/'  model-serving/config/default.yaml
sed -i 's/use_saved_job_config:.*/use_saved_job_config: false/' model-serving/config/default.yaml

FIRST_RUN=true
for scheduler in "${SCHEDULERS[@]}"; do
    echo ">>> Running $scheduler on original trace..."

    if [ "$FIRST_RUN" = true ]; then
        # Generate new trace for first run
        FIRST_RUN=false
    else
        # Reuse trace from first run
        sed -i 's/force_new_job_config:.*/force_new_job_config: false/' model-serving/config/default.yaml
        sed -i 's/use_saved_job_config:.*/use_saved_job_config: true/'  model-serving/config/default.yaml
    fi

    uv run python run_simulation.py \
        --mode "$MODE" \
        --scheduler "$scheduler" \
        --num_jobs "$NUM_JOBS" \
        --system_load "$SYSTEM_LOAD" \
        --random_seed "$RANDOM_SEED" \
        --output_dir "$ORIGINAL_DIR" \
        --verbose

    echo "✓ $scheduler (original) completed"
done

# ============================================================================
# PHASE 2: Create SHUFFLED trace
# ============================================================================
echo ""
echo "========================================================================"
echo "  Creating Shuffled Trace"
echo "========================================================================"

SHUFFLED_DIR="${BASE_OUTPUT_DIR}/shuffled"
mkdir -p "${SHUFFLED_DIR}/cache"

# Create shuffled job configs using Python
uv run python << EOF
import json
import sys
sys.path.insert(0, 'model-serving')
from workload.task_generator import shuffle_job_trace_arousal

# Load original trace (format: {"metadata": {...}, "jobs": [...]})
with open('${ORIGINAL_DIR}/cache/job_configs.json', 'r') as f:
    original_data = json.load(f)

original_jobs = original_data['jobs']
print(f"Loaded {len(original_jobs)} jobs from original trace")

# Shuffle arousal values (but keep service_time unchanged)
shuffled_data = shuffle_job_trace_arousal(original_data, random_seed=${SHUFFLE_SEED})
shuffled_jobs = shuffled_data['jobs']

# Verify service times are unchanged
original_service_times = [j['service_time'] for j in original_jobs]
shuffled_service_times = [j['service_time'] for j in shuffled_jobs]
assert original_service_times == shuffled_service_times, "Service times should be unchanged!"

# Verify arousal values are shuffled
original_arousals = [j['arousal'] for j in original_jobs]
shuffled_arousals = [j['arousal'] for j in shuffled_jobs]
correlation_broken = sum(1 for a, b in zip(original_arousals, shuffled_arousals) if a != b)
print(f"Shuffled {correlation_broken}/{len(original_jobs)} arousal assignments")

# Save shuffled trace
with open('${SHUFFLED_DIR}/cache/job_configs.json', 'w') as f:
    json.dump(shuffled_data, f, indent=2)

print("Shuffled trace saved to ${SHUFFLED_DIR}/cache/job_configs.json")

# Also copy responses.json if it exists
import shutil
try:
    shutil.copy('${ORIGINAL_DIR}/cache/responses.json', '${SHUFFLED_DIR}/cache/responses.json')
    print("Copied responses.json")
except FileNotFoundError:
    pass
EOF

# ============================================================================
# PHASE 3: Run with SHUFFLED trace (emotion→length correlation broken)
# ============================================================================
echo ""
echo "========================================================================"
echo "  PHASE 3: Shuffled Trace (emotion→length correlation BROKEN)"
echo "========================================================================"

# Use saved shuffled trace
sed -i 's/force_new_job_config:.*/force_new_job_config: false/' model-serving/config/default.yaml
sed -i 's/use_saved_job_config:.*/use_saved_job_config: true/'  model-serving/config/default.yaml

for scheduler in "${SCHEDULERS[@]}"; do
    echo ">>> Running $scheduler on shuffled trace..."

    uv run python run_simulation.py \
        --mode "$MODE" \
        --scheduler "$scheduler" \
        --num_jobs "$NUM_JOBS" \
        --system_load "$SYSTEM_LOAD" \
        --random_seed "$RANDOM_SEED" \
        --output_dir "$SHUFFLED_DIR" \
        --verbose

    echo "✓ $scheduler (shuffled) completed"
done

# ============================================================================
# PHASE 4: Generate Analysis and Plots
# ============================================================================
echo ""
echo "========================================================================"
echo "  Generating Shuffle Experiment Analysis"
echo "========================================================================"

uv run python -c "
import sys
sys.path.insert(0, 'analysis')
from plotting import load_shuffle_experiment_results, generate_shuffle_experiment_plots
from plotting.utils import setup_publication_style

setup_publication_style(dpi=300)

results = load_shuffle_experiment_results('results/shuffle_experiment')

# Print comparison table
print()
print('=' * 80)
print('Shuffle Experiment Results: Original vs Shuffled')
print('=' * 80)
print()
print('HYPOTHESIS: If SSJF-Emotion works because of emotion→length correlation,')
print('            its advantage should DISAPPEAR when we shuffle arousal labels.')
print()
print('-' * 80)
print(f\"{'Scheduler':<15} {'Condition':<12} {'Avg Wait':<12} {'P99 Wait':<12} {'Jain':<10}\")
print('-' * 80)

for condition in ['original', 'shuffled']:
    for sched, data in results[condition].items():
        print(f\"{sched:<15} {condition:<12} {data['avg_wait']:>10.2f}s {data['p99']:>10.2f}s {data['jain']:>9.4f}\")
    print()

print('-' * 80)
print()

# Calculate and show deltas
print('Performance Delta (Shuffled - Original):')
print('-' * 80)
for sched in results['original'].keys():
    if sched in results['shuffled']:
        orig = results['original'][sched]
        shuf = results['shuffled'][sched]
        delta_avg = shuf['avg_wait'] - orig['avg_wait']
        delta_p99 = shuf['p99'] - orig['p99']
        delta_jain = shuf['jain'] - orig['jain']
        print(f\"{sched:<15} Δavg={delta_avg:+.2f}s  Δp99={delta_p99:+.2f}s  Δjain={delta_jain:+.4f}\")

print()
print('INTERPRETATION:')
print('  - FCFS: Should show ~0 delta (not affected by emotion labels)')
print('  - SSJF-Emotion: Positive Δavg = worse with shuffle = correlation was real!')
print('  - SSJF-Combined: Similar pattern expected')
print('=' * 80)

# Generate plots
generate_shuffle_experiment_plots(
    results,
    'results/shuffle_experiment/plots',
    formats=['pdf', 'png']
)
print()
print('Plots saved to: results/shuffle_experiment/plots/')
"

# ============================================================================
# COMPLETION
# ============================================================================
echo ""
echo "========================================================================"
echo "  Shuffle Experiment Completed!"
echo "========================================================================"
echo "Results saved to: $BASE_OUTPUT_DIR"
echo ""
echo "Key insight:"
echo "  If SSJF-Emotion performs WORSE on shuffled data, it proves that"
echo "  the algorithm genuinely exploits the emotion→length correlation,"
echo "  not just data distribution artifacts."
echo "========================================================================"
