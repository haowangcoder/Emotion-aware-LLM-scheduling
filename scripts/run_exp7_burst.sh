#!/bin/bash
# Exp-7: Burst Traffic (MMPP Arrival Patterns)
#
# Purpose: Validate AW-SSJF robustness under realistic bursty traffic
# Parameters: Burst intensity ratios (mild=3x, moderate=6.7x, severe=15x)
# Schedulers: FCFS, SJF, AW-SSJF(k=2), AW-SSJF(k=4)
#
# MMPP Model:
#   Two-state Markov Modulated Poisson Process
#   - HIGH state: burst period with rate lambda_high
#   - LOW state: normal period with rate lambda_low
#   - alpha: HIGH->LOW transition rate (mean burst duration = 1/alpha)
#   - beta: LOW->HIGH transition rate (mean normal duration = 1/beta)
#
# Usage:
#   bash scripts/run_exp7_burst.sh

set -e

# Configuration
OUTPUT_BASE="results/experiments/exp7_burst"
NUM_JOBS=100
W_MAX=2.0
RANDOM_SEED=42
K_VALUES=(2 4)

# Create output directory
mkdir -p "$OUTPUT_BASE"

echo "=========================================="
echo "Exp-7: Burst Traffic (MMPP)"
echo "=========================================="
echo "Parameters:"
echo "  burst_configs: mild, moderate, severe"
echo "  num_jobs: $NUM_JOBS"
echo "  w_max: $W_MAX"
echo "  output: $OUTPUT_BASE"
echo "=========================================="

# Function to run experiment for a given burst configuration
run_burst_config() {
    local burst_name=$1
    local lambda_high=$2
    local lambda_low=$3
    local alpha=$4
    local beta=$5

    echo ""
    echo "=========================================="
    echo ">>> Burst Config: $burst_name"
    echo "=========================================="
    echo "  lambda_high=$lambda_high, lambda_low=$lambda_low"
    echo "  alpha=$alpha (mean burst=$(echo "scale=1; 1/$alpha" | bc)s)"
    echo "  beta=$beta (mean normal=$(echo "scale=1; 1/$beta" | bc)s)"
    echo "  intensity=$(echo "scale=1; $lambda_high/$lambda_low" | bc)x"

    BURST_DIR="${OUTPUT_BASE}/${burst_name}"
    mkdir -p "$BURST_DIR"

    FIRST=true

    # Run FCFS
    echo ">>> Running FCFS"
    if [ "$FIRST" = true ]; then
        JOB_CONFIG_FLAG="--force_new_job_config"
        FIRST=false
    else
        JOB_CONFIG_FLAG="--use_saved_job_config"
    fi

    uv run python run_simulation.py \
        --scheduler FCFS \
        --num_jobs "$NUM_JOBS" \
        --random_seed "$RANDOM_SEED" \
        --mode fixed_jobs \
        --output_dir "$BURST_DIR" \
        --mmpp_enabled \
        --mmpp_lambda_high "$lambda_high" \
        --mmpp_lambda_low "$lambda_low" \
        --mmpp_alpha "$alpha" \
        --mmpp_beta "$beta" \
        $JOB_CONFIG_FLAG

    # Run SJF
    echo ">>> Running SJF"
    uv run python run_simulation.py \
        --scheduler SJF \
        --num_jobs "$NUM_JOBS" \
        --random_seed "$RANDOM_SEED" \
        --mode fixed_jobs \
        --output_dir "$BURST_DIR" \
        --mmpp_enabled \
        --mmpp_lambda_high "$lambda_high" \
        --mmpp_lambda_low "$lambda_low" \
        --mmpp_alpha "$alpha" \
        --mmpp_beta "$beta" \
        --use_saved_job_config

    # Run AW-SSJF with different k values
    for k in "${K_VALUES[@]}"; do
        echo ">>> Running AW-SSJF (k=$k)"
        uv run python run_simulation.py \
            --scheduler AW-SSJF \
            --weight_exponent "$k" \
            --w_max "$W_MAX" \
            --num_jobs "$NUM_JOBS" \
            --random_seed "$RANDOM_SEED" \
            --mode fixed_jobs \
            --output_dir "$BURST_DIR" \
            --mmpp_enabled \
            --mmpp_lambda_high "$lambda_high" \
            --mmpp_lambda_low "$lambda_low" \
            --mmpp_alpha "$alpha" \
            --mmpp_beta "$beta" \
            --use_saved_job_config
    done

    echo ">>> Completed burst_config=$burst_name"
}

# Run mild burst configuration (3x intensity)
# Normal diurnal variation
run_burst_config "mild" 1.5 0.5 0.2 0.1

# Run moderate burst configuration (6.7x intensity)
# Peak hours vs off-peak
run_burst_config "moderate" 2.0 0.3 0.15 0.05

# Run severe burst configuration (15x intensity)
# Flash crowd / viral event
run_burst_config "severe" 3.0 0.2 0.1 0.03

echo ""
echo "=========================================="
echo "Burst traffic simulations complete! Running analysis..."
echo "=========================================="

uv run python experiments/exp7_burst_traffic.py --input_dir "$OUTPUT_BASE"

echo ""
echo "=========================================="
echo "Exp-7 complete!"
echo "=========================================="
echo "Results: $OUTPUT_BASE"
echo ""
echo "Output structure:"
echo "  $OUTPUT_BASE/"
echo "  ├── mild/       (3x burst intensity)"
echo "  ├── moderate/   (6.7x burst intensity)"
echo "  ├── severe/     (15x burst intensity)"
echo "  └── plots/"
echo "      ├── depression_vs_burst.png"
echo "      ├── quadrant_heatmap_severe.png"
echo "      ├── tail_latency_comparison.png"
echo "      ├── avg_wait_comparison.png"
echo "      ├── depression_speedup_vs_burst.png"
echo "      └── exp7_report.json"
