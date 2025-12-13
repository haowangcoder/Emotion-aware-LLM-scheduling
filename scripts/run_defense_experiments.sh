#!/bin/bash
# Defense Experiments (A1-A3)
#
# Purpose: Run experiments to defend against potential reviewer questions
#
# A1: Starvation Prevention - Fix P99 tail latency
# A2: Predictor Contribution (Oracle vs Predicted vs Disabled)
# A3: Emotion Noise Robustness
#
# Usage:
#   bash scripts/run_defense_experiments.sh [A1|A2|A3|all]

set -e

EXPERIMENT="${1:-all}"
OUTPUT_BASE="results/experiments/defense"
NUM_JOBS=80
SYSTEM_LOAD=0.9
W_MAX=2.0
K=4
SEED=42

mkdir -p "$OUTPUT_BASE"

echo "=========================================="
echo "Defense Experiments"
echo "=========================================="
echo "Experiment: $EXPERIMENT"
echo "Parameters:"
echo "  num_jobs: $NUM_JOBS"
echo "  system_load: $SYSTEM_LOAD"
echo "  seed: $SEED"
echo "=========================================="

# =============================================================================
# A1: Starvation Prevention Sweep
# =============================================================================
run_a1() {
    echo ""
    echo "=========================================="
    echo "A1: Starvation Prevention Sweep"
    echo "=========================================="

    A1_DIR="${OUTPUT_BASE}/A1"
    mkdir -p "$A1_DIR"

    THRESHOLDS=("inf" "30" "60" "120")
    COEFFICIENTS=("2" "3" "4" "5")

    FIRST=true
    for threshold in "${THRESHOLDS[@]}"; do
        for coef in "${COEFFICIENTS[@]}"; do
            echo ">>> Running threshold=$threshold, coef=$coef"

            if [ "$FIRST" = true ]; then
                JOB_CONFIG_FLAG="--force_new_job_config"
                FIRST=false
            else
                JOB_CONFIG_FLAG="--use_saved_job_config"
            fi

            # Map threshold for CLI
            if [ "$threshold" = "inf" ]; then
                THRESHOLD_ARG="999999"
            else
                THRESHOLD_ARG="$threshold"
            fi

            uv run python run_simulation.py \
                --scheduler AW-SSJF \
                --weight_exponent "$K" \
                --w_max "$W_MAX" \
                --starvation_threshold "$THRESHOLD_ARG" \
                --starvation_coefficient "$coef" \
                --system_load "$SYSTEM_LOAD" \
                --num_jobs "$NUM_JOBS" \
                --random_seed "$SEED" \
                --mode fixed_jobs \
                --output_dir "${A1_DIR}/threshold_${threshold}_coef_${coef}" \
                $JOB_CONFIG_FLAG
        done
    done

    echo ">>> A1 simulation complete. Running analysis..."
    uv run python experiments/defense_experiments.py --experiment A1 --input_dir "$A1_DIR"
}

# =============================================================================
# A2: Predictor Contribution
# =============================================================================
run_a2() {
    echo ""
    echo "=========================================="
    echo "A2: Predictor Contribution"
    echo "=========================================="

    A2_DIR="${OUTPUT_BASE}/A2"
    mkdir -p "$A2_DIR"

    PREDICTOR_MODES=("oracle" "predicted" "disabled")

    FIRST=true
    for mode in "${PREDICTOR_MODES[@]}"; do
        echo ">>> Running predictor_mode=$mode"

        if [ "$FIRST" = true ]; then
            JOB_CONFIG_FLAG="--force_new_job_config"
            FIRST=false
        else
            JOB_CONFIG_FLAG="--use_saved_job_config"
        fi

        # Set predictor mode flags
        case "$mode" in
            "oracle")
                PREDICTOR_FLAG="--use_oracle_service_time"
                ;;
            "predicted")
                PREDICTOR_FLAG=""
                ;;
            "disabled")
                PREDICTOR_FLAG="--disable_predictor"
                ;;
        esac

        uv run python run_simulation.py \
            --scheduler AW-SSJF \
            --weight_exponent "$K" \
            --w_max "$W_MAX" \
            --system_load "$SYSTEM_LOAD" \
            --num_jobs "$NUM_JOBS" \
            --random_seed "$SEED" \
            --mode fixed_jobs \
            --output_dir "${A2_DIR}/${mode}" \
            $JOB_CONFIG_FLAG \
            $PREDICTOR_FLAG
    done

    echo ">>> A2 simulation complete. Running analysis..."
    uv run python experiments/defense_experiments.py --experiment A2 --input_dir "$A2_DIR"
}

# =============================================================================
# A3: Emotion Noise Robustness
# =============================================================================
run_a3() {
    echo ""
    echo "=========================================="
    echo "A3: Emotion Noise Robustness"
    echo "=========================================="

    A3_DIR="${OUTPUT_BASE}/A3"
    mkdir -p "$A3_DIR"

    NOISE_LEVELS=("0.0" "0.1" "0.2" "0.3" "0.5")

    FIRST=true
    for noise in "${NOISE_LEVELS[@]}"; do
        echo ">>> Running emotion_noise=$noise"

        if [ "$FIRST" = true ]; then
            JOB_CONFIG_FLAG="--force_new_job_config"
            FIRST=false
        else
            JOB_CONFIG_FLAG="--use_saved_job_config"
        fi

        uv run python run_simulation.py \
            --scheduler AW-SSJF \
            --weight_exponent "$K" \
            --w_max "$W_MAX" \
            --arousal_noise "$noise" \
            --system_load "$SYSTEM_LOAD" \
            --num_jobs "$NUM_JOBS" \
            --random_seed "$SEED" \
            --mode fixed_jobs \
            --output_dir "${A3_DIR}/noise_${noise}" \
            $JOB_CONFIG_FLAG
    done

    echo ">>> A3 simulation complete. Running analysis..."
    uv run python experiments/defense_experiments.py --experiment A3 --input_dir "$A3_DIR"
}

# =============================================================================
# Main execution
# =============================================================================
case "$EXPERIMENT" in
    "A1")
        run_a1
        ;;
    "A2")
        run_a2
        ;;
    "A3")
        run_a3
        ;;
    "all")
        run_a1
        run_a2
        run_a3
        ;;
    *)
        echo "Usage: $0 [A1|A2|A3|all]"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Defense experiments complete!"
echo "=========================================="
echo "Results: $OUTPUT_BASE"
echo "Analysis: uv run python experiments/defense_experiments.py --experiment all --input_dir $OUTPUT_BASE"
