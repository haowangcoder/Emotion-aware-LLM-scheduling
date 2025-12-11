#!/bin/bash

# ============================================================================
# Full Pipeline: Collect Data -> Preprocess -> Train Predictor
# ============================================================================
# This script runs the complete service time predictor training pipeline:
# 1. Collect LLM responses (if USE_CUSTOMIZED=true)
# 2. Preprocess dataset
# 3. Train BERT predictor
#
# Usage: ./scripts/predictor/run_full_pipeline.sh
# ============================================================================

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================
export TASK_TYPE="${TASK_TYPE:-0}"          # 0=regression
export DATA_SIZE="${DATA_SIZE:-100}"         # 10K samples
export USE_L1_LOSS="${USE_L1_LOSS:-true}"
export USE_CUSTOMIZED="${USE_CUSTOMIZED:-false}"
export MODEL_NAME="${MODEL_NAME:-vicuna-13b}"

# For customized dataset collection
export MAX_SAMPLES="${MAX_SAMPLES:-10000}"

# ============================================================================
# SETUP
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/storage/homefs/hw24w089/Emotion-aware-LLM-scheduling"

cd "$PROJECT_ROOT"

echo "========================================================================"
echo "  Service Time Predictor - Full Training Pipeline"
echo "========================================================================"
echo "Task type:       $TASK_TYPE"
echo "Data size:       ${DATA_SIZE}K"
echo "Use L1 loss:     $USE_L1_LOSS"
echo "Use customized:  $USE_CUSTOMIZED"
echo "Model name:      $MODEL_NAME"
echo "========================================================================"
echo ""

# ============================================================================
# STEP 1: COLLECT DATA (if using customized dataset)
# ============================================================================
if [ "$USE_CUSTOMIZED" = "true" ]; then
    echo "========================================================================"
    echo "  Step 1/3: Collecting LLM Responses"
    echo "========================================================================"
    "$SCRIPT_DIR/collect_data.sh"
    echo ""
fi

# ============================================================================
# STEP 2: PREPROCESS DATA
# ============================================================================
echo "========================================================================"
echo "  Step 2/3: Preprocessing Dataset"
echo "========================================================================"
"$SCRIPT_DIR/preprocess_data.sh"
echo ""

# ============================================================================
# STEP 3: TRAIN PREDICTOR
# ============================================================================
echo "========================================================================"
echo "  Step 3/3: Training BERT Predictor"
echo "========================================================================"
"$SCRIPT_DIR/train_predictor.sh"
echo ""

echo "========================================================================"
echo "  Full Pipeline Complete!"
echo "========================================================================"
