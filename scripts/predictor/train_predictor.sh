#!/bin/bash

# ============================================================================
# Train BERT-based Service Time Predictor
# ============================================================================
# This script trains the BERT regression/classification model for predicting
# LLM output token lengths (and thus service times).
#
# Usage: ./scripts/predictor/train_predictor.sh [OPTIONS]
# ============================================================================

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================
# Task type: 0=regression, 1=binary classification, 2=multi-class classification
TASK_TYPE="${TASK_TYPE:-0}"
DATA_SIZE="${DATA_SIZE:-100}"  # In thousands (10 = 10K samples)
USE_L1_LOSS="${USE_L1_LOSS:-true}"  # Use L1 loss for regression (else MSE)
USE_CUSTOMIZED="${USE_CUSTOMIZED:-true}"
CUSTOMIZED_DATASET_PATH="${CUSTOMIZED_DATASET_PATH:-data/customized_100K}"
MODEL_NAME="${MODEL_NAME:-vicuna-13b}"

# ============================================================================
# SETUP
# ============================================================================
cd /storage/homefs/hw24w089/Emotion-aware-LLM-scheduling/model-serving/predictor/training

echo "========================================================================"
echo "  BERT Predictor Training"
echo "========================================================================"
echo "Task type:       $TASK_TYPE (0=regression, 1=binary-cls, 2=multi-cls)"
echo "Data size:       ${DATA_SIZE}K samples"
echo "Use L1 loss:     $USE_L1_LOSS"
echo "Use customized:  $USE_CUSTOMIZED"
echo "Model name:      $MODEL_NAME"
echo "========================================================================"
echo ""

# ============================================================================
# BUILD COMMAND
# ============================================================================
CMD="uv run python latency_prediction.py --task_type $TASK_TYPE --data_size $DATA_SIZE"

if [ "$USE_L1_LOSS" = "true" ]; then
    CMD="$CMD --l1_loss"
fi

if [ "$USE_CUSTOMIZED" = "true" ]; then
    CMD="$CMD --customized --dataset_path $CUSTOMIZED_DATASET_PATH"
else
    CMD="$CMD --model_name $MODEL_NAME"
fi

# ============================================================================
# RUN TRAINING
# ============================================================================
echo ">>> Running: $CMD"
echo ""
$CMD

echo ""
echo "========================================================================"
echo "  Training Complete!"
echo "========================================================================"
echo "Model weights saved to: ./models/"
echo "Results saved to:       ./results/"
echo "Metrics saved to:       ./metrics/"
echo "========================================================================"
