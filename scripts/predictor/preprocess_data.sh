#!/bin/bash

# ============================================================================
# Preprocess Dataset for BERT Predictor Training
# ============================================================================
# This script preprocesses collected LLM responses for training.
# Supports both LMSYS public dataset and customized (empathetic) dataset.
#
# Usage: ./scripts/predictor/preprocess_data.sh [OPTIONS]
# ============================================================================

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================
# Task type: 0=regression, 1=binary classification, 2=multi-class classification
TASK_TYPE="${TASK_TYPE:-0}"
DATA_SIZE="${DATA_SIZE:-100}"  # In thousands (10 = 10K samples)
USE_CUSTOMIZED="${USE_CUSTOMIZED:-false}"
DATASET_PATH="${DATASET_PATH:-./data/empathetic_responses.csv}"
MODEL_NAME="${MODEL_NAME:-vicuna-13b}"

# ============================================================================
# SETUP
# ============================================================================
cd /storage/homefs/hw24w089/Emotion-aware-LLM-scheduling/model-serving/predictor/training

echo "========================================================================"
echo "  Dataset Preprocessing for BERT Predictor"
echo "========================================================================"
echo "Task type:       $TASK_TYPE (0=regression, 1=binary-cls, 2=multi-cls)"
echo "Data size:       ${DATA_SIZE}K samples"
echo "Use customized:  $USE_CUSTOMIZED"
echo "Dataset path:    $DATASET_PATH"
echo "Model name:      $MODEL_NAME"
echo "========================================================================"
echo ""

# ============================================================================
# RUN PREPROCESSING
# ============================================================================
if [ "$USE_CUSTOMIZED" = "true" ]; then
    echo ">>> Using customized dataset (EmpatheticDialogues responses)..."
    uv run python preprocess_customized_dataset.py \
        --task_type "$TASK_TYPE" \
        --data_size "$DATA_SIZE" \
        --dataset_path "$DATASET_PATH" \
        --single_model
else
    echo ">>> Using LMSYS public dataset..."
    uv run python preprocess_dataset.py \
        --task_type "$TASK_TYPE" \
        --data_size "$DATA_SIZE" \
        --model_name "$MODEL_NAME"
fi

echo ""
echo "========================================================================"
echo "  Preprocessing Complete!"
echo "========================================================================"
echo "Processed data saved to: ./data/"
echo "========================================================================"
