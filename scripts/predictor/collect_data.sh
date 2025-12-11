#!/bin/bash

# ============================================================================
# Collect LLM Responses for Service Time Predictor Training
# ============================================================================
# This script collects real LLM responses from EmpatheticDialogues dataset.
# Output is saved as CSV for preprocessing and training.
#
# Usage: ./scripts/predictor/collect_data.sh [OPTIONS]
# ============================================================================

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================
MODEL_NAME="${MODEL_NAME:-lmsys/vicuna-13b-v1.3}"
DATASET_PATH="${DATASET_PATH:-./dataset}"
OUTPUT_PATH="${OUTPUT_PATH:-./model-serving/predictor/training/data/empathetic_responses.csv}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./model-serving/predictor/training/data/checkpoint.json}"
LOG_FILE="${LOG_FILE:-./model-serving/predictor/training/logs/collection.log}"

# MAX_SAMPLES="${MAX_SAMPLES:-5000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-0.0}"
BATCH_SAVE_INTERVAL="${BATCH_SAVE_INTERVAL:-100}"

# ============================================================================
# SETUP
# ============================================================================
cd /storage/homefs/hw24w089/Emotion-aware-LLM-scheduling

# Ensure directories exist
mkdir -p "$(dirname "$OUTPUT_PATH")"
mkdir -p "$(dirname "$LOG_FILE")"

echo "========================================================================"
echo "  LLM Response Collection for Service Time Predictor"
echo "========================================================================"
echo "Model:           $MODEL_NAME"
echo "Dataset path:    $DATASET_PATH"
echo "Output path:     $OUTPUT_PATH"
echo "Checkpoint:      $CHECKPOINT_PATH"
echo "Log file:        $LOG_FILE"
# echo "Max samples:     $MAX_SAMPLES"
echo "Max new tokens:  $MAX_NEW_TOKENS"
echo "Temperature:     $TEMPERATURE"
echo "========================================================================"
echo ""

# ============================================================================
# RUN COLLECTION
# ============================================================================
uv run python model-serving/predictor/training/collect_llm_responses.py \
    --model_name "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH" \
    --output_path "$OUTPUT_PATH" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --log_file "$LOG_FILE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --batch_save_interval "$BATCH_SAVE_INTERVAL" \
    --log_level INFO

echo ""
echo "========================================================================"
echo "  Collection Complete!"
echo "========================================================================"
echo "Output saved to: $OUTPUT_PATH"
echo "========================================================================"
