#!/bin/bash
# Train BERT bucket predictor for length prediction
#
# Usage (from project root):
#   ./scripts/predictor/train.sh
#
# Prerequisites:
#   - Dataset CSVs in dataset/ (train.csv, valid.csv)
#   - Metadata JSONs in model-serving/results/cache/

set -e

# Change to project root
cd "$(dirname "$0")/../.."

echo "=== BERT Bucket Predictor Training Pipeline ==="
echo "Working directory: $(pwd)"
echo ""

# Configuration
NUM_BINS=5
MAX_LENGTH=256
WEIGHT_METHOD=inverse
EPOCHS=5

# Step 1: Prepare data
echo "Step 1: Preparing training data..."
uv run python model-serving/predictor/training/prepare_data.py \
    --dataset dataset \
    --metadata_dir model-serving/results/cache \
    --output_dir model-serving/predictor/training/data \
    --num_bins $NUM_BINS \
    --max_length $MAX_LENGTH

echo ""

# Step 2: Train model
echo "Step 2: Training BERT classifier..."
uv run python model-serving/predictor/training/train.py \
    --ds_dir model-serving/predictor/training/data \
    --outdir model-serving/predictor/models \
    --model_name bert_bucket \
    --num_labels $NUM_BINS \
    --weight_method $WEIGHT_METHOD \
    --epochs $EPOCHS

echo ""

# Step 3: Copy bin_edges to model directory
echo "Step 3: Copying bin_edges.npy..."
cp model-serving/predictor/training/data/bin_edges.npy model-serving/predictor/models/

echo ""

# Step 4: Evaluate
echo "Step 4: Evaluating model..."
uv run python model-serving/predictor/training/evaluate.py \
    --model_path model-serving/predictor/models/bert_bucket \
    --data_dir model-serving/predictor/training/data \
    --num_labels $NUM_BINS \
    --limit 100

echo ""
echo "=== Training Complete ==="
echo ""
echo "Model saved to: model-serving/predictor/models/bert_bucket/"
echo "Bin edges saved to: model-serving/predictor/models/bin_edges.npy"
