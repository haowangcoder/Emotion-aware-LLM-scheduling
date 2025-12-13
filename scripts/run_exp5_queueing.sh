#!/bin/bash
# Exp-5: Queueing Model Validation
#
# Purpose: Validate M/G/1 queueing model and regression approximation
# Input: Uses results from exp2_load_sweep
#
# This is an analysis-only experiment - it doesn't run new simulations.
# It requires exp2_load_sweep to have been run first.
#
# Usage:
#   bash scripts/run_exp5_queueing.sh

set -e

INPUT_DIR="results/experiments/exp2_load_sweep"
OUTPUT_DIR="results/experiments/exp5_queueing"

echo "=========================================="
echo "Exp-5: Queueing Model Validation"
echo "=========================================="
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# Check if exp2 results exist
if [ ! -d "$INPUT_DIR" ]; then
    echo ""
    echo "ERROR: exp2_load_sweep results not found at $INPUT_DIR"
    echo "Please run exp2 first:"
    echo "  bash scripts/run_exp2_load_sweep.sh"
    exit 1
fi

# Count result directories (handles both load0.5 and load_0.5 patterns)
NUM_LOADS=$(find "$INPUT_DIR" -maxdepth 1 -type d -name "load*" | grep -v plots | wc -l)
if [ "$NUM_LOADS" -eq 0 ]; then
    echo ""
    echo "ERROR: No load* directories found in $INPUT_DIR"
    echo "Please run exp2 first:"
    echo "  bash scripts/run_exp2_load_sweep.sh"
    exit 1
fi

echo ""
echo "Found $NUM_LOADS load configurations from exp2"
echo ""

# Run analysis
uv run python experiments/exp5_queueing_model.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Exp-5 complete!"
echo "=========================================="
echo "Results: $OUTPUT_DIR"
