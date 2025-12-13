#!/usr/bin/env python
"""
Evaluate BERT bucket predictor with expected value method.

Evaluates both:
1. Classification accuracy (bin prediction)
2. Expected value MAE (token prediction using E[T] = sum(q_i * m_i))
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def bin_to_range(bin_idx, bin_edges):
    """Get token range for a bin using provided bin edges."""
    low = int(bin_edges[bin_idx])
    high = int(bin_edges[bin_idx + 1]) if bin_idx + 1 < len(bin_edges) else int(bin_edges[-1])
    return low, high


def main():
    parser = argparse.ArgumentParser(description='Evaluate BERT bucket predictor')
    parser.add_argument('--model_path', type=str, default='model-serving/predictor/models/bert_bucket',
                        help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='model-serving/predictor/training/data',
                        help='Directory with valid.json and bin_edges.npy')
    parser.add_argument('--num_labels', type=int, default=5,
                        help='Number of classification bins')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of samples to evaluate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    args = parser.parse_args()

    # Check paths
    model_path = Path(args.model_path)
    data_dir = Path(args.data_dir)

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    valid_path = data_dir / 'valid.json'
    if not valid_path.exists():
        print(f"Error: Validation data not found at {valid_path}")
        return

    # Load bin edges
    bin_edges_path = data_dir / 'bin_edges.npy'
    if bin_edges_path.exists():
        bin_edges = np.load(bin_edges_path)
        print(f"Bin edges: {bin_edges.astype(int).tolist()}")
    else:
        # Check model directory
        model_bin_edges = model_path.parent / 'bin_edges.npy'
        if model_bin_edges.exists():
            bin_edges = np.load(model_bin_edges)
            print(f"Bin edges (from model dir): {bin_edges.astype(int).tolist()}")
        else:
            # Default equal-width bins
            bin_edges = np.linspace(0, 256, args.num_labels + 1)
            print(f"Using default bin edges: {bin_edges.astype(int).tolist()}")

    num_bins = len(bin_edges) - 1

    # Compute bin midpoints
    bin_midpoints = np.array([
        (bin_edges[i] + bin_edges[i + 1]) / 2
        for i in range(num_bins)
    ])
    print(f"Bin midpoints: {bin_midpoints.astype(int).tolist()}")

    # Load model
    print(f"\nLoading model from {model_path}")
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_bins
    ).to(device)
    model.eval()

    # Load validation data
    print(f"Loading validation data from {valid_path}")
    valid_data = []
    with open(valid_path, 'r') as f:
        for line in f:
            valid_data.append(json.loads(line))

    if args.limit:
        valid_data = valid_data[:args.limit]

    print(f"Evaluating {len(valid_data)} samples\n")

    # Evaluate
    results = []
    correct = 0
    within_1 = 0
    total_bin_error = 0
    total_token_error = 0

    bin_midpoints_tensor = torch.tensor(bin_midpoints, dtype=torch.float32, device=device)

    print("=" * 95)
    print(f"{'ID':>4} | {'Actual':>20} | {'Pred Bin':>15} | {'Expected T':>10} | {'Err':>6} | Status")
    print("=" * 95)

    for item in tqdm(valid_data, desc="Evaluating", leave=False):
        record_id = item['id']
        prompt = item['prompt']
        actual_bin = item['label']
        actual_tokens = item.get('output_tokens', 0)

        # Tokenize and predict
        inputs = tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1).squeeze()
            pred_bin = logits.argmax(dim=1).item()

            # Expected value: T_mean = sum(q_i * m_i)
            expected_tokens = torch.dot(probs, bin_midpoints_tensor).item()

        # Calculate metrics
        bin_error = abs(pred_bin - actual_bin)
        token_error = abs(expected_tokens - actual_tokens)
        is_correct = (pred_bin == actual_bin)
        is_within_1 = bin_error <= 1

        if is_correct:
            correct += 1
        if is_within_1:
            within_1 += 1
        total_bin_error += bin_error
        total_token_error += token_error

        # Get ranges for display
        actual_range = bin_to_range(actual_bin, bin_edges)
        pred_range = bin_to_range(pred_bin, bin_edges)

        status = "ok" if is_correct else ("~" if is_within_1 else "X")

        print(f"{record_id:>4} | {actual_tokens:>3} tok (bin {actual_bin}: {actual_range[0]:>3}-{actual_range[1]:<3}) | "
              f"bin {pred_bin}: {pred_range[0]:>3}-{pred_range[1]:<3} | {expected_tokens:>10.1f} | {token_error:>6.1f} | {status}")

        results.append({
            'record_id': record_id,
            'actual_tokens': actual_tokens,
            'actual_bin': actual_bin,
            'pred_bin': pred_bin,
            'expected_tokens': expected_tokens,
            'bin_error': bin_error,
            'token_error': token_error,
            'correct': is_correct,
        })

    # Summary statistics
    n = len(results)
    accuracy = correct / n * 100
    within_1_acc = within_1 / n * 100
    mae_bin = total_bin_error / n
    mae_tokens = total_token_error / n

    print("=" * 95)
    print(f"\n{'Summary':^95}")
    print("=" * 95)
    print(f"Total samples evaluated:      {n}")
    print(f"Exact bin accuracy:           {correct}/{n} = {accuracy:.1f}%")
    print(f"Within +-1 bin accuracy:      {within_1}/{n} = {within_1_acc:.1f}%")
    print(f"Mean Absolute Bin Error:      {mae_bin:.2f}")
    print(f"Mean Absolute Token Error:    {mae_tokens:.1f} tokens (using E[T] = sum(q*m))")
    print()

    # Distribution of bin errors
    print("Bin Error Distribution:")
    error_counts = {}
    for r in results:
        e = r['bin_error']
        error_counts[e] = error_counts.get(e, 0) + 1
    for e in sorted(error_counts.keys()):
        bar_len = min(error_counts[e] * 50 // n, 50)
        bar = "#" * bar_len
        print(f"  Error {e}: {error_counts[e]:>4} ({100*error_counts[e]/n:>5.1f}%) {bar}")

    # Per-bin accuracy
    print("\nPer-bin Accuracy:")
    for bin_idx in range(num_bins):
        bin_samples = [r for r in results if r['actual_bin'] == bin_idx]
        if len(bin_samples) > 0:
            bin_correct = sum(1 for r in bin_samples if r['correct'])
            bin_range = bin_to_range(bin_idx, bin_edges)
            print(f"  Bin {bin_idx} ({bin_range[0]:>3}-{bin_range[1]:<3}): "
                  f"{bin_correct:>4}/{len(bin_samples):<4} = {100*bin_correct/len(bin_samples):>5.1f}%")

    # Expected value error analysis
    print("\nExpected Value (E[T]) Error Analysis:")
    token_errors = [r['token_error'] for r in results]
    print(f"  Min error:    {min(token_errors):.1f} tokens")
    print(f"  Max error:    {max(token_errors):.1f} tokens")
    print(f"  Median error: {np.median(token_errors):.1f} tokens")
    print(f"  90th pctl:    {np.percentile(token_errors, 90):.1f} tokens")

    print("\nDone!")


if __name__ == '__main__':
    main()
