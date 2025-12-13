#!/usr/bin/env python
"""
Prepare BERT training data for bucket classification.

Combines:
- prompts from original CSV
- output_tokens from metadata (already generated)

Output: train.json, valid.json (JSONL format), bin_edges.npy
"""

import argparse
import csv
import json
import sys
import numpy as np
from pathlib import Path

csv.field_size_limit(sys.maxsize)


def load_prompts_from_csv(csv_path):
    """Load prompts from original CSV, only first turn."""
    prompts = {}
    record_id = 0
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('utterance_idx', '1') != '1':
                continue
            context = row.get('prompt', '').replace('_comma_', ',')
            prompt = f"Context: {context}\n\nRespond appropriately to continue the conversation."
            prompts[record_id] = prompt
            record_id += 1
    return prompts


def tokens_to_bin(num_tokens, bin_edges):
    """Convert token count to bin index."""
    bin_idx = np.digitize([num_tokens], bin_edges)[0] - 1
    return int(np.clip(bin_idx, 0, len(bin_edges) - 2))


def main():
    parser = argparse.ArgumentParser(description='Prepare BERT bucket training data')
    parser.add_argument('--dataset', type=str, default='dataset',
                        help='Dataset directory with CSVs')
    parser.add_argument('--metadata_dir', type=str, default='model-serving/results/cache',
                        help='Directory with metadata JSONs')
    parser.add_argument('--output_dir', type=str, default='model-serving/predictor/training/data',
                        help='Output directory')
    parser.add_argument('--num_bins', type=int, default=5,
                        help='Number of bins')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Max sequence length for equal_width binning')
    parser.add_argument('--binning', type=str, default='equal_width',
                        choices=['equal_width', 'quantile'],
                        help='Binning strategy')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process train and valid splits
    for split in ['train', 'valid']:
        print(f"\nProcessing {split}...")

        # Load prompts from CSV
        csv_path = f"{args.dataset}/{split}.csv"
        print(f"  Loading prompts from {csv_path}")
        prompts = load_prompts_from_csv(csv_path)
        print(f"  Loaded {len(prompts)} prompts")

        # Load metadata
        metadata_path = f"{args.metadata_dir}/{split}_metadata.json"
        print(f"  Loading metadata from {metadata_path}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        records = metadata['records']
        print(f"  Loaded {len(records)} records")

        # Collect all output_tokens for binning
        all_tokens = [r['output_tokens'] for r in records]

        # Compute bin edges
        if split == 'train':
            if args.binning == 'quantile':
                percentiles = np.linspace(0, 100, args.num_bins + 1)
                bin_edges = np.percentile(all_tokens, percentiles)
                bin_edges = np.unique(bin_edges)
            else:
                bin_edges = np.linspace(0, args.max_length, args.num_bins + 1)

            # Save bin edges for later use
            np.save(output_dir / 'bin_edges.npy', bin_edges)
            print(f"  Bin edges ({args.binning}): {bin_edges.astype(int).tolist()}")
        else:
            # Use same bin edges as train
            bin_edges = np.load(output_dir / 'bin_edges.npy')

        # Create BERT training data
        bert_data = []
        for record in records:
            record_id = record['record_id']
            if record_id not in prompts:
                print(f"  Warning: record_id {record_id} not found in CSV")
                continue

            output_tokens = record['output_tokens']
            label = tokens_to_bin(output_tokens, bin_edges)

            bert_data.append({
                'id': str(record_id),
                'prompt': prompts[record_id],
                'label': label,
                'output_tokens': output_tokens,  # Keep for evaluation
            })

        # Save as JSONL (one object per line for HuggingFace datasets)
        output_path = output_dir / f'{split}.json'
        with open(output_path, 'w') as f:
            for item in bert_data:
                f.write(json.dumps(item) + '\n')

        print(f"  Saved {len(bert_data)} records to {output_path}")

        # Print label distribution
        labels = [d['label'] for d in bert_data]
        print(f"  Label distribution:")
        for i in range(len(bin_edges) - 1):
            count = labels.count(i)
            pct = 100 * count / len(labels) if labels else 0
            print(f"    Bin {i} ({int(bin_edges[i]):>3}-{int(bin_edges[i+1]):<3}): {count:>6} ({pct:.1f}%)")

    print(f"\nDone! BERT data saved to {output_dir}/")
    print(f"\nNext step: train the model")
    print(f"  python predictor/training/train.py --ds_dir {output_dir} --num_labels {len(bin_edges)-1}")


if __name__ == '__main__':
    main()
