#!/usr/bin/env python3
"""
Aggregate Results from Multi-Seed Experiments

This script aggregates results from multiple random seeds and calculates
statistical summaries including mean, std, and 95% confidence intervals.

Usage:
    python analysis/aggregate_results.py --input-dir results/multi_seed_runs --output-file results/aggregated.json
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import scipy.stats as stats


def load_summary_files(input_dir: Path) -> Dict[str, List[Dict]]:
    """
    Load all summary JSON files from seed subdirectories.

    Args:
        input_dir: Base directory containing seed_* subdirectories

    Returns:
        Dictionary mapping scheduler name to list of result dicts
    """
    results_by_scheduler = defaultdict(list)

    # Find all seed directories
    seed_dirs = sorted(input_dir.glob("seed_*"))

    if not seed_dirs:
        print(f"Warning: No seed directories found in {input_dir}")
        return {}

    print(f"Found {len(seed_dirs)} seed directories")

    for seed_dir in seed_dirs:
        # Find all summary files in this seed directory
        summary_files = list(seed_dir.glob("*_summary.json"))

        for summary_file in summary_files:
            with open(summary_file, 'r') as f:
                data = json.load(f)

            # Extract scheduler name from filename or metadata
            scheduler = data.get('metadata', {}).get('scheduler', 'unknown')
            results_by_scheduler[scheduler].append(data)

    return dict(results_by_scheduler)


def calculate_confidence_interval(values: List[float], confidence: float = 0.95) -> Dict:
    """
    Calculate mean, std, and confidence interval for a list of values.

    Args:
        values: List of numeric values
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Dictionary with statistics
    """
    if not values or len(values) == 0:
        return {
            'mean': None,
            'std': None,
            'ci_low': None,
            'ci_high': None,
            'n': 0
        }

    values = np.array(values)
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1) if n > 1 else 0

    # Calculate confidence interval using t-distribution
    if n > 1:
        sem = std / np.sqrt(n)
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin = t_value * sem
        ci_low = mean - margin
        ci_high = mean + margin
    else:
        ci_low = mean
        ci_high = mean

    return {
        'mean': float(mean),
        'std': float(std),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
        'n': n
    }


def aggregate_metric(results: List[Dict], metric_path: List[str]) -> Dict:
    """
    Aggregate a specific metric across multiple results.

    Args:
        results: List of result dictionaries
        metric_path: Path to the metric (e.g., ['overall_metrics', 'avg_waiting_time'])

    Returns:
        Aggregated statistics for the metric
    """
    values = []
    for result in results:
        # Navigate to the metric
        value = result
        try:
            for key in metric_path:
                value = value[key]
            if value is not None:
                values.append(float(value))
        except (KeyError, TypeError):
            continue

    return calculate_confidence_interval(values)


def aggregate_scheduler_results(results: List[Dict]) -> Dict:
    """
    Aggregate all metrics for a single scheduler across multiple seeds.

    Args:
        results: List of result dictionaries from different seeds

    Returns:
        Aggregated statistics for all metrics
    """
    # Define metrics to aggregate
    metrics_to_aggregate = {
        # Overall metrics
        'avg_waiting_time': ['overall_metrics', 'avg_waiting_time'],
        'std_waiting_time': ['overall_metrics', 'std_waiting_time'],
        'p50_waiting_time': ['overall_metrics', 'p50_waiting_time'],
        'p95_waiting_time': ['overall_metrics', 'p95_waiting_time'],
        'p99_waiting_time': ['overall_metrics', 'p99_waiting_time'],
        'avg_turnaround_time': ['overall_metrics', 'avg_turnaround_time'],
        'throughput': ['overall_metrics', 'throughput'],

        # Fairness metrics
        'jain_index_waiting': ['fairness_analysis', 'waiting_time_fairness', 'jain_index'],
        'cv_waiting': ['fairness_analysis', 'waiting_time_fairness', 'coefficient_of_variation'],
        'max_min_ratio_waiting': ['fairness_analysis', 'waiting_time_fairness', 'max_min_ratio'],

        # Run metrics
        'total_time': ['run_metrics', 'total_time'],
        'effective_load': ['run_metrics', 'effective_load'],
    }

    aggregated = {}
    for metric_name, metric_path in metrics_to_aggregate.items():
        aggregated[metric_name] = aggregate_metric(results, metric_path)

    # Aggregate per-class metrics
    per_class_metrics = {}
    for emotion_class in ['low', 'medium', 'high']:
        per_class_metrics[emotion_class] = {
            'avg_waiting_time': aggregate_metric(
                results,
                ['per_emotion_class_metrics', emotion_class, 'avg_waiting_time']
            ),
            'avg_turnaround_time': aggregate_metric(
                results,
                ['per_emotion_class_metrics', emotion_class, 'avg_turnaround_time']
            ),
            'avg_predicted_service_time': aggregate_metric(
                results,
                ['per_emotion_class_metrics', emotion_class, 'avg_predicted_service_time']
            ),
        }

    aggregated['per_emotion_class'] = per_class_metrics

    return aggregated


def main():
    parser = argparse.ArgumentParser(description='Aggregate multi-seed experiment results')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing seed_* subdirectories')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Output JSON file for aggregated results')
    parser.add_argument('--confidence', type=float, default=0.95,
                        help='Confidence level for CI (default: 0.95)')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    print(f"Loading results from {input_dir}...")
    results_by_scheduler = load_summary_files(input_dir)

    if not results_by_scheduler:
        print("Error: No results found")
        return

    # Aggregate results for each scheduler
    aggregated_results = {
        'metadata': {
            'input_dir': str(input_dir),
            'confidence_level': args.confidence,
        },
        'schedulers': {}
    }

    for scheduler, results in results_by_scheduler.items():
        print(f"Aggregating {len(results)} results for {scheduler}...")
        aggregated_results['schedulers'][scheduler] = aggregate_scheduler_results(results)
        aggregated_results['schedulers'][scheduler]['num_seeds'] = len(results)

    # Save aggregated results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(aggregated_results, f, indent=2)

    print(f"\nAggregated results saved to {output_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("Summary (Mean ± 95% CI)")
    print("=" * 70)

    for scheduler, data in aggregated_results['schedulers'].items():
        print(f"\n{scheduler} (n={data['num_seeds']}):")
        if 'avg_waiting_time' in data and data['avg_waiting_time']['mean'] is not None:
            wt = data['avg_waiting_time']
            print(f"  Avg Waiting Time: {wt['mean']:.3f} ± [{wt['ci_low']:.3f}, {wt['ci_high']:.3f}]")
        if 'jain_index_waiting' in data and data['jain_index_waiting']['mean'] is not None:
            jain = data['jain_index_waiting']
            print(f"  Jain Index:       {jain['mean']:.4f} ± [{jain['ci_low']:.4f}, {jain['ci_high']:.4f}]")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
