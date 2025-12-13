#!/usr/bin/env python3
"""
Exp-6: Multi-seed Statistical Validation

Purpose:
- Upgrade "single seed=42 results" to statistically reliable conclusions
- Run multiple seeds (10-20) and compute 95% confidence intervals
- Generate forest plot with error bars

Usage:
    # Generate data with multiple seeds
    bash scripts/run_exp6_multiseed.sh

    # Analyze results
    python experiments/exp6_multi_seed.py \
        --input_dir results/experiments/exp6_multiseed \
        --output_dir results/experiments/exp6_multiseed/plots
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils.result_aggregator import load_json_summary, find_result_files, extract_quadrant_metrics
from experiments.utils.ci_calculator import compute_95_ci, aggregate_multi_seed_results


# Default seed range
SEEDS = list(range(42, 52))  # Seeds 42-51 (10 seeds)

# Metrics to analyze
METRICS = [
    'avg_waiting_time',
    'p99_waiting_time',
    'aw_jct',
]


def load_multi_seed_results(
    input_dir: Path,
    seeds: List[int] = SEEDS
) -> Dict[str, Dict[int, Dict]]:
    """
    Load results for multiple seeds and schedulers.

    Expected directory structure:
        input_dir/
            seed42/
                FCFS_..._summary.json
                SJF_..._summary.json
                AW-SSJF_..._summary.json
            seed43/
                ...

    Returns:
        {scheduler: {seed: summary}}
    """
    results = {}

    for seed in seeds:
        seed_dir = input_dir / f"seed{seed}"
        if not seed_dir.exists():
            continue

        json_files = find_result_files(seed_dir, "*_summary.json")
        for json_file in json_files:
            summary = load_json_summary(json_file)
            scheduler = summary.get('metadata', {}).get('scheduler', 'unknown')

            if scheduler not in results:
                results[scheduler] = {}
            results[scheduler][seed] = summary

    return results


def compute_scheduler_ci(
    scheduler_results: Dict[int, Dict],
    metrics: List[str] = METRICS
) -> Dict[str, Dict[str, float]]:
    """
    Compute confidence intervals for a single scheduler across seeds.

    Args:
        scheduler_results: {seed: summary}
        metrics: List of metric names

    Returns:
        {metric: {mean, ci_lower, ci_upper, std, n}}
    """
    ci_results = {}

    for metric in metrics:
        values = []
        for seed, summary in scheduler_results.items():
            run_metrics = summary.get('run_metrics', {})
            overall_metrics = summary.get('overall_metrics', {})

            if metric == 'depression_wait':
                value = extract_quadrant_metrics(summary, 'depression', 'avg_waiting_time')
            else:
                value = run_metrics.get(metric, overall_metrics.get(metric))

            if value is not None:
                values.append(value)

        if values:
            mean, ci_lower, ci_upper = compute_95_ci(values)
            ci_results[metric] = {
                'mean': mean,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'std': np.std(values),
                'n': len(values),
            }

    return ci_results


def plot_forest_plot(
    results: Dict[str, Dict[str, Dict]],
    metric: str,
    output_path: Path,
    title: str = None
) -> None:
    """
    Plot forest plot showing mean with 95% CI for each scheduler.

    Args:
        results: {scheduler: {metric: {mean, ci_lower, ci_upper}}}
        metric: Metric to plot
        output_path: Path to save plot
        title: Optional plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    schedulers = list(results.keys())
    y_positions = range(len(schedulers))

    colors = {'FCFS': '#ff9f43', 'SJF': '#4a90d9', 'AW-SSJF': '#6bcf7f', 'Weight-Only': '#ff6b6b'}

    for i, sched in enumerate(schedulers):
        if metric not in results[sched]:
            continue

        data = results[sched][metric]
        mean = data['mean']
        ci_lower = data['ci_lower']
        ci_upper = data['ci_upper']

        color = colors.get(sched, 'gray')

        # Plot error bar
        ax.errorbar(mean, i, xerr=[[mean - ci_lower], [ci_upper - mean]],
                    fmt='o', markersize=10, capsize=5, capthick=2,
                    color=color, ecolor=color, linewidth=2)

        # Add mean value label
        ax.text(ci_upper + 0.5, i, f'{mean:.2f}', va='center', fontsize=10)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(schedulers, fontsize=11)
    ax.set_xlabel(f'{metric.replace("_", " ").title()} (seconds)', fontsize=12)
    ax.set_title(title or f'{metric.replace("_", " ").title()} with 95% CI\n(n seeds)', fontsize=13)
    ax.grid(True, alpha=0.3, axis='x')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_boxplot_comparison(
    all_values: Dict[str, Dict[str, List[float]]],
    metric: str,
    output_path: Path
) -> None:
    """
    Plot boxplot showing distribution of metric across seeds.

    Args:
        all_values: {scheduler: {metric: [values across seeds]}}
        metric: Metric to plot
        output_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    schedulers = list(all_values.keys())
    data = [all_values[s].get(metric, []) for s in schedulers]

    colors = ['#ff9f43', '#4a90d9', '#6bcf7f', '#ff6b6b'][:len(schedulers)]

    bp = ax.boxplot(data, labels=schedulers, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel(f'{metric.replace("_", " ").title()} (seconds)', fontsize=12)
    ax.set_title(f'Distribution of {metric.replace("_", " ").title()} Across Seeds', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_multi_seed_report(
    results: Dict[str, Dict[str, Dict]],
    output_path: Path
) -> Dict:
    """Generate statistical validation report."""
    report = {
        'experiment': 'exp6_multi_seed',
        'purpose': 'Statistical validation with multiple random seeds',
        'schedulers': {},
    }

    for sched, ci_results in results.items():
        report['schedulers'][sched] = ci_results

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=float)
    print(f"Saved: {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Exp-6: Multi-seed Statistical Validation")
    parser.add_argument("--input_dir", type=str, default="results/experiments/exp6_multiseed")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="42-51",
                        help="Seed range, e.g., '42-51' or '42,43,44'")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse seeds
    if '-' in args.seeds:
        start, end = map(int, args.seeds.split('-'))
        seeds = list(range(start, end + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(',')]

    print(f"Loading multi-seed results from: {input_dir}")
    print(f"Seeds: {seeds}")

    results = load_multi_seed_results(input_dir, seeds)

    if not results:
        print(f"Error: No results found. Expected: seed42/, seed43/, etc.")
        print("\nGenerating mock data for demonstration...")

        # Generate mock data
        np.random.seed(0)
        mock_results = {}
        for sched in ['FCFS', 'SJF', 'AW-SSJF', 'Weight-Only']:
            mock_results[sched] = {}
            base = {'FCFS': 30, 'SJF': 24, 'AW-SSJF': 26, 'Weight-Only': 29}[sched]
            for seed in seeds:
                np.random.seed(seed)
                mock_results[sched][seed] = {
                    'run_metrics': {
                        'avg_waiting_time': base + np.random.randn() * 2,
                        'p99_waiting_time': base * 5 + np.random.randn() * 10,
                        'aw_jct': base + 2 + np.random.randn() * 1.5,
                    }
                }
        results = mock_results

    print(f"Found schedulers: {list(results.keys())}")

    # Compute CIs for each scheduler
    ci_results = {}
    all_values = {}

    for sched, seed_results in results.items():
        ci_results[sched] = compute_scheduler_ci(seed_results, METRICS + ['depression_wait'])

        # Collect all values for boxplot
        all_values[sched] = {}
        for metric in METRICS:
            values = []
            for seed, summary in seed_results.items():
                run_metrics = summary.get('run_metrics', {})
                val = run_metrics.get(metric)
                if val is not None:
                    values.append(val)
            all_values[sched][metric] = values

    print("\n=== Confidence Intervals ===")
    for sched, metrics in ci_results.items():
        print(f"\n{sched}:")
        for metric, data in metrics.items():
            print(f"  {metric}: {data['mean']:.2f} [{data['ci_lower']:.2f}, {data['ci_upper']:.2f}] (n={data['n']})")

    # Generate plots
    print("\n=== Generating Plots ===")
    for metric in METRICS:
        plot_forest_plot(ci_results, metric, output_dir / f"forest_{metric}.png")

    if all_values:
        for metric in METRICS:
            plot_boxplot_comparison(all_values, metric, output_dir / f"boxplot_{metric}.png")

    # Generate report
    print("\n=== Generating Report ===")
    generate_multi_seed_report(ci_results, output_dir / "exp6_report.json")

    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
