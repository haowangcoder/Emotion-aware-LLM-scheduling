#!/usr/bin/env python3
"""
Exp-1: k (weight_exponent) Sweep - Core Static Profiling

Purpose:
- Demonstrate controllable trade-off via k parameter
- Show k-sweep curve (efficiency vs fairness trade-off)
- Generate Pareto scatter plot

This is the most "course-like" experiment showing:
  k↑ → Depression QoE↑ (shorter wait) → Overall efficiency may decrease

Parameters:
- k (weight_exponent) ∈ {1, 2, 3, 4}
- Fixed: w_max=2.0, gamma_panic=0.3, load=0.9

Usage:
    # First run the bash script to generate data:
    bash scripts/run_exp1_k_sweep.sh

    # Then analyze:
    python experiments/exp1_k_sweep.py \
        --input_dir results/experiments/exp1_k_sweep \
        --output_dir results/experiments/exp1_k_sweep/plots
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.utils.result_aggregator import (
    load_json_summary,
    load_csv_results,
    find_result_files,
    extract_quadrant_metrics,
)


# Default k values for sweep
K_VALUES = [1, 2, 3, 4]

# Metrics to extract
METRICS = [
    'avg_waiting_time',
    'p95_waiting_time',
    'p99_waiting_time',
    'aw_jct',
]


def load_k_sweep_results(input_dir: Path, k_values: List[int] = K_VALUES) -> Dict[int, Dict]:
    """
    Load results for each k value from sweep directories.

    Args:
        input_dir: Base directory containing k1/, k2/, k3/, k4/ subdirectories
        k_values: List of k values to load

    Returns:
        Dictionary mapping k value to result summary
    """
    results = {}

    for k in k_values:
        # Try different directory naming patterns
        patterns = [f"k{k}", f"k_{k}", f"k={k}"]
        for pattern in patterns:
            subdir = input_dir / pattern
            if subdir.exists():
                json_files = find_result_files(subdir, "*_summary.json")
                if json_files:
                    results[k] = load_json_summary(json_files[0])
                    # Also try to load CSV for detailed analysis
                    csv_files = list(subdir.glob("*_jobs.csv")) + list(subdir.glob("*_fixed_jobs.csv"))
                    if csv_files:
                        results[k]['_csv_data'] = pd.read_csv(csv_files[0])
                    break

    return results


def extract_sweep_metrics(results: Dict[int, Dict]) -> pd.DataFrame:
    """
    Extract key metrics from sweep results into a DataFrame.

    Args:
        results: Dictionary mapping k to result summaries

    Returns:
        DataFrame with k as index and metrics as columns
    """
    rows = []

    for k, summary in sorted(results.items()):
        row = {'k': k}

        # Overall metrics
        run_metrics = summary.get('run_metrics', {})
        overall_metrics = summary.get('overall_metrics', {})

        row['avg_waiting_time'] = run_metrics.get('avg_waiting_time', overall_metrics.get('avg_waiting_time'))
        row['p95_waiting_time'] = run_metrics.get('p95_waiting_time', overall_metrics.get('p95_waiting_time'))
        row['p99_waiting_time'] = run_metrics.get('p99_waiting_time', overall_metrics.get('p99_waiting_time'))
        row['aw_jct'] = run_metrics.get('aw_jct', overall_metrics.get('aw_jct'))

        # Depression-specific metrics
        row['depression_wait'] = extract_quadrant_metrics(summary, 'depression', 'avg_waiting_time')
        row['depression_p99'] = extract_quadrant_metrics(summary, 'depression', 'p99_waiting_time')

        # Fairness metrics
        fairness = summary.get('fairness_analysis', {}).get('waiting_time_fairness', {})
        row['jain_index'] = fairness.get('jain_index')
        row['depression_vs_others'] = fairness.get('depression_vs_others')

        rows.append(row)

    return pd.DataFrame(rows).set_index('k')


def plot_k_tradeoff_curves(
    metrics_df: pd.DataFrame,
    output_path: Path,
    baseline_sjf: Optional[Dict] = None,
    baseline_fcfs: Optional[Dict] = None
) -> None:
    """
    Plot trade-off curves: k vs efficiency and k vs fairness.

    Args:
        metrics_df: DataFrame with metrics indexed by k
        output_path: Path to save the plot
        baseline_sjf: Optional SJF baseline metrics
        baseline_fcfs: Optional FCFS baseline metrics
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    k_values = metrics_df.index.tolist()

    # Plot 1: Efficiency (Avg Wait, P99 Wait)
    ax1 = axes[0]
    ax1.plot(k_values, metrics_df['avg_waiting_time'], 'o-', label='Avg Wait', color='#4a90d9', linewidth=2, markersize=8)
    ax1.plot(k_values, metrics_df['p99_waiting_time'], 's--', label='P99 Wait', color='#ff6b6b', linewidth=2, markersize=8)

    # Add baselines
    if baseline_sjf:
        ax1.axhline(y=baseline_sjf.get('avg_waiting_time'), color='#4a90d9', linestyle=':', alpha=0.7, label='SJF (Avg)')
        ax1.axhline(y=baseline_sjf.get('p99_waiting_time'), color='#ff6b6b', linestyle=':', alpha=0.7, label='SJF (P99)')

    ax1.set_xlabel('k (weight_exponent)', fontsize=12)
    ax1.set_ylabel('Waiting Time (seconds)', fontsize=12)
    ax1.set_title('Efficiency vs k\n(Lower is better)', fontsize=13)
    ax1.set_xticks(k_values)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Annotate k=2 as recommended
    if 2 in k_values:
        idx = k_values.index(2)
        ax1.annotate('Recommended', xy=(2, metrics_df.loc[2, 'avg_waiting_time']),
                     xytext=(2.3, metrics_df.loc[2, 'avg_waiting_time'] * 1.1),
                     arrowprops=dict(arrowstyle='->', color='green'),
                     fontsize=10, color='green')

    # Plot 2: Fairness (Depression Wait, AW-JCT)
    ax2 = axes[1]
    ax2.plot(k_values, metrics_df['depression_wait'], 'o-', label='Depression Wait', color='#ff6b6b', linewidth=2, markersize=8)
    ax2.plot(k_values, metrics_df['aw_jct'], 's--', label='AW-JCT', color='#6bcf7f', linewidth=2, markersize=8)

    # Add baselines
    if baseline_fcfs:
        ax2.axhline(y=baseline_fcfs.get('depression_wait'), color='#ff6b6b', linestyle=':', alpha=0.7, label='FCFS (Depression)')

    ax2.set_xlabel('k (weight_exponent)', fontsize=12)
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.set_title('Fairness (QoE) vs k\n(Lower is better for Depression)', fontsize=13)
    ax2.set_xticks(k_values)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Add annotation
    ax2.text(0.02, 0.98, 'k↑ → More emotion priority',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_pareto_scatter(
    metrics_df: pd.DataFrame,
    output_path: Path,
    baseline_points: Optional[Dict[str, Dict]] = None
) -> None:
    """
    Plot Pareto scatter: Efficiency (Avg Wait) vs Fairness (Depression Wait).

    Args:
        metrics_df: DataFrame with metrics indexed by k
        output_path: Path to save the plot
        baseline_points: Optional dict of baseline results {name: {avg_wait, depression_wait}}
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot AW-SSJF points for different k
    k_values = metrics_df.index.tolist()
    x = metrics_df['avg_waiting_time'].values
    y = metrics_df['depression_wait'].values

    # Color gradient based on k
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(k_values)))

    scatter = ax.scatter(x, y, c=colors, s=200, edgecolors='black', linewidth=2, zorder=5)

    # Add k labels
    for i, k in enumerate(k_values):
        ax.annotate(f'k={k}', (x[i], y[i]), textcoords='offset points',
                    xytext=(10, 10), fontsize=11, fontweight='bold')

    # Connect points with line
    ax.plot(x, y, 'k--', alpha=0.5, linewidth=1, zorder=1)

    # Add baseline points
    if baseline_points:
        markers = {'FCFS': '^', 'SJF': 'v', 'Weight-Only': 'D'}
        for name, metrics in baseline_points.items():
            if 'avg_waiting_time' in metrics and 'depression_wait' in metrics:
                ax.scatter(metrics['avg_waiting_time'], metrics['depression_wait'],
                           marker=markers.get(name, 'x'), s=150, label=name,
                           edgecolors='black', linewidth=1.5, zorder=4)

    # Ideal point annotation (lower-left corner, using axes transform for stable positioning)
    ax.text(0.02, 0.02, 'Ideal\n(low wait, low depression wait)',
            transform=ax.transAxes, fontsize=9, color='green', alpha=0.8,
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='green', alpha=0.7))

    ax.set_xlabel('Average Waiting Time (seconds) - Efficiency', fontsize=12)
    ax.set_ylabel('Depression Waiting Time (seconds) - Fairness', fontsize=12)
    ax.set_title('Pareto Trade-off: Efficiency vs Fairness\n(AW-SSJF with different k values)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_jain_index_vs_k(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """
    Plot Jain Fairness Index vs k.

    Args:
        metrics_df: DataFrame with metrics indexed by k
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    k_values = metrics_df.index.tolist()
    jain_values = metrics_df['jain_index'].values

    ax.plot(k_values, jain_values, 'o-', color='#9b59b6', linewidth=2, markersize=10)
    ax.fill_between(k_values, jain_values, alpha=0.3, color='#9b59b6')

    ax.set_xlabel('k (weight_exponent)', fontsize=12)
    ax.set_ylabel('Jain Fairness Index', fontsize=12)
    ax.set_title('Global Fairness vs k\n(Jain Index: 1.0 = perfect equality)', fontsize=13)
    ax.set_xticks(k_values)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.text(0.98, 0.02, 'Lower Jain Index = More inequality\n(intentional for Depression-First)',
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_sweep_report(
    metrics_df: pd.DataFrame,
    output_path: Path
) -> Dict:
    """
    Generate a JSON report of the k-sweep results.

    Args:
        metrics_df: DataFrame with metrics indexed by k
        output_path: Path to save the report

    Returns:
        Report dictionary
    """
    report = {
        'experiment': 'exp1_k_sweep',
        'purpose': 'Demonstrate controllable trade-off via weight_exponent (k)',
        'parameters': {
            'k_values': metrics_df.index.tolist(),
            'fixed': {
                'w_max': 2.0,
                'gamma_panic': 0.3,
                'system_load': 0.9,
            }
        },
        'results': metrics_df.to_dict(orient='index'),
        'findings': {
            'best_efficiency_k': int(metrics_df['avg_waiting_time'].idxmin()),
            'best_depression_k': int(metrics_df['depression_wait'].idxmin()),
            'best_aw_jct_k': int(metrics_df['aw_jct'].idxmin()),
            'recommended_k': 2,  # Balance point
        }
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=float)

    print(f"Saved report: {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Exp-1: k (weight_exponent) Sweep Analysis"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="results/experiments/exp1_k_sweep",
        help="Directory containing k-sweep results (k1/, k2/, k3/, k4/ subdirs)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save plots and report (default: input_dir/plots)",
    )
    parser.add_argument(
        "--baseline_dir",
        type=str,
        default=None,
        help="Optional directory containing SJF/FCFS baselines for comparison",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading k-sweep results from: {input_dir}")
    results = load_k_sweep_results(input_dir)

    if not results:
        print(f"Error: No results found in {input_dir}")
        print("Expected subdirectories: k1/, k2/, k3/, k4/")
        sys.exit(1)

    print(f"Found k values: {sorted(results.keys())}")

    # Extract metrics
    metrics_df = extract_sweep_metrics(results)
    print("\n=== Extracted Metrics ===")
    print(metrics_df.to_string())

    # Load baselines if available
    baseline_points = None
    if args.baseline_dir:
        baseline_dir = Path(args.baseline_dir)
        baseline_points = {}
        for sched in ['FCFS', 'SJF', 'Weight-Only']:
            json_files = list(baseline_dir.glob(f"{sched}*_summary.json"))
            if json_files:
                summary = load_json_summary(json_files[0])
                baseline_points[sched] = {
                    'avg_waiting_time': summary.get('run_metrics', {}).get('avg_waiting_time'),
                    'depression_wait': extract_quadrant_metrics(summary, 'depression', 'avg_waiting_time'),
                }

    # Generate plots
    print("\n=== Generating Plots ===")
    plot_k_tradeoff_curves(metrics_df, output_dir / "k_tradeoff_curves.png")
    plot_pareto_scatter(metrics_df, output_dir / "pareto_scatter.png", baseline_points)
    plot_jain_index_vs_k(metrics_df, output_dir / "jain_index_vs_k.png")

    # Generate report
    print("\n=== Generating Report ===")
    report = generate_sweep_report(metrics_df, output_dir / "exp1_report.json")

    print(f"\n=== Summary ===")
    print(f"Best k for efficiency (lowest avg wait): k={report['findings']['best_efficiency_k']}")
    print(f"Best k for fairness (lowest depression wait): k={report['findings']['best_depression_k']}")
    print(f"Recommended k (balance): k={report['findings']['recommended_k']}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
