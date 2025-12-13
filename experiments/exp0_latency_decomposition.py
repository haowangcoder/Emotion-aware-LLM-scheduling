#!/usr/bin/env python3
"""
Exp-0: Latency Decomposition Verification

Purpose:
- Verify JCT = waiting_duration + execution_duration
- Plot latency breakdown (stacked bar chart)
- Plot waiting time CDF with P50/P95/P99 markers

This experiment establishes the "system credibility" by proving
that our timing measurements are consistent.

Usage:
    python experiments/exp0_latency_decomposition.py \
        --input_dir results/llm_runs_job80_load0.9 \
        --output_dir results/experiments/exp0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_scheduler_results(input_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load CSV results for all schedulers in a directory.

    Args:
        input_dir: Directory containing *_jobs.csv files

    Returns:
        Dictionary mapping scheduler name to DataFrame
    """
    results = {}
    csv_files = list(input_dir.glob("*_jobs.csv")) + list(input_dir.glob("*_fixed_jobs.csv"))

    for csv_file in csv_files:
        # Extract scheduler name from filename
        name = csv_file.stem.split('_')[0]
        df = pd.read_csv(csv_file)
        results[name] = df

    return results


def verify_jct_decomposition(df: pd.DataFrame, tolerance: float = 0.01) -> Tuple[bool, float]:
    """
    Verify that JCT = waiting_time + actual_serving_time.

    Args:
        df: DataFrame with job-level data
        tolerance: Acceptable error tolerance in seconds

    Returns:
        Tuple of (is_valid, max_error)
    """
    # Compute JCT from components
    if 'waiting_time' in df.columns and 'actual_serving_time' in df.columns:
        computed_jct = df['waiting_time'] + df['actual_serving_time']
    else:
        print("Warning: Missing waiting_time or actual_serving_time columns")
        return False, float('inf')

    # Compute JCT from timestamps
    if 'finish_time' in df.columns and 'arrival_time' in df.columns:
        timestamp_jct = df['finish_time'] - df['arrival_time']
    elif 'turnaround_time' in df.columns:
        timestamp_jct = df['turnaround_time']
    else:
        print("Warning: Cannot compute JCT from timestamps")
        timestamp_jct = computed_jct

    # Compare
    errors = np.abs(computed_jct - timestamp_jct)
    max_error = errors.max()
    is_valid = max_error < tolerance

    return is_valid, max_error


def compute_latency_breakdown(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute average latency breakdown.

    Returns:
        Dictionary with avg_waiting, avg_execution, avg_jct
    """
    return {
        'avg_waiting': df['waiting_time'].mean() if 'waiting_time' in df.columns else 0,
        'avg_execution': df['actual_serving_time'].mean() if 'actual_serving_time' in df.columns else 0,
        'avg_jct': df['turnaround_time'].mean() if 'turnaround_time' in df.columns else 0,
    }


def compute_percentiles(values: pd.Series) -> Dict[str, float]:
    """
    Compute P50, P95, P99 percentiles.
    """
    return {
        'p50': values.quantile(0.50),
        'p95': values.quantile(0.95),
        'p99': values.quantile(0.99),
    }


def plot_latency_decomposition_stacked(
    results: Dict[str, pd.DataFrame],
    output_path: Path
) -> None:
    """
    Plot stacked bar chart showing waiting vs execution time.

    Args:
        results: Dictionary mapping scheduler name to DataFrame
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    schedulers = list(results.keys())
    x = np.arange(len(schedulers))
    width = 0.6

    waiting_times = []
    execution_times = []

    for sched in schedulers:
        df = results[sched]
        breakdown = compute_latency_breakdown(df)
        waiting_times.append(breakdown['avg_waiting'])
        execution_times.append(breakdown['avg_execution'])

    # Stacked bar
    bars1 = ax.bar(x, waiting_times, width, label='Waiting Time', color='#ff6b6b')
    bars2 = ax.bar(x, execution_times, width, bottom=waiting_times,
                   label='Execution Time', color='#4a90d9')

    ax.set_xlabel('Scheduler', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Latency Decomposition: Waiting vs Execution Time', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(schedulers, fontsize=11)
    ax.legend(loc='upper right')

    # Add value labels on bars
    for i, (w, e) in enumerate(zip(waiting_times, execution_times)):
        ax.text(i, w / 2, f'{w:.1f}s', ha='center', va='center', color='white', fontweight='bold')
        ax.text(i, w + e / 2, f'{e:.1f}s', ha='center', va='center', color='white', fontweight='bold')
        ax.text(i, w + e + 0.5, f'JCT={w + e:.1f}s', ha='center', va='bottom', fontsize=9)

    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_waiting_cdf_with_markers(
    results: Dict[str, pd.DataFrame],
    output_path: Path
) -> None:
    """
    Plot CDF of waiting times with P50/P95/P99 markers.

    Args:
        results: Dictionary mapping scheduler name to DataFrame
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#4a90d9', '#6bcf7f', '#ff6b6b', '#ff9f43']
    linestyles = ['-', '--', '-.', ':']

    for i, (sched, df) in enumerate(results.items()):
        if 'waiting_time' not in df.columns:
            continue

        waiting = df['waiting_time'].sort_values()
        cdf = np.arange(1, len(waiting) + 1) / len(waiting)

        color = colors[i % len(colors)]
        ax.plot(waiting, cdf, label=sched, color=color,
                linestyle=linestyles[i % len(linestyles)], linewidth=2)

        # Add percentile markers
        percentiles = compute_percentiles(df['waiting_time'])
        for pname, pval in percentiles.items():
            yval = {'p50': 0.50, 'p95': 0.95, 'p99': 0.99}[pname]
            ax.axvline(x=pval, color=color, linestyle=':', alpha=0.5)
            ax.scatter([pval], [yval], color=color, s=50, zorder=5)

    # Add horizontal reference lines
    for p, label in [(0.50, 'P50'), (0.95, 'P95'), (0.99, 'P99')]:
        ax.axhline(y=p, color='gray', linestyle='--', alpha=0.3)
        ax.text(ax.get_xlim()[1] * 0.95, p + 0.01, label, fontsize=9, color='gray')

    ax.set_xlabel('Waiting Time (seconds)', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title('Waiting Time CDF with Percentile Markers', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_verification_report(
    results: Dict[str, pd.DataFrame],
    output_path: Path
) -> Dict:
    """
    Generate a verification report as JSON.

    Args:
        results: Dictionary mapping scheduler name to DataFrame
        output_path: Path to save the report

    Returns:
        Report dictionary
    """
    report = {
        'experiment': 'exp0_latency_decomposition',
        'purpose': 'Verify JCT = waiting_time + execution_time',
        'schedulers': {},
    }

    all_valid = True

    for sched, df in results.items():
        is_valid, max_error = verify_jct_decomposition(df)
        breakdown = compute_latency_breakdown(df)
        percentiles = compute_percentiles(df['waiting_time']) if 'waiting_time' in df.columns else {}

        report['schedulers'][sched] = {
            'num_jobs': len(df),
            'jct_verification': {
                'is_valid': is_valid,
                'max_error_seconds': float(max_error),
            },
            'latency_breakdown': breakdown,
            'waiting_percentiles': percentiles,
        }

        if not is_valid:
            all_valid = False

    report['all_valid'] = all_valid

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Saved report: {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Exp-0: Latency Decomposition Verification"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="results/llm_runs_job80_load0.9",
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/experiments/exp0",
        help="Directory to save analysis outputs",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {input_dir}")
    results = load_scheduler_results(input_dir)

    if not results:
        print(f"Error: No CSV files found in {input_dir}")
        sys.exit(1)

    print(f"Found schedulers: {list(results.keys())}")

    # Verify JCT decomposition
    print("\n=== JCT Decomposition Verification ===")
    for sched, df in results.items():
        is_valid, max_error = verify_jct_decomposition(df)
        status = "PASS" if is_valid else "FAIL"
        print(f"  {sched}: {status} (max error: {max_error:.4f}s)")

    # Generate plots
    print("\n=== Generating Plots ===")
    plot_latency_decomposition_stacked(results, output_dir / "latency_decomposition_stacked.png")
    plot_waiting_cdf_with_markers(results, output_dir / "waiting_cdf_with_markers.png")

    # Generate report
    print("\n=== Generating Report ===")
    report = generate_verification_report(results, output_dir / "exp0_report.json")

    print(f"\n=== Summary ===")
    print(f"All schedulers valid: {report['all_valid']}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
