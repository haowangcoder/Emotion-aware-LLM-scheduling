#!/usr/bin/env python3
"""
Exp-2: Load Sweep (ρ Scan)

Purpose:
- Demonstrate that results hold across different load levels
- Show how AW-SSJF behaves from low load (ρ=0.5) to saturation (ρ≥1.0)
- Compare SJF, AW-SSJF(k=2), AW-SSJF(k=4) at each load level

Parameters:
- system_load (ρ) ∈ {0.5, 0.7, 0.9, 1.0, 1.2}
- Schedulers: SJF, AW-SSJF(k=2), AW-SSJF(k=4)

Key insights:
- ρ=0.5: Low load, scheduling differences minimal
- ρ=0.9: Current experiment point, differences visible
- ρ≥1.0: System saturation, AW-SSJF differences amplified

Usage:
    # First run the bash script to generate data:
    bash scripts/run_exp2_load_sweep.sh

    # Then analyze:
    python experiments/exp2_load_sweep.py \
        --input_dir results/experiments/exp2_load_sweep \
        --output_dir results/experiments/exp2_load_sweep/plots
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
    find_result_files,
    extract_quadrant_metrics,
)


# Default load values for sweep
LOAD_VALUES = [0.5, 0.7, 0.9, 1.0, 1.2]

# Scheduler configurations
SCHEDULERS = ['SJF', 'AW-SSJF_k2', 'AW-SSJF_k4']


def load_load_sweep_results(
    input_dir: Path,
    load_values: List[float] = LOAD_VALUES
) -> Dict[float, Dict[str, Dict]]:
    """
    Load results for each load value and scheduler combination.

    Args:
        input_dir: Base directory containing load0.5/, load0.7/, etc.
        load_values: List of load values to search for

    Returns:
        Nested dict: {load: {scheduler: summary}}
    """
    results = {}

    for load in load_values:
        # Try different directory naming patterns
        patterns = [f"load{load}", f"load_{load}", f"load={load}", f"rho{load}"]
        for pattern in patterns:
            load_dir = input_dir / pattern
            if load_dir.exists():
                results[load] = {}

                # Find all scheduler results in this load directory
                json_files = find_result_files(load_dir, "*_summary.json")
                for json_file in json_files:
                    summary = load_json_summary(json_file)

                    # Determine scheduler name
                    sched = summary.get('metadata', {}).get('scheduler', 'unknown')
                    k = summary.get('metadata', {}).get('weight_exponent')

                    if sched == 'AW-SSJF' and k:
                        sched_key = f"AW-SSJF_k{int(k)}"
                    else:
                        sched_key = sched

                    results[load][sched_key] = summary
                break

    return results


def extract_load_sweep_metrics(
    results: Dict[float, Dict[str, Dict]],
    metric: str
) -> pd.DataFrame:
    """
    Extract a single metric across all loads and schedulers.

    Args:
        results: Nested dict from load_load_sweep_results
        metric: Metric name to extract

    Returns:
        DataFrame with load as index and schedulers as columns
    """
    data = {}

    for load, schedulers in sorted(results.items()):
        data[load] = {}
        for sched, summary in schedulers.items():
            run_metrics = summary.get('run_metrics', {})
            overall_metrics = summary.get('overall_metrics', {})

            if metric == 'depression_wait':
                value = extract_quadrant_metrics(summary, 'depression', 'avg_waiting_time')
            else:
                value = run_metrics.get(metric, overall_metrics.get(metric))

            data[load][sched] = value

    return pd.DataFrame(data).T


def plot_performance_vs_load(
    results: Dict[float, Dict[str, Dict]],
    output_path: Path,
    metric: str = 'avg_waiting_time',
    ylabel: str = 'Average Waiting Time (seconds)',
    title: str = 'Performance vs Load'
) -> None:
    """
    Plot performance metric vs load for all schedulers.

    Args:
        results: Nested dict from load_load_sweep_results
        output_path: Path to save the plot
        metric: Metric to plot
        ylabel: Y-axis label
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    df = extract_load_sweep_metrics(results, metric)
    loads = df.index.tolist()

    colors = {'SJF': '#4a90d9', 'AW-SSJF_k2': '#6bcf7f', 'AW-SSJF_k4': '#ff6b6b'}
    markers = {'SJF': 'o', 'AW-SSJF_k2': 's', 'AW-SSJF_k4': '^'}
    linestyles = {'SJF': '-', 'AW-SSJF_k2': '--', 'AW-SSJF_k4': '-.'}

    for sched in df.columns:
        if sched in df.columns:
            ax.plot(loads, df[sched], marker=markers.get(sched, 'o'),
                    color=colors.get(sched, 'gray'),
                    linestyle=linestyles.get(sched, '-'),
                    linewidth=2, markersize=8, label=sched)

    # Mark saturation point at ρ=1.0
    ax.axvline(x=1.0, color='red', linestyle=':', alpha=0.7, linewidth=2)
    ax.text(1.02, ax.get_ylim()[1] * 0.95, 'Saturation\n(ρ=1.0)',
            fontsize=9, color='red', alpha=0.8)

    ax.set_xlabel('System Load (ρ)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(loads)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_depression_vs_load(
    results: Dict[float, Dict[str, Dict]],
    output_path: Path
) -> None:
    """
    Plot Depression waiting time vs load for all schedulers.

    Args:
        results: Nested dict from load_load_sweep_results
        output_path: Path to save the plot
    """
    plot_performance_vs_load(
        results, output_path,
        metric='depression_wait',
        ylabel='Depression Quadrant Avg Wait (seconds)',
        title='Depression User Wait Time vs Load\n(Lower is better for fairness)'
    )


def plot_tail_latency_vs_load(
    results: Dict[float, Dict[str, Dict]],
    output_path: Path
) -> None:
    """
    Plot P99 tail latency vs load.

    Args:
        results: Nested dict from load_load_sweep_results
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    df_p95 = extract_load_sweep_metrics(results, 'p95_waiting_time')
    df_p99 = extract_load_sweep_metrics(results, 'p99_waiting_time')
    loads = df_p99.index.tolist()

    colors = {'SJF': '#4a90d9', 'AW-SSJF_k2': '#6bcf7f', 'AW-SSJF_k4': '#ff6b6b'}

    for sched in df_p99.columns:
        color = colors.get(sched, 'gray')
        ax.plot(loads, df_p99[sched], marker='o', color=color,
                linestyle='-', linewidth=2, markersize=8, label=f'{sched} (P99)')
        ax.plot(loads, df_p95[sched], marker='s', color=color,
                linestyle='--', linewidth=1.5, markersize=6, alpha=0.6)

    # Mark saturation point
    ax.axvline(x=1.0, color='red', linestyle=':', alpha=0.7, linewidth=2)

    ax.set_xlabel('System Load (ρ)', fontsize=12)
    ax.set_ylabel('Tail Latency (seconds)', fontsize=12)
    ax.set_title('Tail Latency (P99) vs Load\n(Solid=P99, Dashed=P95)', fontsize=14)
    ax.set_xticks(loads)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_speedup_vs_load(
    results: Dict[float, Dict[str, Dict]],
    output_path: Path
) -> None:
    """
    Plot Depression speedup (vs SJF) at each load level.

    Args:
        results: Nested dict from load_load_sweep_results
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    df = extract_load_sweep_metrics(results, 'depression_wait')
    loads = df.index.tolist()

    colors = {'AW-SSJF_k2': '#6bcf7f', 'AW-SSJF_k4': '#ff6b6b'}

    for sched in ['AW-SSJF_k2', 'AW-SSJF_k4']:
        if sched in df.columns and 'SJF' in df.columns:
            # Speedup = SJF time / AW-SSJF time
            speedup = df['SJF'] / df[sched]
            ax.plot(loads, speedup, marker='o', color=colors.get(sched),
                    linewidth=2, markersize=8, label=f'{sched} speedup')

    # Reference line at 1.0 (no speedup)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)

    # Mark saturation point
    ax.axvline(x=1.0, color='red', linestyle=':', alpha=0.7)

    ax.set_xlabel('System Load (ρ)', fontsize=12)
    ax.set_ylabel('Depression Speedup (vs SJF)', fontsize=12)
    ax.set_title('Depression User Speedup vs Load\n(Higher is better)', fontsize=14)
    ax.set_xticks(loads)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_load_sweep_report(
    results: Dict[float, Dict[str, Dict]],
    output_path: Path
) -> Dict:
    """
    Generate a JSON report of the load sweep results.

    Args:
        results: Nested dict from load_load_sweep_results
        output_path: Path to save the report

    Returns:
        Report dictionary
    """
    report = {
        'experiment': 'exp2_load_sweep',
        'purpose': 'Demonstrate results across different load levels',
        'parameters': {
            'load_values': list(results.keys()),
            'schedulers': list(next(iter(results.values())).keys()) if results else [],
        },
        'results_by_load': {},
    }

    for load, schedulers in sorted(results.items()):
        report['results_by_load'][str(load)] = {}
        for sched, summary in schedulers.items():
            run_metrics = summary.get('run_metrics', {})
            report['results_by_load'][str(load)][sched] = {
                'avg_waiting_time': run_metrics.get('avg_waiting_time'),
                'p99_waiting_time': run_metrics.get('p99_waiting_time'),
                'depression_wait': extract_quadrant_metrics(summary, 'depression', 'avg_waiting_time'),
            }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=float)

    print(f"Saved report: {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Exp-2: Load Sweep Analysis"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="results/experiments/exp2_load_sweep",
        help="Directory containing load sweep results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save plots and report (default: input_dir/plots)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading load sweep results from: {input_dir}")
    results = load_load_sweep_results(input_dir)

    if not results:
        print(f"Error: No results found in {input_dir}")
        print("Expected subdirectories: load0.5/, load0.7/, load0.9/, load1.0/, load1.2/")
        sys.exit(1)

    print(f"Found loads: {sorted(results.keys())}")
    for load, scheds in sorted(results.items()):
        print(f"  ρ={load}: {list(scheds.keys())}")

    # Generate plots
    print("\n=== Generating Plots ===")
    plot_performance_vs_load(results, output_dir / "avg_wait_vs_load.png")
    plot_depression_vs_load(results, output_dir / "depression_wait_vs_load.png")
    plot_tail_latency_vs_load(results, output_dir / "tail_latency_vs_load.png")
    plot_speedup_vs_load(results, output_dir / "depression_speedup_vs_load.png")

    # Generate report
    print("\n=== Generating Report ===")
    report = generate_load_sweep_report(results, output_dir / "exp2_report.json")

    print(f"\n=== Summary ===")
    print(f"Loads analyzed: {sorted(results.keys())}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
