#!/usr/bin/env python3
"""
Exp-7: Burst Traffic Analysis (MMPP Arrival Patterns)

Purpose:
- Validate AW-SSJF robustness under realistic bursty traffic
- Compare scheduler performance during burst vs normal periods
- Analyze how burst intensity affects emotional group waiting times

Parameters:
- Burst intensity ratios: 3x (mild), 6.7x (moderate), 15x (severe)
- Schedulers: FCFS, SJF, AW-SSJF(k=2), AW-SSJF(k=4)

Key insights:
- Mild bursts (3x): Minimal stress, scheduling differences small
- Moderate bursts (6.7x): Realistic peak hours vs off-peak
- Severe bursts (15x): Flash crowd / viral event simulation

Usage:
    # First run the bash script to generate data:
    bash scripts/run_exp7_burst.sh

    # Then analyze:
    python experiments/exp7_burst_traffic.py \
        --input_dir results/experiments/exp7_burst \
        --output_dir results/experiments/exp7_burst/plots
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

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


# Burst intensity configurations
BURST_CONFIGS = {
    'mild': {
        'lambda_high': 1.5,
        'lambda_low': 0.5,
        'alpha': 0.2,
        'beta': 0.1,
        'intensity': 3.0,
        'description': 'Normal diurnal variation'
    },
    'moderate': {
        'lambda_high': 2.0,
        'lambda_low': 0.3,
        'alpha': 0.15,
        'beta': 0.05,
        'intensity': 6.7,
        'description': 'Peak hours vs off-peak'
    },
    'severe': {
        'lambda_high': 3.0,
        'lambda_low': 0.2,
        'alpha': 0.1,
        'beta': 0.03,
        'intensity': 15.0,
        'description': 'Flash crowd / viral event'
    },
}

# Scheduler configurations
SCHEDULERS = ['FCFS', 'SJF', 'AW-SSJF_k2', 'AW-SSJF_k4']


def load_burst_sweep_results(input_dir: Path) -> Dict[str, Dict[str, Dict]]:
    """
    Load results for each burst intensity and scheduler combination.

    Args:
        input_dir: Base directory containing mild/, moderate/, severe/

    Returns:
        Nested dict: {burst_name: {scheduler: summary}}
    """
    results = {}

    for burst_name in BURST_CONFIGS.keys():
        burst_dir = input_dir / burst_name
        if burst_dir.exists():
            results[burst_name] = {}

            # Find all scheduler results in this burst directory
            json_files = find_result_files(burst_dir, "*_summary.json")
            for json_file in json_files:
                summary = load_json_summary(json_file)

                # Determine scheduler name
                sched = summary.get('metadata', {}).get('scheduler', 'unknown')
                k = summary.get('metadata', {}).get('weight_exponent')

                if sched == 'AW-SSJF' and k:
                    sched_key = f"AW-SSJF_k{int(k)}"
                else:
                    sched_key = sched

                results[burst_name][sched_key] = summary

    return results


def extract_burst_metrics(
    results: Dict[str, Dict[str, Dict]],
    metric: str
) -> pd.DataFrame:
    """
    Extract a single metric across all burst intensities and schedulers.

    Args:
        results: Nested dict from load_burst_sweep_results
        metric: Metric name to extract

    Returns:
        DataFrame with burst_name as index and schedulers as columns
    """
    data = {}
    burst_order = ['mild', 'moderate', 'severe']

    for burst_name in burst_order:
        if burst_name not in results:
            continue
        data[burst_name] = {}
        for sched, summary in results[burst_name].items():
            run_metrics = summary.get('run_metrics', {})
            overall_metrics = summary.get('overall_metrics', {})

            if metric == 'depression_wait':
                value = extract_quadrant_metrics(summary, 'depression', 'avg_waiting_time')
            elif metric == 'panic_wait':
                value = extract_quadrant_metrics(summary, 'panic', 'avg_waiting_time')
            elif metric == 'excited_wait':
                value = extract_quadrant_metrics(summary, 'excited', 'avg_waiting_time')
            elif metric == 'calm_wait':
                value = extract_quadrant_metrics(summary, 'calm', 'avg_waiting_time')
            else:
                value = run_metrics.get(metric, overall_metrics.get(metric))

            data[burst_name][sched] = value

    return pd.DataFrame(data).T


def plot_burst_impact_on_depression(
    results: Dict[str, Dict[str, Dict]],
    output_path: Path
) -> None:
    """
    Plot Depression waiting time vs burst intensity for each scheduler.

    Args:
        results: Nested dict from load_burst_sweep_results
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    burst_order = ['mild', 'moderate', 'severe']
    intensities = [BURST_CONFIGS[b]['intensity'] for b in burst_order if b in results]
    burst_labels = [b for b in burst_order if b in results]

    colors = {
        'FCFS': '#ff9f43',
        'SJF': '#4a90d9',
        'AW-SSJF_k2': '#6bcf7f',
        'AW-SSJF_k4': '#ff6b6b'
    }
    markers = {'FCFS': 'o', 'SJF': 's', 'AW-SSJF_k2': '^', 'AW-SSJF_k4': 'D'}

    for sched in SCHEDULERS:
        depression_waits = []
        for burst_name in burst_labels:
            if burst_name in results and sched in results[burst_name]:
                wait = extract_quadrant_metrics(
                    results[burst_name][sched], 'depression', 'avg_waiting_time'
                )
                depression_waits.append(wait if wait else 0)
            else:
                depression_waits.append(None)

        if any(w is not None for w in depression_waits):
            ax.plot(intensities, depression_waits, marker=markers.get(sched, 'o'),
                    color=colors.get(sched, 'gray'), linewidth=2, markersize=8,
                    label=sched)

    ax.set_xlabel('Burst Intensity (λ_high / λ_low)', fontsize=12)
    ax.set_ylabel('Depression Avg Waiting Time (s)', fontsize=12)
    ax.set_title('Impact of Burst Intensity on Depression Users\n(Lower is better)',
                 fontsize=14)
    ax.set_xscale('log')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_quadrant_heatmap(
    results: Dict[str, Dict[str, Dict]],
    output_path: Path,
    burst_name: str = 'severe'
) -> None:
    """
    Plot heatmap of waiting times by quadrant and scheduler.

    Args:
        results: Nested dict from load_burst_sweep_results
        output_path: Path to save the plot
        burst_name: Which burst config to use for heatmap
    """
    if burst_name not in results:
        print(f"Warning: {burst_name} burst config not found, skipping heatmap")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    quadrants = ['excited', 'calm', 'panic', 'depression']
    data = []

    for sched in SCHEDULERS:
        row = []
        if sched in results[burst_name]:
            for quad in quadrants:
                wait = extract_quadrant_metrics(
                    results[burst_name][sched], quad, 'avg_waiting_time'
                )
                row.append(wait if wait else 0)
        else:
            row = [0] * len(quadrants)
        data.append(row)

    data = np.array(data)
    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto')

    ax.set_xticks(range(len(quadrants)))
    ax.set_xticklabels([q.capitalize() for q in quadrants])
    ax.set_yticks(range(len(SCHEDULERS)))
    ax.set_yticklabels(SCHEDULERS)

    # Add values
    for i in range(len(SCHEDULERS)):
        for j in range(len(quadrants)):
            color = 'white' if data[i, j] > np.mean(data) else 'black'
            ax.text(j, i, f'{data[i, j]:.1f}', ha='center', va='center',
                    color=color, fontweight='bold')

    plt.colorbar(im, label='Avg Waiting Time (s)')
    intensity = BURST_CONFIGS[burst_name]['intensity']
    ax.set_title(f'Quadrant Waiting Times Under {burst_name.capitalize()} Bursts '
                 f'({intensity}x intensity)', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_tail_latency_comparison(
    results: Dict[str, Dict[str, Dict]],
    output_path: Path
) -> None:
    """
    Compare P99 tail latencies across burst intensities.

    Args:
        results: Nested dict from load_burst_sweep_results
        output_path: Path to save the plot
    """
    burst_order = ['mild', 'moderate', 'severe']
    available_bursts = [b for b in burst_order if b in results]

    fig, axes = plt.subplots(1, len(available_bursts), figsize=(5 * len(available_bursts), 5))
    if len(available_bursts) == 1:
        axes = [axes]

    colors = ['#ff9f43', '#4a90d9', '#6bcf7f', '#ff6b6b']

    for idx, burst_name in enumerate(available_bursts):
        ax = axes[idx]

        scheds = []
        p99_values = []

        for sched in SCHEDULERS:
            if sched in results[burst_name]:
                p99 = results[burst_name][sched].get('run_metrics', {}).get(
                    'p99_waiting_time', 0
                )
                scheds.append(sched)
                p99_values.append(p99)

        bars = ax.bar(scheds, p99_values, color=colors[:len(scheds)])

        ax.set_ylabel('P99 Waiting Time (s)')
        intensity = BURST_CONFIGS[burst_name]['intensity']
        ax.set_title(f'{burst_name.capitalize()} Burst\n(Intensity: {intensity}x)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val in zip(bars, p99_values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_avg_wait_comparison(
    results: Dict[str, Dict[str, Dict]],
    output_path: Path
) -> None:
    """
    Compare average waiting times across burst intensities (grouped bar chart).

    Args:
        results: Nested dict from load_burst_sweep_results
        output_path: Path to save the plot
    """
    burst_order = ['mild', 'moderate', 'severe']
    available_bursts = [b for b in burst_order if b in results]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(available_bursts))
    width = 0.2
    colors = {
        'FCFS': '#ff9f43',
        'SJF': '#4a90d9',
        'AW-SSJF_k2': '#6bcf7f',
        'AW-SSJF_k4': '#ff6b6b'
    }

    for i, sched in enumerate(SCHEDULERS):
        values = []
        for burst_name in available_bursts:
            if sched in results.get(burst_name, {}):
                val = results[burst_name][sched].get('run_metrics', {}).get(
                    'avg_waiting_time', 0
                )
                values.append(val)
            else:
                values.append(0)

        bars = ax.bar(x + i * width, values, width, label=sched,
                      color=colors.get(sched, 'gray'))

        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                        f'{val:.1f}', ha='center', fontsize=8)

    ax.set_xlabel('Burst Intensity')
    ax.set_ylabel('Average Waiting Time (s)')
    ax.set_title('Average Waiting Time by Scheduler and Burst Intensity', fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f'{b.capitalize()}\n({BURST_CONFIGS[b]["intensity"]}x)'
                        for b in available_bursts])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_depression_speedup_vs_burst(
    results: Dict[str, Dict[str, Dict]],
    output_path: Path
) -> None:
    """
    Plot Depression speedup (vs SJF) at each burst intensity.

    Args:
        results: Nested dict from load_burst_sweep_results
        output_path: Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    burst_order = ['mild', 'moderate', 'severe']
    available_bursts = [b for b in burst_order if b in results]
    intensities = [BURST_CONFIGS[b]['intensity'] for b in available_bursts]

    colors = {'AW-SSJF_k2': '#6bcf7f', 'AW-SSJF_k4': '#ff6b6b'}

    for sched in ['AW-SSJF_k2', 'AW-SSJF_k4']:
        speedups = []
        for burst_name in available_bursts:
            sjf_wait = extract_quadrant_metrics(
                results.get(burst_name, {}).get('SJF', {}),
                'depression', 'avg_waiting_time'
            )
            awssjf_wait = extract_quadrant_metrics(
                results.get(burst_name, {}).get(sched, {}),
                'depression', 'avg_waiting_time'
            )

            if sjf_wait and awssjf_wait and awssjf_wait > 0:
                speedups.append(sjf_wait / awssjf_wait)
            else:
                speedups.append(None)

        if any(s is not None for s in speedups):
            ax.plot(intensities, speedups, marker='o', color=colors.get(sched),
                    linewidth=2, markersize=8, label=f'{sched} speedup')

    # Reference line at 1.0 (no speedup)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)

    ax.set_xlabel('Burst Intensity (λ_high / λ_low)', fontsize=12)
    ax.set_ylabel('Depression Speedup (vs SJF)', fontsize=12)
    ax.set_title('Depression User Speedup Under Bursty Traffic\n(Higher is better)',
                 fontsize=14)
    ax.set_xscale('log')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_burst_report(
    results: Dict[str, Dict[str, Dict]],
    output_path: Path
) -> Dict:
    """
    Generate JSON report of burst traffic experiment.

    Args:
        results: Nested dict from load_burst_sweep_results
        output_path: Path to save the report

    Returns:
        Report dictionary
    """
    report = {
        'experiment': 'exp7_burst_traffic',
        'purpose': 'Validate AW-SSJF robustness under bursty MMPP traffic',
        'burst_configurations': BURST_CONFIGS,
        'results_by_burst': {},
        'findings': {}
    }

    for burst_name, schedulers in results.items():
        report['results_by_burst'][burst_name] = {}
        for sched, summary in schedulers.items():
            run_metrics = summary.get('run_metrics', {})
            report['results_by_burst'][burst_name][sched] = {
                'avg_waiting_time': run_metrics.get('avg_waiting_time'),
                'p99_waiting_time': run_metrics.get('p99_waiting_time'),
                'depression_wait': extract_quadrant_metrics(
                    summary, 'depression', 'avg_waiting_time'
                ),
                'panic_wait': extract_quadrant_metrics(
                    summary, 'panic', 'avg_waiting_time'
                ),
                'aw_jct': run_metrics.get('aw_jct'),
            }

    # Compute key findings
    if 'severe' in results:
        severe = results['severe']
        if 'SJF' in severe and 'AW-SSJF_k4' in severe:
            sjf_dep = extract_quadrant_metrics(
                severe['SJF'], 'depression', 'avg_waiting_time'
            ) or 0
            awssjf_dep = extract_quadrant_metrics(
                severe['AW-SSJF_k4'], 'depression', 'avg_waiting_time'
            ) or 1
            report['findings']['depression_speedup_severe'] = (
                sjf_dep / awssjf_dep if awssjf_dep > 0 else 0
            )

    # Compare mild vs severe degradation
    if 'mild' in results and 'severe' in results:
        for sched in SCHEDULERS:
            if sched in results['mild'] and sched in results['severe']:
                mild_wait = results['mild'][sched].get('run_metrics', {}).get(
                    'avg_waiting_time', 0
                )
                severe_wait = results['severe'][sched].get('run_metrics', {}).get(
                    'avg_waiting_time', 0
                )
                if mild_wait > 0:
                    report['findings'][f'{sched}_degradation_ratio'] = (
                        severe_wait / mild_wait
                    )

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=float)

    print(f"Saved report: {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Exp-7: Burst Traffic Analysis"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="results/experiments/exp7_burst",
        help="Directory containing burst traffic results",
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

    print(f"Loading burst traffic results from: {input_dir}")
    results = load_burst_sweep_results(input_dir)

    if not results:
        print(f"Error: No results found in {input_dir}")
        print("Expected subdirectories: mild/, moderate/, severe/")
        sys.exit(1)

    print(f"Found burst configs: {list(results.keys())}")
    for burst_name, scheds in results.items():
        intensity = BURST_CONFIGS[burst_name]['intensity']
        print(f"  {burst_name} ({intensity}x): {list(scheds.keys())}")

    # Generate plots
    print("\n=== Generating Plots ===")
    plot_burst_impact_on_depression(results, output_dir / "depression_vs_burst.png")
    plot_quadrant_heatmap(results, output_dir / "quadrant_heatmap_severe.png", 'severe')
    plot_tail_latency_comparison(results, output_dir / "tail_latency_comparison.png")
    plot_avg_wait_comparison(results, output_dir / "avg_wait_comparison.png")
    plot_depression_speedup_vs_burst(results, output_dir / "depression_speedup_vs_burst.png")

    # Generate report
    print("\n=== Generating Report ===")
    report = generate_burst_report(results, output_dir / "exp7_report.json")

    print(f"\n=== Summary ===")
    print(f"Burst configs analyzed: {list(results.keys())}")
    print(f"Output directory: {output_dir}")

    # Print key findings
    if report.get('findings'):
        print("\nKey Findings:")
        for key, val in report['findings'].items():
            print(f"  {key}: {val:.2f}")


if __name__ == "__main__":
    main()
