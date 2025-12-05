"""
Forest/Dot-Whisker Plots (Mean + 95% CI)

These plots show point estimates with confidence intervals for comparing
schedulers across metrics.
"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from ..constants import SCHEDULER_ORDER, get_scheduler_color, get_scheduler_marker
from ..utils import save_figure


def plot_forest_metric(agg_data: dict, metric_name: str, output_path: str,
                       title: str = None, ylabel: str = None,
                       formats: List[str] = None):
    """
    Forest plot showing mean +/- 95% CI for each scheduler.

    Args:
        agg_data: Aggregated results dict with schedulers
        metric_name: Name of metric (e.g., 'avg_waiting_time')
        output_path: Path to save figure
        title: Plot title
        ylabel: Y-axis label
        formats: Output formats
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    schedulers = []
    means = []
    ci_lows = []
    ci_highs = []

    for sched in SCHEDULER_ORDER:
        if sched in agg_data['schedulers']:
            data = agg_data['schedulers'][sched].get(metric_name)
            if data:
                schedulers.append(sched)
                means.append(data['mean'])
                ci_lows.append(data['ci_low'])
                ci_highs.append(data['ci_high'])

    if not schedulers:
        print(f"  Warning: No data found for metric '{metric_name}'")
        plt.close()
        return

    y_pos = np.arange(len(schedulers))
    errors_low = [m - cl for m, cl in zip(means, ci_lows)]
    errors_high = [ch - m for m, ch in zip(means, ci_highs)]

    for i, (sched, mean, el, eh) in enumerate(zip(schedulers, means, errors_low, errors_high)):
        color = get_scheduler_color(sched)
        marker = get_scheduler_marker(sched)

        ax.errorbar(
            mean, i,
            xerr=[[el], [eh]],
            fmt=marker,
            color=color,
            markersize=12,
            markerfacecolor=color,
            markeredgecolor='white',
            markeredgewidth=1.5,
            capsize=5,
            capthick=2,
            elinewidth=2,
            label=sched
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(schedulers)
    ax.set_xlabel(ylabel or metric_name.replace('_', ' ').title())

    if title:
        ax.set_title(title, fontweight='bold', pad=10)

    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)

    for i, (mean, el, eh) in enumerate(zip(means, errors_low, errors_high)):
        ax.annotate(
            f'{mean:.2f}',
            xy=(mean, i),
            xytext=(5, 0),
            textcoords='offset points',
            va='center',
            fontsize=9,
            fontweight='bold'
        )

    plt.tight_layout()
    save_figure(fig, output_path, formats)
    plt.close()


def plot_forest_delta_vs_baseline(agg_data: dict, metric_name: str,
                                   baseline: str, output_path: str,
                                   title: str = None,
                                   formats: List[str] = None):
    """
    Forest plot showing improvement (delta) vs baseline scheduler.

    Positive values indicate improvement over baseline.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    baseline_data = agg_data['schedulers'].get(baseline, {}).get(metric_name)
    if not baseline_data:
        print(f"  Warning: No baseline data for '{baseline}'")
        plt.close()
        return

    baseline_mean = baseline_data['mean']

    schedulers = []
    deltas = []
    errors_low = []
    errors_high = []

    for sched in SCHEDULER_ORDER:
        if sched == baseline:
            continue
        if sched in agg_data['schedulers']:
            data = agg_data['schedulers'][sched].get(metric_name)
            if data:
                schedulers.append(sched)
                delta = baseline_mean - data['mean']
                deltas.append(delta)
                ci_range = (data['ci_high'] - data['ci_low']) / 2
                errors_low.append(ci_range)
                errors_high.append(ci_range)

    if not schedulers:
        plt.close()
        return

    y_pos = np.arange(len(schedulers))

    for i, (sched, delta, el, eh) in enumerate(zip(schedulers, deltas, errors_low, errors_high)):
        color = get_scheduler_color(sched)
        marker = get_scheduler_marker(sched)

        ax.errorbar(
            delta, i,
            xerr=[[el], [eh]],
            fmt=marker,
            color=color,
            markersize=12,
            markerfacecolor=color,
            markeredgecolor='white',
            markeredgewidth=1.5,
            capsize=5,
            capthick=2,
            elinewidth=2
        )

    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(schedulers)
    ax.set_xlabel(f'Improvement over {baseline} (seconds)')

    if title:
        ax.set_title(title, fontweight='bold', pad=10)

    for i, delta in enumerate(deltas):
        sign = '+' if delta > 0 else ''
        ax.annotate(
            f'{sign}{delta:.2f}s',
            xy=(delta, i),
            xytext=(5, 0),
            textcoords='offset points',
            va='center',
            fontsize=9,
            fontweight='bold'
        )

    xlim = ax.get_xlim()
    ax.axvspan(0, xlim[1], alpha=0.1, color='green', label='Better than FCFS')
    ax.axvspan(xlim[0], 0, alpha=0.1, color='red', label='Worse than FCFS')
    ax.set_xlim(xlim)

    ax.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    save_figure(fig, output_path, formats)
    plt.close()


def generate_forest_plots(agg_data: dict, output_dir: str, formats: List[str] = None):
    """Generate all forest plots."""
    print("\n[A] Generating Forest Plots...")
    forest_dir = Path(output_dir) / 'forest'

    metrics = [
        ('avg_waiting_time', 'Average Waiting Time (s)', 'Average Waiting Time Comparison'),
        ('p50_waiting_time', 'Median Waiting Time (s)', 'Median (P50) Waiting Time Comparison'),
        ('p99_waiting_time', 'P99 Waiting Time (s)', 'Tail Latency (P99) Comparison'),
        ('jain_index_waiting', 'Jain Fairness Index', 'Fairness Index Comparison'),
        ('avg_turnaround_time', 'Average Turnaround Time (s)', 'Average JCT Comparison'),
    ]

    for metric_name, ylabel, title in metrics:
        output_path = forest_dir / f'forest_{metric_name}'
        plot_forest_metric(agg_data, metric_name, str(output_path),
                          title=title, ylabel=ylabel, formats=formats)

    output_path = forest_dir / 'forest_delta_vs_fcfs'
    plot_forest_delta_vs_baseline(
        agg_data, 'avg_waiting_time', 'FCFS', str(output_path),
        title='Average Waiting Time Improvement vs FCFS',
        formats=formats
    )
