"""
Pareto Tradeoff Plots

These plots visualize the fairness-efficiency tradeoff between schedulers.
"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

from ..constants import SCHEDULER_ORDER, get_scheduler_color, get_scheduler_marker
from ..utils import save_figure


def plot_pareto_tradeoff(agg_data: dict, x_metric: str, y_metric: str,
                          output_path: str, title: str = None,
                          xlabel: str = None, ylabel: str = None,
                          formats: List[str] = None):
    """
    Scatter plot showing fairness-efficiency tradeoff.

    Each scheduler is a point with bidirectional error bars (95% CI).

    Args:
        agg_data: Aggregated results dict
        x_metric: X-axis metric (e.g., 'jain_index_waiting')
        y_metric: Y-axis metric (e.g., 'avg_waiting_time')
        output_path: Output file path
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    for sched in SCHEDULER_ORDER:
        if sched not in agg_data['schedulers']:
            continue

        sched_data = agg_data['schedulers'][sched]
        x_data = sched_data.get(x_metric)
        y_data = sched_data.get(y_metric)

        if not x_data or not y_data:
            continue

        color = get_scheduler_color(sched)
        marker = get_scheduler_marker(sched)

        ax.errorbar(
            x_data['mean'], y_data['mean'],
            xerr=[[x_data['mean'] - x_data['ci_low']],
                  [x_data['ci_high'] - x_data['mean']]],
            yerr=[[y_data['mean'] - y_data['ci_low']],
                  [y_data['ci_high'] - y_data['mean']]],
            fmt=marker,
            color=color,
            markersize=15,
            markerfacecolor=color,
            markeredgecolor='white',
            markeredgewidth=2,
            capsize=6,
            capthick=2,
            elinewidth=2,
            label=sched,
            alpha=0.9
        )

        ax.annotate(
            sched,
            xy=(x_data['mean'], y_data['mean']),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            color=color,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor=color, alpha=0.8)
        )

    ax.set_xlabel(xlabel or x_metric.replace('_', ' ').title())
    ax.set_ylabel(ylabel or y_metric.replace('_', ' ').title())

    if title:
        ax.set_title(title, fontweight='bold', pad=10)

    ax.annotate(
        'Ideal\n(High Fairness,\nLow Latency)',
        xy=(0.95, 0.05),
        xycoords='axes fraction',
        fontsize=9,
        color='green',
        alpha=0.7,
        ha='right',
        va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3)
    )

    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure(fig, output_path, formats)
    plt.close()


def generate_pareto_plots(agg_data: dict, output_dir: str, formats: List[str] = None):
    """Generate all Pareto tradeoff plots."""
    print("\n[C] Generating Pareto Tradeoff Plots...")
    tradeoff_dir = Path(output_dir) / 'tradeoff'

    plot_pareto_tradeoff(
        agg_data, 'jain_index_waiting', 'avg_waiting_time',
        str(tradeoff_dir / 'pareto_jain_vs_avg_waiting'),
        title='Fairness-Efficiency Tradeoff',
        xlabel='Jain Fairness Index (Higher = Fairer)',
        ylabel='Average Waiting Time (s) (Lower = Better)',
        formats=formats
    )

    plot_pareto_tradeoff(
        agg_data, 'jain_index_waiting', 'p99_waiting_time',
        str(tradeoff_dir / 'pareto_jain_vs_p99_waiting'),
        title='Fairness vs Tail Risk Tradeoff',
        xlabel='Jain Fairness Index (Higher = Fairer)',
        ylabel='P99 Waiting Time (s) (Lower = Better)',
        formats=formats
    )
