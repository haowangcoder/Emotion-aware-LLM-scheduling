"""
Slopegraph / Dumbbell Plots (Per-Group Analysis)

These plots show how different emotion/valence groups are affected
by each scheduler.
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from ..constants import (
    SCHEDULER_ORDER, AROUSAL_ORDER, VALENCE_ORDER,
    AROUSAL_LABELS, VALENCE_LABELS,
    get_scheduler_color, get_scheduler_marker
)
from ..utils import save_figure


def plot_slopegraph_arousal(agg_data: dict, output_path: str,
                             metric: str = 'avg_waiting_time',
                             title: str = None,
                             formats: List[str] = None):
    """
    Slopegraph showing per-arousal-class metrics.

    Each scheduler is a line connecting low -> medium -> high arousal points.

    Args:
        agg_data: Aggregated results dict
        output_path: Output file path
        metric: Metric to plot within per_emotion_class
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    x_positions = np.arange(len(AROUSAL_ORDER))

    for sched in SCHEDULER_ORDER:
        if sched not in agg_data['schedulers']:
            continue

        sched_data = agg_data['schedulers'][sched]
        per_class = sched_data.get('per_emotion_class', {})

        values = []
        ci_lows = []
        ci_highs = []

        for arousal in AROUSAL_ORDER:
            if arousal in per_class and metric in per_class[arousal]:
                data = per_class[arousal][metric]
                values.append(data['mean'])
                ci_lows.append(data['ci_low'])
                ci_highs.append(data['ci_high'])
            else:
                values.append(np.nan)
                ci_lows.append(np.nan)
                ci_highs.append(np.nan)

        if all(np.isnan(values)):
            continue

        color = get_scheduler_color(sched)
        marker = get_scheduler_marker(sched)

        errors_low = [v - cl for v, cl in zip(values, ci_lows)]
        errors_high = [ch - v for v, ch in zip(values, ci_highs)]

        ax.errorbar(
            x_positions, values,
            yerr=[errors_low, errors_high],
            fmt=f'{marker}-',
            color=color,
            linewidth=2.5,
            markersize=12,
            markerfacecolor=color,
            markeredgecolor='white',
            markeredgewidth=1.5,
            capsize=4,
            capthick=2,
            label=sched,
            alpha=0.9
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([AROUSAL_LABELS[a] for a in AROUSAL_ORDER])
    ax.set_xlabel('Arousal Class')
    ax.set_ylabel(metric.replace('_', ' ').title() + ' (s)')

    if title:
        ax.set_title(title, fontweight='bold', pad=10)

    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.3, len(AROUSAL_ORDER) - 0.7)

    plt.tight_layout()
    save_figure(fig, output_path, formats)
    plt.close()


def plot_slopegraph_valence(summaries: Dict[str, dict], output_path: str,
                             title: str = None,
                             formats: List[str] = None):
    """
    Slopegraph showing per-valence-class waiting time.

    Uses per_valence_values from single-seed summary files.

    Args:
        summaries: Dict mapping scheduler name -> summary dict
        output_path: Output file path
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    x_positions = np.arange(len(VALENCE_ORDER))

    for sched in SCHEDULER_ORDER:
        if sched not in summaries:
            continue

        summary = summaries[sched]
        per_valence = None

        if 'fairness_analysis' in summary:
            fa = summary['fairness_analysis']
            if 'valence_fairness' in fa and 'per_valence_values' in fa['valence_fairness']:
                per_valence = fa['valence_fairness']['per_valence_values']

        if not per_valence:
            continue

        values = [per_valence.get(v, np.nan) for v in VALENCE_ORDER]

        valid_values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
        if len(valid_values) == 0:
            continue

        color = get_scheduler_color(sched)
        marker = get_scheduler_marker(sched)

        ax.plot(
            x_positions, values,
            f'{marker}-',
            color=color,
            linewidth=2.5,
            markersize=12,
            markerfacecolor=color,
            markeredgecolor='white',
            markeredgewidth=1.5,
            label=sched,
            alpha=0.9
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([VALENCE_LABELS[v] for v in VALENCE_ORDER])
    ax.set_xlabel('Valence Class')
    ax.set_ylabel('Average Waiting Time (s)')

    if title:
        ax.set_title(title, fontweight='bold', pad=10)

    ax.legend(loc='best', framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.3, len(VALENCE_ORDER) - 0.7)

    plt.tight_layout()
    save_figure(fig, output_path, formats)
    plt.close()


def plot_dumbbell_arousal(agg_data: dict, output_path: str,
                           metric: str = 'avg_waiting_time',
                           title: str = None,
                           formats: List[str] = None):
    """
    Dumbbell plot comparing low vs high arousal classes.

    Shows the gap between low and high arousal for each scheduler.

    Args:
        agg_data: Aggregated results dict
        output_path: Output file path
        metric: Metric to compare
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(SCHEDULER_ORDER))

    for i, sched in enumerate(SCHEDULER_ORDER):
        if sched not in agg_data['schedulers']:
            continue

        per_class = agg_data['schedulers'][sched].get('per_emotion_class', {})

        low_data = per_class.get('low', {}).get(metric, {})
        high_data = per_class.get('high', {}).get(metric, {})

        if not low_data or not high_data:
            continue

        low_val = low_data['mean']
        high_val = high_data['mean']

        color = get_scheduler_color(sched)

        # Draw connecting line
        ax.plot([low_val, high_val], [i, i], color=color, linewidth=2, alpha=0.5)

        # Draw points
        ax.scatter(low_val, i, s=150, c='#6bcf7f', marker='o',
                  edgecolors='white', linewidths=2, zorder=3, label='Low' if i == 0 else '')
        ax.scatter(high_val, i, s=150, c='#ff6b6b', marker='s',
                  edgecolors='white', linewidths=2, zorder=3, label='High' if i == 0 else '')

        # Add value labels
        ax.annotate(f'{low_val:.1f}', xy=(low_val, i), xytext=(-10, 15),
                   textcoords='offset points', fontsize=9, ha='center', color='#2d8a4e')
        ax.annotate(f'{high_val:.1f}', xy=(high_val, i), xytext=(-10, -20),
                   textcoords='offset points', fontsize=9, ha='center', color='#cc4444')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(SCHEDULER_ORDER)
    ax.set_xlabel(metric.replace('_', ' ').title() + ' (s)')

    if title:
        ax.set_title(title, fontweight='bold', pad=10)

    ax.legend(loc='upper right', title='Arousal Class', framealpha=0.9)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_path, formats)
    plt.close()


def generate_group_analysis_plots(agg_data: dict, summaries: Dict[str, dict],
                                   output_dir: str, formats: List[str] = None):
    """Generate all per-group analysis plots."""
    print("\n[D] Generating Group Analysis Plots...")
    group_dir = Path(output_dir) / 'group_analysis'

    plot_slopegraph_arousal(agg_data, str(group_dir / 'slopegraph_arousal_waiting'),
                            metric='avg_waiting_time',
                            title='Waiting Time by Arousal Class', formats=formats)

    plot_slopegraph_valence(summaries, str(group_dir / 'slopegraph_valence_waiting'),
                            title='Waiting Time by Valence Class', formats=formats)

    plot_dumbbell_arousal(agg_data, str(group_dir / 'dumbbell_arousal_comparison'),
                          metric='avg_waiting_time',
                          title='Low vs High Arousal Waiting Time Comparison', formats=formats)
