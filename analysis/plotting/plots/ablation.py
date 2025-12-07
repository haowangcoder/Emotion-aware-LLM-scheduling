"""
Ablation Study Plots

These plots visualize ablation experiments like shuffle tests
to validate that observed effects are genuine.
"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from ..utils import save_figure


def plot_shuffle_comparison(results: dict, output_path: str,
                            formats: List[str] = None):
    """
    Plot comparison bar chart for shuffle experiment.

    Shows side-by-side bars for Original vs Shuffled conditions for each scheduler.

    Args:
        results: Output from load_shuffle_experiment_results()
        output_path: Output file path (without extension)
        formats: Output formats (default: ['pdf', 'png'])
    """
    schedulers = list(results['original'].keys())
    n_schedulers = len(schedulers)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    metrics = [
        ('avg_wait', 'Average Waiting Time (s)', 'lower is better'),
        ('p99', 'P99 Waiting Time (s)', 'lower is better'),
        ('jain', 'Jain Fairness Index', 'higher is better')
    ]

    x = np.arange(n_schedulers)
    width = 0.35

    for ax, (metric, label, direction) in zip(axes, metrics):
        original_vals = [results['original'].get(s, {}).get(metric, 0) for s in schedulers]
        shuffled_vals = [results['shuffled'].get(s, {}).get(metric, 0) for s in schedulers]

        bars1 = ax.bar(x - width/2, original_vals, width, label='Original',
                       color='#0072B2', alpha=0.8)
        bars2 = ax.bar(x + width/2, shuffled_vals, width, label='Shuffled',
                       color='#E69F00', alpha=0.8, hatch='//')

        ax.set_xlabel('Scheduler', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'{label}\n({direction})', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(schedulers, rotation=15, ha='right')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            fmt = '.3f' if metric == 'jain' else '.1f'
            ax.annotate(f'{height:{fmt}}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            fmt = '.3f' if metric == 'jain' else '.1f'
            ax.annotate(f'{height:{fmt}}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    save_figure(fig, output_path, formats)
    plt.close()


def plot_shuffle_delta(results: dict, output_path: str,
                       formats: List[str] = None):
    """
    Plot delta (Shuffled - Original) as diverging bar chart.

    Positive delta for avg_wait/p99 means shuffling hurt performance (good!).
    Negative delta for jain means shuffling hurt fairness.

    Args:
        results: Output from load_shuffle_experiment_results()
        output_path: Output file path (without extension)
        formats: Output formats (default: ['pdf', 'png'])
    """
    schedulers = [s for s in results['original'].keys() if s in results['shuffled']]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate deltas (shuffled - original) for avg_wait
    # Positive = shuffled is worse = emotion correlation was helping
    deltas = []
    colors = []
    for sched in schedulers:
        orig = results['original'][sched]['avg_wait']
        shuf = results['shuffled'][sched]['avg_wait']
        delta = shuf - orig
        deltas.append(delta)
        # Green if positive (shuffle hurt = correlation was real)
        # Red if negative (shuffle helped = concerning)
        colors.append('#009E73' if delta > 0 else '#D55E00')

    y = np.arange(len(schedulers))
    bars = ax.barh(y, deltas, color=colors, alpha=0.8, edgecolor='black')

    ax.axvline(0, color='black', linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(schedulers, fontsize=11)
    ax.set_xlabel('Δ Average Waiting Time (Shuffled - Original) [seconds]', fontsize=11)
    ax.set_title('Impact of Breaking Emotion→Length Correlation\n'
                 '(Positive = shuffling hurt performance = correlation was real)',
                 fontsize=12, fontweight='bold')

    # Add value annotations
    for bar, delta in zip(bars, deltas):
        width = bar.get_width()
        ax.annotate(f'{delta:+.2f}s',
                   xy=(width, bar.get_y() + bar.get_height()/2),
                   xytext=(5 if width >= 0 else -5, 0),
                   textcoords='offset points',
                   ha='left' if width >= 0 else 'right',
                   va='center', fontsize=10, fontweight='bold')

    # Add interpretation legend
    ax.annotate('← Shuffle helped\n(unexpected)',
               xy=(0.02, 0.02), xycoords='axes fraction',
               fontsize=9, color='#D55E00', ha='left', va='bottom')
    ax.annotate('Shuffle hurt →\n(validates hypothesis)',
               xy=(0.98, 0.02), xycoords='axes fraction',
               fontsize=9, color='#009E73', ha='right', va='bottom')

    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save_figure(fig, output_path, formats)
    plt.close()


def generate_shuffle_experiment_plots(results: dict, output_dir: str,
                                       formats: List[str] = None):
    """Generate plots for shuffle experiment analysis."""
    print("\n[Shuffle] Generating Shuffle Experiment Plots...")
    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Comparison bar chart
    plot_shuffle_comparison(
        results,
        str(plot_dir / 'shuffle_comparison'),
        formats=formats
    )
    print(f"  Saved: shuffle_comparison")

    # Delta diverging bar chart
    plot_shuffle_delta(
        results,
        str(plot_dir / 'shuffle_delta'),
        formats=formats
    )
    print(f"  Saved: shuffle_delta")
