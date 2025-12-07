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


def plot_robustness_heatmap(results: dict, metric: str, output_path: str,
                            formats: List[str] = None):
    """
    Plot 2x2 heatmap for robustness experiment results.

    Shows scheduler performance across 4 conditions:
    - Rows: Distribution (Uniform vs Real)
    - Columns: Arrival Pattern (Poisson vs Bursty)

    Args:
        results: Output from load_robustness_experiment_results()
        metric: Metric to plot ('avg_wait', 'p99', 'jain')
        output_path: Output file path (without extension)
        formats: Output formats (default: ['pdf', 'png'])
    """
    from ..constants import get_scheduler_color

    conditions = ['uniform_poisson', 'real_poisson', 'uniform_bursty', 'real_bursty']
    schedulers = ['FCFS', 'SSJF-Emotion', 'SSJF-Combined']

    # Reshape into 2x2 grid labels
    dist_labels = ['Uniform', 'Real']
    arrival_labels = ['Poisson', 'Bursty']

    metric_labels = {
        'avg_wait': 'Average Waiting Time (s)',
        'p99': 'P99 Waiting Time (s)',
        'jain': 'Jain Fairness Index'
    }

    fig, axes = plt.subplots(1, len(schedulers), figsize=(14, 5))

    for ax, sched in zip(axes, schedulers):
        # Build 2x2 data matrix
        data = np.full((2, 2), np.nan)
        for i, dist in enumerate(['uniform', 'real']):
            for j, arr in enumerate(['poisson', 'bursty']):
                cond = f'{dist}_{arr}'
                if cond in results and sched in results[cond]:
                    data[i, j] = results[cond][sched][metric]

        # Choose colormap based on metric direction
        if metric == 'jain':
            cmap = 'RdYlGn'  # Green = good (high fairness)
        else:
            cmap = 'RdYlGn_r'  # Green = good (low wait time)

        im = ax.imshow(data, cmap=cmap, aspect='auto')

        # Set ticks
        ax.set_xticks(range(2))
        ax.set_xticklabels(arrival_labels)
        ax.set_yticks(range(2))
        ax.set_yticklabels(dist_labels)

        ax.set_xlabel('Arrival Pattern', fontsize=11)
        ax.set_ylabel('Emotion Distribution', fontsize=11)
        ax.set_title(sched, fontsize=12, fontweight='bold',
                     color=get_scheduler_color(sched))

        # Annotate cells
        for i in range(2):
            for j in range(2):
                val = data[i, j]
                if not np.isnan(val):
                    text = f'{val:.3f}' if metric == 'jain' else f'{val:.1f}'
                    ax.text(j, i, text, ha='center', va='center',
                           fontsize=11, fontweight='bold',
                           color='white' if val > np.nanmean(data) else 'black')

        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f'Robustness Analysis: {metric_labels.get(metric, metric)}',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    save_figure(fig, output_path, formats)
    plt.close()


def plot_robustness_grouped_bar(results: dict, output_path: str,
                                 formats: List[str] = None):
    """
    Plot grouped bar chart comparing schedulers across all conditions.

    Args:
        results: Output from load_robustness_experiment_results()
        output_path: Output file path (without extension)
        formats: Output formats (default: ['pdf', 'png'])
    """
    from ..constants import get_scheduler_color

    conditions = ['uniform_poisson', 'real_poisson', 'uniform_bursty', 'real_bursty']
    condition_labels = ['Uniform\n+Poisson', 'Real\n+Poisson',
                        'Uniform\n+Bursty', 'Real\n+Bursty']
    schedulers = ['FCFS', 'SSJF-Emotion', 'SSJF-Combined']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = [
        ('avg_wait', 'Average Waiting Time (s)', 'lower is better'),
        ('p99', 'P99 Waiting Time (s)', 'lower is better'),
        ('jain', 'Jain Fairness Index', 'higher is better')
    ]

    x = np.arange(len(conditions))
    width = 0.25

    for ax, (metric, label, direction) in zip(axes, metrics):
        for i, sched in enumerate(schedulers):
            values = []
            for cond in conditions:
                if cond in results and sched in results[cond]:
                    values.append(results[cond][sched][metric])
                else:
                    values.append(0)

            color = get_scheduler_color(sched)
            bars = ax.bar(x + i * width, values, width, label=sched,
                         color=color, alpha=0.8)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                fmt = '.3f' if metric == 'jain' else '.1f'
                ax.annotate(f'{height:{fmt}}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=7, rotation=45)

        ax.set_xlabel('Condition', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'{label}\n({direction})', fontsize=12, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(condition_labels, fontsize=9)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_path, formats)
    plt.close()


def plot_robustness_delta(results: dict, output_path: str,
                          formats: List[str] = None):
    """
    Plot delta from baseline (uniform_poisson) for each condition.

    Shows how much performance degrades when moving to more challenging conditions.

    Args:
        results: Output from load_robustness_experiment_results()
        output_path: Output file path (without extension)
        formats: Output formats (default: ['pdf', 'png'])
    """
    from ..constants import get_scheduler_color

    baseline = 'uniform_poisson'
    compare_conditions = ['real_poisson', 'uniform_bursty', 'real_bursty']
    condition_labels = ['Real\nDistribution', 'Bursty\nArrivals', 'Real + Bursty\n(Worst Case)']
    schedulers = ['FCFS', 'SSJF-Emotion', 'SSJF-Combined']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Avg wait delta
    ax = axes[0]
    x = np.arange(len(compare_conditions))
    width = 0.25

    for i, sched in enumerate(schedulers):
        if sched not in results.get(baseline, {}):
            continue
        base_val = results[baseline][sched]['avg_wait']

        deltas = []
        for cond in compare_conditions:
            if cond in results and sched in results[cond]:
                deltas.append(results[cond][sched]['avg_wait'] - base_val)
            else:
                deltas.append(0)

        color = get_scheduler_color(sched)
        ax.bar(x + i * width, deltas, width, label=sched, color=color, alpha=0.8)

    ax.axhline(0, color='black', linewidth=1, linestyle='-')
    ax.set_xlabel('Condition vs Baseline', fontsize=11)
    ax.set_ylabel('Δ Average Wait Time (s)', fontsize=11)
    ax.set_title('Impact on Average Wait Time\n(vs uniform+poisson baseline)',
                fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(condition_labels, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # P99 delta
    ax = axes[1]
    for i, sched in enumerate(schedulers):
        if sched not in results.get(baseline, {}):
            continue
        base_val = results[baseline][sched]['p99']

        deltas = []
        for cond in compare_conditions:
            if cond in results and sched in results[cond]:
                deltas.append(results[cond][sched]['p99'] - base_val)
            else:
                deltas.append(0)

        color = get_scheduler_color(sched)
        ax.bar(x + i * width, deltas, width, label=sched, color=color, alpha=0.8)

    ax.axhline(0, color='black', linewidth=1, linestyle='-')
    ax.set_xlabel('Condition vs Baseline', fontsize=11)
    ax.set_ylabel('Δ P99 Wait Time (s)', fontsize=11)
    ax.set_title('Impact on P99 Tail Latency\n(vs uniform+poisson baseline)',
                fontsize=12, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(condition_labels, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_path, formats)
    plt.close()


def generate_robustness_experiment_plots(results: dict, output_dir: str,
                                          formats: List[str] = None):
    """Generate plots for workload robustness experiment analysis."""
    print("\n[Robustness] Generating Robustness Experiment Plots...")
    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Grouped bar chart (main comparison)
    plot_robustness_grouped_bar(
        results,
        str(plot_dir / 'robustness_grouped_bar'),
        formats=formats
    )
    print(f"  Saved: robustness_grouped_bar")

    # Heatmaps for each metric
    for metric in ['avg_wait', 'p99', 'jain']:
        plot_robustness_heatmap(
            results, metric,
            str(plot_dir / f'robustness_heatmap_{metric}'),
            formats=formats
        )
        print(f"  Saved: robustness_heatmap_{metric}")

    # Delta from baseline
    plot_robustness_delta(
        results,
        str(plot_dir / 'robustness_delta'),
        formats=formats
    )
    print(f"  Saved: robustness_delta")
