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
        xy=(0.05, 0.95),
        xycoords='axes fraction',
        fontsize=9,
        color='green',
        alpha=0.7,
        ha='left',
        va='top',
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


def plot_starvation_pareto(sweep_data: dict, output_path: str,
                            formats: List[str] = None):
    """
    Plot Pareto tradeoff for starvation coefficient sweep.

    Each scheduler is shown as a trajectory line with arrows indicating
    the direction of increasing coefficient. Points are labeled with
    their coefficient values.

    Args:
        sweep_data: Output from load_starvation_sweep_results()
        output_path: Output file path (without extension)
        formats: Output formats (default: ['pdf', 'png'])
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    coefficients = sweep_data['coefficients']

    # Plot each scheduler's trajectory
    for sched, coeff_data in sweep_data['schedulers'].items():
        color = get_scheduler_color(sched)
        marker = get_scheduler_marker(sched)

        # Sort by coefficient to get correct trajectory order
        sorted_coeffs = sorted(coeff_data.keys())
        p99_vals = [coeff_data[c]['p99'] for c in sorted_coeffs]
        jain_vals = [coeff_data[c]['jain'] for c in sorted_coeffs]

        # Plot trajectory line with arrows
        for i in range(len(sorted_coeffs) - 1):
            ax.annotate(
                '',
                xy=(p99_vals[i + 1], jain_vals[i + 1]),
                xytext=(p99_vals[i], jain_vals[i]),
                arrowprops=dict(
                    arrowstyle='->',
                    color=color,
                    lw=2,
                    alpha=0.6
                )
            )

        # Plot points
        ax.scatter(
            p99_vals, jain_vals,
            c=color,
            marker=marker,
            s=150,
            edgecolors='white',
            linewidths=2,
            label=sched,
            zorder=5
        )

        # Label each point with coefficient value
        for c, p99, jain in zip(sorted_coeffs, p99_vals, jain_vals):
            ax.annotate(
                f'c={c}',
                xy=(p99, jain),
                xytext=(8, -12),
                textcoords='offset points',
                fontsize=9,
                fontweight='bold',
                color=color,
                bbox=dict(
                    boxstyle='round,pad=0.2',
                    facecolor='white',
                    edgecolor=color,
                    alpha=0.8
                )
            )

    # Plot FCFS baseline as reference (draw first, below other points)
    baseline = sweep_data.get('baseline')
    if baseline:
        # Draw a larger circle behind as highlight
        ax.scatter(
            baseline['p99'], baseline['jain'],
            c='none',
            marker='o',
            s=400,
            edgecolors='#4D4D4D',
            linewidths=3,
            zorder=3,
            alpha=0.5
        )
        ax.scatter(
            baseline['p99'], baseline['jain'],
            c='#4D4D4D',
            marker='o',
            s=200,
            edgecolors='white',
            linewidths=2,
            label='FCFS (baseline)',
            zorder=6
        )
        ax.annotate(
            'FCFS',
            xy=(baseline['p99'], baseline['jain']),
            xytext=(-15, -20),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            color='#4D4D4D',
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='lightgray',
                edgecolor='#4D4D4D',
                alpha=0.9
            ),
            zorder=7
        )

    # Add ideal direction indicator
    ax.annotate(
        'Ideal\n(Low P99, High Fairness)',
        xy=(0.98, 0.98),
        xycoords='axes fraction',
        fontsize=9,
        color='green',
        alpha=0.8,
        ha='right',
        va='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3)
    )

    # Add arrow direction legend
    ax.annotate(
        r'$\rightarrow$ coeff increases',
        xy=(0.98, 0.02),
        xycoords='axes fraction',
        fontsize=9,
        color='gray',
        alpha=0.8,
        ha='right',
        va='bottom'
    )

    ax.set_xlabel('P99 Waiting Time (s)', fontsize=12)
    ax.set_ylabel('Jain Fairness Index', fontsize=12)
    ax.set_title('Starvation Coefficient Trade-off: Fairness vs Tail Latency',
                 fontsize=14, fontweight='bold', pad=15)

    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_path, formats)
    plt.close()


def generate_starvation_sweep_plots(sweep_data: dict, output_dir: str,
                                     formats: List[str] = None):
    """Generate plots for starvation coefficient sweep analysis."""
    print("\n[Starvation] Generating Starvation Sweep Plots...")
    sweep_plot_dir = Path(output_dir)
    sweep_plot_dir.mkdir(parents=True, exist_ok=True)

    plot_starvation_pareto(
        sweep_data,
        str(sweep_plot_dir / 'pareto_starvation_p99_vs_jain'),
        formats=formats
    )
    print(f"  Saved: pareto_starvation_p99_vs_jain")
