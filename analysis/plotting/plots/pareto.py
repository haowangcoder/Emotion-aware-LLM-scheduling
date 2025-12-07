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


def plot_param_heatmap(sweep_data: dict, metric: str, output_path: str,
                       formats: List[str] = None, cmap: str = None,
                       title: str = None, vmin: float = None, vmax: float = None):
    """
    Plot 2D heatmap for α × β parameter sweep.

    Args:
        sweep_data: Output from load_param_sweep_results()
        metric: Metric to plot ('avg_wait', 'p99', 'jain')
        output_path: Output file path (without extension)
        formats: Output formats (default: ['pdf', 'png'])
        cmap: Colormap name (default: metric-dependent)
        title: Plot title
        vmin, vmax: Color scale limits
    """
    import numpy as np

    alphas = sweep_data['alphas']
    betas = sweep_data['betas']
    grid = sweep_data['grid']

    # Build 2D array
    data = np.full((len(alphas), len(betas)), np.nan)
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            key = (alpha, beta)
            if key in grid:
                data[i, j] = grid[key][metric]

    # Default colormap based on metric
    if cmap is None:
        if metric == 'jain':
            cmap = 'RdYlGn'  # Red=bad, Green=good for fairness
        else:
            cmap = 'RdYlGn_r'  # Reversed: Green=good (low wait), Red=bad

    # Default title
    metric_labels = {
        'avg_wait': 'Average Waiting Time (s)',
        'p99': 'P99 Waiting Time (s)',
        'jain': 'Jain Fairness Index'
    }
    if title is None:
        title = f'Parameter Sweep: {metric_labels.get(metric, metric)}'

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(data, cmap=cmap, aspect='auto', origin='lower',
                   vmin=vmin, vmax=vmax)

    # Set ticks
    ax.set_xticks(range(len(betas)))
    ax.set_xticklabels([f'{b:.2f}' for b in betas])
    ax.set_yticks(range(len(alphas)))
    ax.set_yticklabels([f'{a:.2f}' for a in alphas])

    ax.set_xlabel(r'$\beta$ (Valence Weight)', fontsize=12)
    ax.set_ylabel(r'$\alpha$ (Arousal $\rightarrow$ Length)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(metric_labels.get(metric, metric), fontsize=11)

    # Annotate cells with values
    for i in range(len(alphas)):
        for j in range(len(betas)):
            val = data[i, j]
            if not np.isnan(val):
                # Choose text color based on background
                text_color = 'white' if (val < (vmin or data.min()) + (data.max() - data.min()) * 0.3 or
                                          val > (vmax or data.max()) - (data.max() - data.min()) * 0.3) else 'black'
                if metric == 'jain':
                    text = f'{val:.3f}'
                else:
                    text = f'{val:.1f}'
                ax.text(j, i, text, ha='center', va='center',
                       fontsize=9, fontweight='bold', color=text_color)

    plt.tight_layout()
    save_figure(fig, output_path, formats)
    plt.close()


def plot_param_dual_line(sweep_data: dict, output_path: str,
                         formats: List[str] = None):
    """
    Plot dual line charts: one for α sweep (fixed β), one for β sweep (fixed α).

    Args:
        sweep_data: Output from load_param_sweep_results()
        output_path: Output file path (without extension)
        formats: Output formats (default: ['pdf', 'png'])
    """
    alphas = sweep_data['alphas']
    betas = sweep_data['betas']
    grid = sweep_data['grid']
    baseline = sweep_data.get('baseline')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: α sweep (fix β at middle value)
    ax1 = axes[0]
    fixed_beta = betas[len(betas) // 2] if betas else 0.5

    avg_waits = []
    jains = []
    for alpha in alphas:
        key = (alpha, fixed_beta)
        if key in grid:
            avg_waits.append(grid[key]['avg_wait'])
            jains.append(grid[key]['jain'])
        else:
            avg_waits.append(None)
            jains.append(None)

    ax1_twin = ax1.twinx()
    line1, = ax1.plot(alphas, avg_waits, 'o-', color='#E69F00', lw=2,
                      markersize=8, label='Avg Wait')
    line2, = ax1_twin.plot(alphas, jains, 's--', color='#0072B2', lw=2,
                           markersize=8, label='Jain Index')

    if baseline:
        ax1.axhline(baseline['avg_wait'], color='gray', linestyle=':', alpha=0.7,
                   label='FCFS Avg Wait')
        ax1_twin.axhline(baseline['jain'], color='gray', linestyle='-.', alpha=0.7)

    ax1.set_xlabel(r'$\alpha$ (Arousal $\rightarrow$ Length Strength)', fontsize=11)
    ax1.set_ylabel('Avg Waiting Time (s)', color='#E69F00', fontsize=11)
    ax1_twin.set_ylabel('Jain Fairness Index', color='#0072B2', fontsize=11)
    ax1.set_title(rf'$\alpha$ Sweep (fixed $\beta$={fixed_beta})', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#E69F00')
    ax1_twin.tick_params(axis='y', labelcolor='#0072B2')
    ax1.grid(alpha=0.3)

    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=9)

    # Right: β sweep (fix α at middle value)
    ax2 = axes[1]
    fixed_alpha = alphas[len(alphas) // 2] if alphas else 0.5

    avg_waits = []
    jains = []
    for beta in betas:
        key = (fixed_alpha, beta)
        if key in grid:
            avg_waits.append(grid[key]['avg_wait'])
            jains.append(grid[key]['jain'])
        else:
            avg_waits.append(None)
            jains.append(None)

    ax2_twin = ax2.twinx()
    line3, = ax2.plot(betas, avg_waits, 'o-', color='#E69F00', lw=2,
                      markersize=8, label='Avg Wait')
    line4, = ax2_twin.plot(betas, jains, 's--', color='#0072B2', lw=2,
                           markersize=8, label='Jain Index')

    if baseline:
        ax2.axhline(baseline['avg_wait'], color='gray', linestyle=':', alpha=0.7,
                   label='FCFS Avg Wait')
        ax2_twin.axhline(baseline['jain'], color='gray', linestyle='-.', alpha=0.7)

    ax2.set_xlabel(r'$\beta$ (Valence Weight Strength)', fontsize=11)
    ax2.set_ylabel('Avg Waiting Time (s)', color='#E69F00', fontsize=11)
    ax2_twin.set_ylabel('Jain Fairness Index', color='#0072B2', fontsize=11)
    ax2.set_title(rf'$\beta$ Sweep (fixed $\alpha$={fixed_alpha})', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#E69F00')
    ax2_twin.tick_params(axis='y', labelcolor='#0072B2')
    ax2.grid(alpha=0.3)

    lines = [line3, line4]
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left', fontsize=9)

    plt.tight_layout()
    save_figure(fig, output_path, formats)
    plt.close()


def generate_param_sweep_plots(sweep_data: dict, output_dir: str,
                                formats: List[str] = None):
    """Generate plots for α × β parameter sweep analysis."""
    print("\n[ParamSweep] Generating Parameter Sweep Plots...")
    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Heatmap for avg_wait
    plot_param_heatmap(
        sweep_data, 'avg_wait',
        str(plot_dir / 'heatmap_avg_wait'),
        formats=formats,
        title=r'Average Waiting Time: $\alpha$ vs $\beta$'
    )
    print(f"  Saved: heatmap_avg_wait")

    # Heatmap for Jain index
    plot_param_heatmap(
        sweep_data, 'jain',
        str(plot_dir / 'heatmap_jain'),
        formats=formats,
        title=r'Jain Fairness Index: $\alpha$ vs $\beta$'
    )
    print(f"  Saved: heatmap_jain")

    # Heatmap for P99
    plot_param_heatmap(
        sweep_data, 'p99',
        str(plot_dir / 'heatmap_p99'),
        formats=formats,
        title=r'P99 Waiting Time: $\alpha$ vs $\beta$'
    )
    print(f"  Saved: heatmap_p99")

    # Dual line chart
    plot_param_dual_line(
        sweep_data,
        str(plot_dir / 'line_alpha_beta_sweep'),
        formats=formats
    )
    print(f"  Saved: line_alpha_beta_sweep")
