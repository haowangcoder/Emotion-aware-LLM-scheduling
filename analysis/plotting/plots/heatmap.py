"""
Heatmap Plots for Parameter Sweep Analysis

These plots visualize 2D parameter grids (e.g., α × β sweep).
"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from ..utils import save_figure


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
