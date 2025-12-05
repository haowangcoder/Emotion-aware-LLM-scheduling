"""
Distribution Plots (ECDF/CCDF)

These plots show the full distribution shape of metrics,
especially useful for tail analysis.
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..constants import SCHEDULER_ORDER, get_scheduler_color, get_scheduler_marker
from ..utils import save_figure


def plot_ecdf(dfs: Dict[str, pd.DataFrame], metric: str, output_path: str,
              title: str = None, xlabel: str = None,
              formats: List[str] = None):
    """
    Plot ECDF (Empirical Cumulative Distribution Function).

    Args:
        dfs: Dict mapping scheduler name -> DataFrame
        metric: Column name to plot
        output_path: Output file path
        title: Plot title
        xlabel: X-axis label
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for sched in SCHEDULER_ORDER:
        if sched not in dfs:
            continue

        df = dfs[sched]
        if metric not in df.columns:
            continue

        data = df[metric].dropna().sort_values()
        cdf = np.arange(1, len(data) + 1) / len(data)

        color = get_scheduler_color(sched)

        ax.plot(
            data, cdf,
            color=color,
            linewidth=2.5,
            label=sched,
            alpha=0.9
        )

        # Mark P50, P90, P99
        for pct, style in [(50, ':'), (90, '--'), (99, '-.')]:
            idx = int(len(data) * pct / 100)
            if idx < len(data):
                pct_val = data.iloc[idx]
                ax.axvline(x=pct_val, ymin=0, ymax=pct/100,
                          color=color, linestyle=style, alpha=0.4, linewidth=1)

    ax.set_xlabel(xlabel or metric.replace('_', ' ').title())
    ax.set_ylabel('Cumulative Probability')
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0)

    if title:
        ax.set_title(title, fontweight='bold', pad=10)

    # Add percentile reference lines
    for pct in [0.5, 0.9, 0.99]:
        ax.axhline(y=pct, color='gray', linestyle=':', alpha=0.3, linewidth=1)
        ax.text(ax.get_xlim()[1] * 0.98, pct + 0.01, f'P{int(pct*100)}',
               ha='right', va='bottom', fontsize=8, color='gray')

    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_path, formats)
    plt.close()


def plot_ccdf_log(dfs: Dict[str, pd.DataFrame], metric: str, output_path: str,
                  title: str = None, xlabel: str = None,
                  formats: List[str] = None):
    """
    Plot CCDF (Complementary CDF) with log y-axis for tail analysis.

    This visualization emphasizes tail behavior (P95, P99, max).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for sched in SCHEDULER_ORDER:
        if sched not in dfs:
            continue

        df = dfs[sched]
        if metric not in df.columns:
            continue

        data = df[metric].dropna().sort_values()
        ccdf = 1 - np.arange(1, len(data) + 1) / len(data)
        ccdf = np.maximum(ccdf, 1e-10)  # Avoid log(0)

        color = get_scheduler_color(sched)
        marker = get_scheduler_marker(sched)

        ax.plot(
            data, ccdf,
            color=color,
            linewidth=2.5,
            label=sched,
            alpha=0.9
        )

        # Mark P95 and P99
        for pct in [95, 99]:
            idx = int(len(data) * pct / 100)
            if idx < len(data):
                pct_val = data.iloc[idx]
                ax.plot(pct_val, 1 - pct/100, marker, color=color,
                       markersize=10, markeredgecolor='white', markeredgewidth=1)

    ax.set_yscale('log')
    ax.set_xlabel(xlabel or metric.replace('_', ' ').title())
    ax.set_ylabel('Tail Probability (1 - CDF)')
    ax.set_xlim(left=0)

    if title:
        ax.set_title(title, fontweight='bold', pad=10)

    for pct, label in [(0.05, 'P95'), (0.01, 'P99')]:
        ax.axhline(y=pct, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax.text(ax.get_xlim()[1] * 0.02, pct * 1.3, label,
               fontsize=9, color='gray')

    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.3, which='both')

    plt.tight_layout()
    save_figure(fig, output_path, formats)
    plt.close()


def generate_distribution_plots(dfs: Dict[str, pd.DataFrame], output_dir: str,
                                 formats: List[str] = None):
    """Generate all ECDF/CCDF distribution plots."""
    print("\n[B] Generating Distribution Plots...")
    dist_dir = Path(output_dir) / 'distribution'

    plot_ecdf(dfs, 'waiting_time', str(dist_dir / 'ecdf_waiting_time'),
              title='Waiting Time Distribution (ECDF)',
              xlabel='Waiting Time (s)', formats=formats)

    plot_ecdf(dfs, 'turnaround_time', str(dist_dir / 'ecdf_turnaround_time'),
              title='Turnaround Time Distribution (ECDF)',
              xlabel='Turnaround Time (s)', formats=formats)

    plot_ccdf_log(dfs, 'waiting_time', str(dist_dir / 'ccdf_waiting_time_log'),
                  title='Waiting Time Tail Analysis (CCDF)',
                  xlabel='Waiting Time (s)', formats=formats)

    plot_ccdf_log(dfs, 'turnaround_time', str(dist_dir / 'ccdf_turnaround_time_log'),
                  title='Turnaround Time Tail Analysis (CCDF)',
                  xlabel='Turnaround Time (s)', formats=formats)
