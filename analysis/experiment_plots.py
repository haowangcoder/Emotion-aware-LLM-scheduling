#!/usr/bin/env python3
"""
Experiment Plotting Module

Consolidated visualization functions for all experiments (Exp-0 through Exp-6 + Defense A1-A3).
These functions are designed to produce publication-quality figures.

Usage:
    from analysis.experiment_plots import (
        plot_k_tradeoff_curves,
        plot_load_vs_metrics,
        plot_forest_with_ci,
        ...
    )
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Set consistent style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Color scheme
SCHEDULER_COLORS = {
    'FCFS': '#ff9f43',
    'SJF': '#4a90d9',
    'AW-SSJF': '#6bcf7f',
    'Weight-Only': '#ff6b6b',
    'Online': '#9b59b6',
}

QUADRANT_COLORS = {
    'excited': '#4a90d9',
    'calm': '#6bcf7f',
    'panic': '#ff9f43',
    'depression': '#ff6b6b',
}

K_COLORS = {
    1: '#ff6b6b',
    2: '#ff9f43',
    3: '#4a90d9',
    4: '#6bcf7f',
}


# =============================================================================
# Exp-0: Latency Decomposition
# =============================================================================

def plot_latency_decomposition_stacked(
    dfs: Dict[str, pd.DataFrame],
    output_path: Path,
    title: str = "JCT Decomposition by Scheduler"
) -> None:
    """
    Plot stacked bar chart showing JCT = waiting_time + execution_time.

    Args:
        dfs: {scheduler_name: DataFrame with waiting_time, actual_serving_time}
        output_path: Path to save plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    schedulers = list(dfs.keys())
    x = np.arange(len(schedulers))
    width = 0.6

    wait_means = []
    exec_means = []

    for sched in schedulers:
        df = dfs[sched]
        wait_means.append(df['waiting_time'].mean())
        exec_means.append(df['actual_serving_time'].mean())

    # Stacked bars
    bars1 = ax.bar(x, wait_means, width, label='Waiting Time', color='#4a90d9')
    bars2 = ax.bar(x, exec_means, width, bottom=wait_means, label='Execution Time', color='#6bcf7f')

    # Add total JCT labels on top
    for i, (w, e) in enumerate(zip(wait_means, exec_means)):
        total = w + e
        ax.text(i, total + 0.5, f'{total:.1f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Scheduler')
    ax.set_ylabel('Time (seconds)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(schedulers)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_waiting_cdf_with_markers(
    dfs: Dict[str, pd.DataFrame],
    output_path: Path,
    percentiles: List[int] = [50, 95, 99]
) -> None:
    """
    Plot waiting time CDF with P50/P95/P99 vertical markers.

    Args:
        dfs: {scheduler_name: DataFrame with waiting_time}
        output_path: Path to save plot
        percentiles: Percentiles to mark with vertical lines
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for sched, df in dfs.items():
        wait_times = df['waiting_time'].dropna().sort_values()
        y = np.arange(1, len(wait_times) + 1) / len(wait_times)

        color = SCHEDULER_COLORS.get(sched, 'gray')
        ax.plot(wait_times, y, label=sched, linewidth=2, color=color)

        # Add percentile markers
        for p in percentiles:
            pval = np.percentile(wait_times, p)
            ax.axvline(x=pval, color=color, linestyle=':', alpha=0.5)

    # Add percentile labels
    for p in percentiles:
        ax.axhline(y=p/100, color='gray', linestyle='--', alpha=0.3)
        ax.text(ax.get_xlim()[1] * 0.95, p/100 + 0.02, f'P{p}', fontsize=9, ha='right')

    ax.set_xlabel('Waiting Time (seconds)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Waiting Time CDF with Percentile Markers')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Exp-1: k-Sweep
# =============================================================================

def plot_k_tradeoff_curves(
    results: Dict[int, Dict],
    output_path: Path,
    metrics: List[str] = ['avg_waiting_time', 'depression_wait']
) -> None:
    """
    Plot trade-off curves showing metrics vs k (weight_exponent).

    Args:
        results: {k: {metric_name: value}}
        output_path: Path to save plot
        metrics: Metrics to plot on y-axis
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    k_values = sorted(results.keys())

    for metric in metrics:
        values = [results[k].get(metric, 0) for k in k_values]
        linestyle = '-' if 'depression' in metric.lower() else '--'
        marker = 'o' if 'depression' in metric.lower() else 's'
        label = metric.replace('_', ' ').title()
        ax.plot(k_values, values, f'{marker}{linestyle}', label=label, linewidth=2, markersize=10)

    ax.set_xlabel('Weight Exponent (k)', fontsize=12)
    ax.set_ylabel('Waiting Time (seconds)', fontsize=12)
    ax.set_title('Efficiency vs Fairness Trade-off Across k\n(Higher k → More fairness, Less efficiency)', fontsize=13)
    ax.set_xticks(k_values)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_pareto_frontier(
    results: Dict[int, Dict],
    output_path: Path,
    x_metric: str = 'avg_waiting_time',
    y_metric: str = 'depression_wait'
) -> None:
    """
    Plot Pareto scatter showing efficiency vs fairness trade-off.

    Args:
        results: {k: {metric_name: value}}
        output_path: Path to save plot
        x_metric: Metric for x-axis (efficiency)
        y_metric: Metric for y-axis (fairness)
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    k_values = sorted(results.keys())
    x_values = [results[k].get(x_metric, 0) for k in k_values]
    y_values = [results[k].get(y_metric, 0) for k in k_values]

    # Plot points with k as color
    for i, k in enumerate(k_values):
        color = K_COLORS.get(k, 'gray')
        ax.scatter(x_values[i], y_values[i], c=color, s=200, label=f'k={k}', zorder=5, edgecolor='black', linewidth=1.5)
        ax.annotate(f'k={k}', (x_values[i], y_values[i]), textcoords="offset points",
                   xytext=(10, 5), fontsize=11, fontweight='bold')

    # Connect points with line to show progression
    ax.plot(x_values, y_values, 'k--', alpha=0.4, linewidth=1.5, zorder=1)

    # Add ideal region indicator (bottom-left)
    ax.annotate('Better\n(Ideal)', xy=(min(x_values) * 0.9, min(y_values) * 0.8),
               fontsize=12, ha='center', color='green', fontweight='bold')

    ax.set_xlabel(x_metric.replace('_', ' ').title() + ' (seconds)', fontsize=12)
    ax.set_ylabel(y_metric.replace('_', ' ').title() + ' (seconds)', fontsize=12)
    ax.set_title('Pareto Trade-off: Efficiency vs Depression Fairness', fontsize=13)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_jain_fairness_vs_k(
    results: Dict[int, Dict],
    output_path: Path
) -> None:
    """
    Plot Jain's Fairness Index vs k.

    Args:
        results: {k: {jain_index: value}}
        output_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    k_values = sorted(results.keys())
    jain_values = [results[k].get('jain_index', 0) for k in k_values]

    colors = [K_COLORS.get(k, 'gray') for k in k_values]
    bars = ax.bar(k_values, jain_values, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, val in zip(bars, jain_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Weight Exponent (k)', fontsize=12)
    ax.set_ylabel("Jain's Fairness Index", fontsize=12)
    ax.set_title("Fairness Index vs k\n(Higher is better, 1.0 = perfect fairness)", fontsize=13)
    ax.set_xticks(k_values)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect fairness')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Exp-2: Load Sweep
# =============================================================================

def plot_metrics_vs_load(
    results: Dict[str, Dict[float, Dict]],
    output_path: Path,
    metric: str = 'avg_waiting_time'
) -> None:
    """
    Plot metric vs system load for multiple schedulers.

    Args:
        results: {scheduler: {load: {metric: value}}}
        output_path: Path to save plot
        metric: Metric to plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for sched, load_results in results.items():
        loads = sorted(load_results.keys())
        values = [load_results[l].get(metric, 0) for l in loads]

        color = SCHEDULER_COLORS.get(sched, 'gray')
        ax.plot(loads, values, 'o-', label=sched, linewidth=2, markersize=8, color=color)

    ax.axvline(x=1.0, color='red', linestyle=':', alpha=0.7, label='Saturation (ρ=1)')
    ax.set_xlabel('System Load (ρ)', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title() + ' (seconds)', fontsize=12)
    ax.set_title(f'{metric.replace("_", " ").title()} vs System Load', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_crossover_analysis(
    results: Dict[str, Dict[float, Dict]],
    output_path: Path,
    metric1: str = 'avg_waiting_time',
    metric2: str = 'depression_wait'
) -> None:
    """
    Plot two metrics to show crossover point where one scheduler beats another.

    Args:
        results: {scheduler: {load: {metrics}}}
        output_path: Path to save plot
        metric1, metric2: Two metrics to compare
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, metric in zip(axes, [metric1, metric2]):
        for sched, load_results in results.items():
            loads = sorted(load_results.keys())
            values = [load_results[l].get(metric, 0) for l in loads]

            color = SCHEDULER_COLORS.get(sched, 'gray')
            ax.plot(loads, values, 'o-', label=sched, linewidth=2, markersize=8, color=color)

        ax.axvline(x=1.0, color='red', linestyle=':', alpha=0.5)
        ax.set_xlabel('System Load (ρ)', fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title() + ' (s)', fontsize=11)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Exp-3: gamma_panic Sweep
# =============================================================================

def plot_quadrant_comparison_vs_gamma(
    results: Dict[float, Dict],
    output_path: Path,
    quadrants: List[str] = ['panic', 'depression']
) -> None:
    """
    Plot quadrant waiting times vs gamma_panic.

    Args:
        results: {gamma: {quadrant_wait: value}}
        output_path: Path to save plot
        quadrants: Quadrants to plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    gamma_values = sorted(results.keys())

    for quadrant in quadrants:
        metric_key = f'{quadrant}_wait'
        values = [results[g].get(metric_key, 0) for g in gamma_values]

        color = QUADRANT_COLORS.get(quadrant, 'gray')
        ax.plot(gamma_values, values, 'o-', label=quadrant.title(), linewidth=2, markersize=10, color=color)

    ax.set_xlabel('γ_panic (Panic Channel Weight)', fontsize=12)
    ax.set_ylabel('Average Waiting Time (seconds)', fontsize=12)
    ax.set_title('Quadrant Wait Times vs γ_panic\n(DUAL_CHANNEL Mode)', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Exp-4: Online Control
# =============================================================================

def plot_online_control_timeseries(
    k_history: List[Tuple[float, float]],
    load_history: List[Tuple[float, int]],
    metrics_history: Optional[List[Tuple[float, float]]] = None,
    output_path: Path = None,
    burst_phases: List[Tuple[float, float]] = None
) -> None:
    """
    Plot three-panel time series for online control.

    Args:
        k_history: [(time, k_value), ...]
        load_history: [(time, queue_length), ...]
        metrics_history: [(time, metric_value), ...] optional
        output_path: Path to save plot
        burst_phases: [(start, end), ...] for shading burst periods
    """
    fig, axes = plt.subplots(3 if metrics_history else 2, 1, figsize=(12, 10), sharex=True)

    # Panel 1: Queue length
    ax1 = axes[0]
    if load_history:
        times, lengths = zip(*load_history)
        ax1.plot(times, lengths, 'b-', linewidth=1, alpha=0.7)
    ax1.set_ylabel('Queue Length', fontsize=11)
    ax1.set_title('Online Control: Load Signal and k(t) Trajectory', fontsize=13)
    ax1.grid(True, alpha=0.3)

    # Shade burst phases
    if burst_phases:
        for start, end in burst_phases:
            ax1.axvspan(start, end, color='red', alpha=0.1)

    # Panel 2: k trajectory
    ax2 = axes[1]
    if k_history:
        times, k_values = zip(*k_history)
        ax2.step(times, k_values, where='post', linewidth=2, color='purple')
        ax2.scatter(times, k_values, color='purple', s=50, zorder=5)
    ax2.set_ylabel('k (weight_exponent)', fontsize=11)
    ax2.set_ylim(0.5, 4.5)
    ax2.grid(True, alpha=0.3)

    if burst_phases:
        for start, end in burst_phases:
            ax2.axvspan(start, end, color='red', alpha=0.1)

    # Panel 3: Optional metrics
    if metrics_history:
        ax3 = axes[2]
        times, values = zip(*metrics_history)
        ax3.plot(times, values, 'g-', linewidth=1, alpha=0.7)
        ax3.set_ylabel('Depression Wait (s)', fontsize=11)
        ax3.set_xlabel('Time (seconds)', fontsize=11)
        ax3.grid(True, alpha=0.3)

        if burst_phases:
            for start, end in burst_phases:
                ax3.axvspan(start, end, color='red', alpha=0.1)
    else:
        ax2.set_xlabel('Time (seconds)', fontsize=11)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close()


def plot_online_vs_static_comparison(
    results: Dict[str, Dict],
    output_path: Path,
    metrics: List[str] = ['avg_waiting_time', 'depression_wait']
) -> None:
    """
    Bar chart comparing online adaptive policy vs static k values.

    Args:
        results: {policy_name: {metric: value}}
        output_path: Path to save plot
        metrics: Metrics to compare
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    policies = list(results.keys())

    for ax, metric in zip(axes, metrics):
        values = [results[p].get(metric, 0) for p in policies]
        colors = ['#9b59b6' if 'online' in p.lower() else '#4a90d9' for p in policies]

        bars = ax.bar(policies, values, color=colors, edgecolor='black')
        ax.set_ylabel(metric.replace('_', ' ').title() + ' (s)', fontsize=11)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        # Value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Exp-5: Queueing Model
# =============================================================================

def plot_model_prediction_scatter(
    actual: np.ndarray,
    predicted: np.ndarray,
    output_path: Path,
    title: str = "Model Prediction vs Actual"
) -> None:
    """
    Scatter plot of predicted vs actual with y=x reference line.

    Args:
        actual: Actual values
        predicted: Predicted values
        output_path: Path to save plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(actual, predicted, alpha=0.5, s=30, c='#4a90d9')

    # y=x reference line
    lims = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    ax.plot(lims, lims, 'r--', linewidth=2, label='y=x (perfect)')

    # Calculate R2
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, fontsize=12,
           verticalalignment='top', fontweight='bold')

    ax.set_xlabel('Actual Waiting Time (seconds)', fontsize=12)
    ax.set_ylabel('Predicted Waiting Time (seconds)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_residual_vs_load(
    loads: List[float],
    residuals: List[float],
    output_path: Path
) -> None:
    """
    Bar plot of model residuals vs system load.

    Args:
        loads: System load values
        residuals: Residual values (actual - predicted)
        output_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ['#6bcf7f' if r >= 0 else '#ff6b6b' for r in residuals]
    ax.bar(loads, residuals, width=0.08, color=colors, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    ax.set_xlabel('System Load (ρ)', fontsize=12)
    ax.set_ylabel('Residual (Actual - Predicted)', fontsize=12)
    ax.set_title('Model Residuals by Load\n(Shows where model over/under-predicts)', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Exp-6: Multi-Seed Statistical Validation
# =============================================================================

def plot_forest_with_ci(
    results: Dict[str, Dict[str, Dict]],
    metric: str,
    output_path: Path,
    title: str = None
) -> None:
    """
    Forest plot showing mean with 95% CI for each scheduler.

    Args:
        results: {scheduler: {metric: {mean, ci_lower, ci_upper, n}}}
        metric: Metric to plot
        output_path: Path to save plot
        title: Optional title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    schedulers = list(results.keys())
    y_positions = range(len(schedulers))

    for i, sched in enumerate(schedulers):
        if metric not in results[sched]:
            continue

        data = results[sched][metric]
        mean = data['mean']
        ci_lower = data['ci_lower']
        ci_upper = data['ci_upper']
        n = data.get('n', 0)

        color = SCHEDULER_COLORS.get(sched, 'gray')

        # Plot error bar
        ax.errorbar(mean, i, xerr=[[mean - ci_lower], [ci_upper - mean]],
                   fmt='o', markersize=10, capsize=5, capthick=2,
                   color=color, ecolor=color, linewidth=2)

        # Add annotation
        ax.text(ci_upper + 0.5, i, f'{mean:.2f} (n={n})', va='center', fontsize=10)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(schedulers, fontsize=11)
    ax.set_xlabel(f'{metric.replace("_", " ").title()} (seconds)', fontsize=12)
    ax.set_title(title or f'{metric.replace("_", " ").title()} with 95% CI', fontsize=13)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_boxplot_distribution(
    all_values: Dict[str, List[float]],
    output_path: Path,
    metric_name: str = "Metric"
) -> None:
    """
    Boxplot showing distribution across seeds.

    Args:
        all_values: {scheduler: [values across seeds]}
        output_path: Path to save plot
        metric_name: Name of metric for labels
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    schedulers = list(all_values.keys())
    data = [all_values[s] for s in schedulers]
    colors = [SCHEDULER_COLORS.get(s, 'gray') for s in schedulers]

    bp = ax.boxplot(data, labels=schedulers, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel(f'{metric_name} (seconds)', fontsize=12)
    ax.set_title(f'Distribution of {metric_name} Across Seeds', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Defense Experiments (A1-A3)
# =============================================================================

def plot_starvation_sweep(
    df: pd.DataFrame,
    output_path: Path,
    x_col: str = 'threshold',
    y_col: str = 'p99_waiting_time',
    hue_col: str = 'coefficient'
) -> None:
    """
    Plot starvation parameter sweep results.

    Args:
        df: DataFrame with threshold, coefficient, metrics
        output_path: Path to save plot
        x_col, y_col, hue_col: Column names for plotting
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for hue_val in df[hue_col].unique():
        subset = df[df[hue_col] == hue_val].sort_values(x_col)
        x = [str(t) if t != float('inf') else '∞' for t in subset[x_col]]
        ax.plot(range(len(x)), subset[y_col], 'o-', label=f'{hue_col}={hue_val}', linewidth=2, markersize=8)
        ax.set_xticks(range(len(x)))
        ax.set_xticklabels(x)

    ax.set_xlabel(x_col.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(y_col.replace('_', ' ').title() + ' (seconds)', fontsize=12)
    ax.set_title(f'{y_col.replace("_", " ").title()} vs {x_col.replace("_", " ").title()}', fontsize=13)
    ax.legend(title=hue_col.replace('_', ' ').title())
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_ablation_comparison(
    df: pd.DataFrame,
    output_path: Path,
    category_col: str,
    metrics: List[str] = ['avg_waiting_time', 'depression_wait', 'p99_waiting_time']
) -> None:
    """
    Multi-panel bar chart for ablation comparison.

    Args:
        df: DataFrame with category and metrics
        output_path: Path to save plot
        category_col: Column to group by
        metrics: Metrics to show in panels
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    categories = df[category_col].tolist()
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(categories)))

    for ax, metric in zip(axes, metrics):
        values = df[metric].tolist()
        bars = ax.bar(categories, values, color=colors, edgecolor='black')
        ax.set_ylabel(metric.replace('_', ' ').title() + ' (s)', fontsize=11)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        # Value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                   f'{val:.1f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# Utility Functions
# =============================================================================

def set_publication_style():
    """Set matplotlib parameters for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def create_figure_grid(
    nrows: int,
    ncols: int,
    figsize: Tuple[int, int] = None,
    sharex: bool = False,
    sharey: bool = False
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a figure with subplots grid.

    Args:
        nrows, ncols: Grid dimensions
        figsize: Figure size (auto-calculated if None)
        sharex, sharey: Whether to share axes

    Returns:
        Tuple of (figure, axes array)
    """
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey)
    return fig, axes


if __name__ == "__main__":
    print("Experiment Plots Module")
    print("=======================")
    print("Available functions:")
    print("  - plot_latency_decomposition_stacked")
    print("  - plot_waiting_cdf_with_markers")
    print("  - plot_k_tradeoff_curves")
    print("  - plot_pareto_frontier")
    print("  - plot_jain_fairness_vs_k")
    print("  - plot_metrics_vs_load")
    print("  - plot_crossover_analysis")
    print("  - plot_quadrant_comparison_vs_gamma")
    print("  - plot_online_control_timeseries")
    print("  - plot_online_vs_static_comparison")
    print("  - plot_model_prediction_scatter")
    print("  - plot_residual_vs_load")
    print("  - plot_forest_with_ci")
    print("  - plot_boxplot_distribution")
    print("  - plot_starvation_sweep")
    print("  - plot_ablation_comparison")
