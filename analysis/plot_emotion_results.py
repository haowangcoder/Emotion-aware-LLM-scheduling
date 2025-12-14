"""
Visualization Module for Emotion-aware LLM Scheduling Results

This module provides comprehensive visualization functions for analyzing and comparing
emotion-aware scheduling strategies. It generates various plots including:
- Performance comparison bar charts
- Fairness analysis across emotion classes
- Cumulative task completion curves
- Tail latency distributions
- Per-emotion class performance breakdowns
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import pandas as pd
import numpy as np
import seaborn as sns
from typing import List, Dict, Tuple
import os
import glob

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# Russell Quadrant definitions (circumplex model)
RUSSELL_QUADRANTS = ['excited', 'calm', 'panic', 'depression']

# Color scheme matching Russell's Circumplex emotional semantics
QUADRANT_COLORS = {
    'excited': '#4a90d9',    # Blue - pleasant + activated
    'calm': '#6bcf7f',       # Green - pleasant + deactivated
    'panic': '#ff9f43',      # Orange - unpleasant + activated
    'depression': '#ff6b6b', # Red - unpleasant + deactivated (TARGET)
}

# 2x2 grid layout for Russell heatmaps (rows: arousal desc, cols: valence asc)
# Layout:  panic(0,0)    excited(0,1)
#          depression(1,0)  calm(1,1)
QUADRANT_GRID_LAYOUT = {
    'panic': (0, 0),      # High arousal, Negative valence
    'excited': (0, 1),    # High arousal, Positive valence
    'depression': (1, 0), # Low arousal, Negative valence
    'calm': (1, 1),       # Low arousal, Positive valence
}

# Legacy emotion classes (kept for backward compatibility)
LEGACY_EMOTION_CLASSES = ['high', 'medium', 'low']


def _shorten_scheduler_name(name: str) -> str:
    """Shorten scheduler name for display by removing common suffixes."""
    import re
    # Remove common suffixes like "_80jobs_load0.90_fixed_jobs"
    shortened = re.sub(r'_\d+jobs_load[\d.]+_\w+$', '', name)
    # Also handle _jobs suffix
    shortened = re.sub(r'_jobs$', '', shortened)
    return shortened


def load_job_logs(csv_path: str) -> pd.DataFrame:
    """
    Load job logs from CSV file

    Args:
        csv_path: Path to CSV file with job logs

    Returns:
        DataFrame with job data
    """
    df = pd.read_csv(csv_path)
    return df


def plot_scheduler_comparison_barplot(results: Dict[str, Dict],
                                       metrics: List[str] = None,
                                       output_path: str = 'scheduler_comparison.png'):
    """
    Create bar plot comparing multiple schedulers across various metrics

    Args:
        results: Dictionary mapping scheduler_name -> metrics_dict
        metrics: List of metric names to plot
        output_path: Path to save the figure
    """
    if metrics is None:
        metrics = ['avg_waiting_time', 'p99_waiting_time', 'avg_jct', 'throughput_p25']

    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(5*num_metrics, 5))

    if num_metrics == 1:
        axes = [axes]

    schedulers = list(results.keys())
    short_names = [_shorten_scheduler_name(s) for s in schedulers]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        values = [results[sched].get(metric, 0) for sched in schedulers]
        colors = sns.color_palette("husl", len(schedulers))

        bars = ax.bar(range(len(schedulers)), values, color=colors, alpha=0.8)
        ax.set_xticks(range(len(schedulers)))
        ax.set_xticklabels(short_names, rotation=45, ha='right')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()}')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved scheduler comparison plot to: {output_path}")
    plt.close()


def plot_fairness_comparison(fairness_results: Dict[str, Dict],
                              output_path: str = 'fairness_comparison.png'):
    """
    Plot Fairness metrics comparison across schedulers.

    Shows:
    - Jain Fairness Index (by Russell quadrant)
    - Depression vs Others ratio (< 1.0 = depression prioritized)

    Args:
        fairness_results: Dict mapping scheduler_name -> fairness_dict
                         (with keys: 'jain_index', 'depression_vs_others', 'cv')
        output_path: Path to save figure
    """
    schedulers = list(fairness_results.keys())
    short_names = [_shorten_scheduler_name(s) for s in schedulers]

    # Handle both old format (float) and new format (dict)
    is_new_format = isinstance(list(fairness_results.values())[0], dict) if fairness_results else False

    if is_new_format:
        # New format with dict values
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Jain Fairness Index
        jain_indices = [fairness_results[s].get('jain_index', 0) for s in schedulers]
        colors = sns.color_palette("RdYlGn", len(schedulers))

        bars1 = ax1.barh(range(len(schedulers)), jain_indices, color=colors, alpha=0.8)
        ax1.set_yticks(range(len(schedulers)))
        ax1.set_yticklabels(short_names)
        ax1.set_xlabel('Jain Fairness Index')
        ax1.set_title('Quadrant Fairness\n(Higher = More Fair, 1.0 = Perfect)')
        ax1.set_xlim(0, 1.05)
        ax1.axvline(x=1.0, color='green', linestyle='--', alpha=0.5)
        ax1.grid(axis='x', alpha=0.3)

        for i, bar in enumerate(bars1):
            ax1.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2.,
                     f'{jain_indices[i]:.4f}', ha='left', va='center', fontsize=9)

        # Plot 2: Depression vs Others Ratio
        dep_ratios = [fairness_results[s].get('depression_vs_others', 1.0) for s in schedulers]
        colors2 = ['green' if r < 1.0 else ('red' if r > 1.0 else 'gray') for r in dep_ratios]

        bars2 = ax2.barh(range(len(schedulers)), dep_ratios, color=colors2, alpha=0.8)
        ax2.set_yticks(range(len(schedulers)))
        ax2.set_yticklabels(short_names)
        ax2.set_xlabel('Depression / Others Ratio')
        ax2.set_title('Depression-First Effectiveness\n(< 1.0 = Depression prioritized)')
        ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
        ax2.grid(axis='x', alpha=0.3)

        for i, bar in enumerate(bars2):
            ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2.,
                     f'{dep_ratios[i]:.2f}', ha='left', va='center', fontsize=9)

        plt.tight_layout()
    else:
        # Legacy format with float values
        fig, ax = plt.subplots(figsize=(8, 6))

        jain_indices = list(fairness_results.values())
        colors = sns.color_palette("RdYlGn", len(schedulers))
        bars = ax.barh(range(len(schedulers)), jain_indices, color=colors, alpha=0.8)

        ax.set_yticks(range(len(schedulers)))
        ax.set_yticklabels(short_names)
        ax.set_xlabel('Fairness Index')
        ax.set_title('Fairness Comparison Across Schedulers\n(Higher is Better, 1.0 = Perfect Fairness)')
        ax.set_xlim(0, 1.05)
        ax.axvline(x=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect Fairness')
        ax.grid(axis='x', alpha=0.3)
        ax.legend()

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                    f'{width:.4f}', ha='left', va='center', fontsize=10)

        plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved fairness comparison plot to: {output_path}")
    plt.close()


def _compute_scheduler_metrics_from_df(df: pd.DataFrame) -> Dict[str, float]:
    """Compute core metrics from a jobs CSV DataFrame."""
    waiting = df['waiting_time'].dropna().to_numpy()
    turnaround = df['turnaround_time'].dropna().to_numpy()
    finish_times = df['finish_time'].dropna().to_numpy()

    # Basic metrics
    metrics = {
        'avg_waiting_time': float(np.mean(waiting)) if waiting.size else 0.0,
        'p50_waiting_time': float(np.percentile(waiting, 50)) if waiting.size else 0.0,
        'p95_waiting_time': float(np.percentile(waiting, 95)) if waiting.size else 0.0,
        'p99_waiting_time': float(np.percentile(waiting, 99)) if waiting.size else 0.0,
        'avg_jct': float(np.mean(turnaround)) if turnaround.size else 0.0,
        'p50_jct': float(np.percentile(turnaround, 50)) if turnaround.size else 0.0,
        'p95_jct': float(np.percentile(turnaround, 95)) if turnaround.size else 0.0,
        'p99_jct': float(np.percentile(turnaround, 99)) if turnaround.size else 0.0,
        'throughput': float(len(df) / finish_times.max()) if finish_times.size and finish_times.max() > 0 else 0.0,
    }

    # Percentile throughput
    if finish_times.size > 0:
        sorted_times = np.sort(finish_times)
        n = len(sorted_times)
        for pct in [25, 50, 75]:
            idx = int(n * pct / 100)
            if idx > 0 and sorted_times[idx] > 0:
                metrics[f'throughput_p{pct}'] = float((idx + 1) / sorted_times[idx])
            else:
                metrics[f'throughput_p{pct}'] = 0.0

    return metrics


def _compute_jain_fairness_from_df_legacy(df: pd.DataFrame) -> float:
    """Legacy: Compute Jain Fairness Index over per-emotion-class average waiting time."""
    if 'emotion_class' not in df.columns:
        return 0.0
    grp = df.dropna(subset=['waiting_time']).groupby('emotion_class')['waiting_time'].mean()
    vals = grp.to_numpy()
    if vals.size == 0:
        return 0.0
    numerator = (vals.sum() ** 2)
    denominator = (len(vals) * (vals ** 2).sum())
    return float(numerator / denominator) if denominator > 0 else 0.0


def _compute_jain_fairness_by_quadrant(df: pd.DataFrame, metric: str = 'waiting_time') -> float:
    """
    Compute Jain Fairness Index over per-Russell-quadrant average metric.

    Args:
        df: DataFrame with job logs containing 'russell_quadrant' column
        metric: Metric to compute fairness over ('waiting_time' or 'turnaround_time')

    Returns:
        Jain Fairness Index in [0.25, 1.0] (1.0 = perfect fairness)
    """
    if 'russell_quadrant' not in df.columns:
        # Fallback to legacy emotion_class if available
        return _compute_jain_fairness_from_df_legacy(df)

    grp = df.dropna(subset=[metric]).groupby('russell_quadrant')[metric].mean()
    vals = grp.reindex(RUSSELL_QUADRANTS).dropna().to_numpy()

    if vals.size == 0:
        return 0.0

    numerator = (vals.sum() ** 2)
    denominator = (len(vals) * (vals ** 2).sum())
    return float(numerator / denominator) if denominator > 0 else 0.0


def _compute_weighted_jain_fairness_from_df_legacy(df: pd.DataFrame, beta: float = 0.8) -> float:
    """
    Legacy: Compute Weighted Jain Fairness Index over per-valence-class average waiting time.

    Used for old SSJF-Valence and SSJF-Combined schedulers.
    Weight: π_i = 1 + β(-v_i), where v_i is the average valence of the class.
    """
    if 'valence_class' not in df.columns:
        return 0.0

    df_valid = df.dropna(subset=['waiting_time'])
    if df_valid.empty:
        return 0.0

    grp = df_valid.groupby('valence_class')['waiting_time'].mean()

    if grp.empty:
        return 0.0

    # Define valence values for each class
    valence_map = {'negative': -0.8, 'neutral': 0.0, 'positive': 0.8}

    values = []
    weights = []
    for valence_class, avg_waiting in grp.items():
        valence_val = valence_map.get(valence_class, 0.0)
        weight = 1.0 + beta * (-valence_val)
        if weight <= 0:
            weight = 1e-6
        values.append(avg_waiting)
        weights.append(weight)

    if not values:
        return 0.0

    values = np.array(values)
    weights = np.array(weights)

    if np.all(values == 0):
        return 1.0

    numerator = (np.sum(weights * values)) ** 2
    denominator = len(values) * np.sum((weights ** 2) * (values ** 2))

    if denominator == 0:
        return 1.0

    return float(numerator / denominator)


def _compute_fairness_for_scheduler(df: pd.DataFrame, scheduler_name: str) -> Dict[str, float]:
    """
    Compute fairness metrics appropriate for the scheduler type.

    Returns:
        Dictionary with:
        - 'jain_index': Jain Fairness Index by quadrant (or legacy emotion_class)
        - 'depression_vs_others': Ratio of depression quadrant avg to others
        - 'cv': Coefficient of variation across quadrants
    """
    # Check if using new quadrant format
    if 'russell_quadrant' in df.columns:
        # Compute per-quadrant metrics
        quadrant_metrics = {}
        for quadrant in RUSSELL_QUADRANTS:
            quadrant_data = df[df['russell_quadrant'] == quadrant]['waiting_time'].dropna()
            if len(quadrant_data) > 0:
                quadrant_metrics[quadrant] = quadrant_data.mean()

        if not quadrant_metrics:
            return {'jain_index': 1.0, 'depression_vs_others': 1.0, 'cv': 0.0}

        values = list(quadrant_metrics.values())
        jain_index = _compute_jain_fairness_by_quadrant(df)

        # Depression vs others ratio
        depression_val = quadrant_metrics.get('depression', 0)
        other_vals = [v for k, v in quadrant_metrics.items() if k != 'depression']
        depression_vs_others = depression_val / np.mean(other_vals) if other_vals and np.mean(other_vals) > 0 else 1.0

        # Coefficient of variation
        cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 0.0

        return {
            'jain_index': float(jain_index),
            'depression_vs_others': float(depression_vs_others),
            'cv': float(cv),
        }
    else:
        # Legacy fallback
        return {'jain_index': _compute_jain_fairness_from_df_legacy(df), 'depression_vs_others': 1.0, 'cv': 0.0}


def plot_percentile_throughput(results: Dict[str, Dict],
                                output_path: str = 'percentile_throughput.png'):
    """
    Plot percentile throughput comparison across schedulers.

    Shows throughput at P25, P50, P75 completion milestones.

    Args:
        results: Dictionary mapping scheduler_name -> metrics_dict
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    schedulers = list(results.keys())
    short_names = [_shorten_scheduler_name(s) for s in schedulers]
    percentiles = ['throughput_p25', 'throughput_p50', 'throughput_p75', 'throughput']
    labels = ['P25', 'P50', 'P75', 'Overall']

    x = np.arange(len(schedulers))
    width = 0.2
    colors = sns.color_palette("viridis", len(percentiles))

    for i, (pct, label) in enumerate(zip(percentiles, labels)):
        values = [results[sched].get(pct, 0) for sched in schedulers]
        bars = ax.bar(x + i * width, values, width, label=label, color=colors[i], alpha=0.8)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Scheduler')
    ax.set_ylabel('Throughput (jobs/sec)')
    ax.set_title('Percentile Throughput Comparison\n(Higher is Better)')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(short_names, rotation=45, ha='right')
    ax.legend(title='Completion %', loc='upper center', bbox_to_anchor=(0.5, -0.32), ncol=4)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved percentile throughput plot to: {output_path}")
    plt.close()


def scan_llm_runs_and_generate_plots(runs_dir: str = 'results/llm_runs', output_dir: str = None):
    """
    Scan results/llm_runs for *_jobs.csv files and generate comparison plots.

    Saves:
      - scheduler_comparison.png
      - fairness_comparison.png
    under output_dir/plots (defaults to runs_dir/plots).
    """
    runs_dir = os.path.abspath(runs_dir)
    if output_dir is None:
        output_dir = runs_dir
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    csv_paths = sorted(glob.glob(os.path.join(runs_dir, '*_jobs.csv')))
    if not csv_paths:
        print(f"No job CSVs found in: {runs_dir}")
        return

    scheduler_results: Dict[str, Dict[str, float]] = {}
    fairness_results: Dict[str, float] = {}

    for csv_path in csv_paths:
        df = load_job_logs(csv_path)
        # Key name from filename without extension
        key = os.path.splitext(os.path.basename(csv_path))[0]
        metrics = _compute_scheduler_metrics_from_df(df)
        scheduler_results[key] = metrics
        # Use appropriate fairness metric based on scheduler type
        fairness_results[key] = _compute_fairness_for_scheduler(df, scheduler_name=key)

    # Generate plots
    plot_scheduler_comparison_barplot(
        scheduler_results,
        output_path=os.path.join(plots_dir, 'scheduler_comparison.png')
    )
    plot_fairness_comparison(
        fairness_results,
        output_path=os.path.join(plots_dir, 'fairness_comparison.png')
    )
    plot_percentile_throughput(
        scheduler_results,
        output_path=os.path.join(plots_dir, 'percentile_throughput.png')
    )
    print(f"Comparison plots saved to: {plots_dir}")

    # Generate comprehensive report plots (completion curves, CDF, heatmaps, etc.)
    result_dirs = {
        os.path.splitext(os.path.basename(path))[0]: path
        for path in csv_paths
    }
    generate_comprehensive_report(result_dirs, output_dir=plots_dir)


def plot_per_emotion_class_metrics_legacy(df: pd.DataFrame,
                                          metric: str = 'waiting_time',
                                          scheduler_name: str = None,
                                          output_path: str = 'per_emotion_metrics.png'):
    """
    Legacy: Plot metrics broken down by emotion class (high/medium/low arousal)

    Args:
        df: DataFrame with job logs including emotion_class column
        metric: Metric to plot ('waiting_time', 'turnaround_time', or 'service_time')
        scheduler_name: Name of scheduler for title
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by emotion class
    emotion_classes = LEGACY_EMOTION_CLASSES
    data = []
    for emotion_class in emotion_classes:
        class_data = df[df['emotion_class'] == emotion_class][metric].dropna()
        data.append(class_data)

    # Create violin plot
    parts = ax.violinplot(data, positions=range(len(emotion_classes)),
                          showmeans=True, showmedians=True)

    # Customize colors
    colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']  # red, yellow, green for high, medium, low
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(emotion_classes)))
    ax.set_xticklabels([f'{ec.capitalize()} Arousal' for ec in emotion_classes])
    ax.set_ylabel(metric.replace('_', ' ').title())
    title = f'{metric.replace("_", " ").title()} by Emotion Class'
    if scheduler_name:
        title += f'\n({scheduler_name})'
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)

    # Add statistical annotations
    for i, emotion_class in enumerate(emotion_classes):
        class_data = df[df['emotion_class'] == emotion_class][metric].dropna()
        mean_val = class_data.mean()
        ax.text(i, ax.get_ylim()[1] * 0.95,
                f'Mean: {mean_val:.2f}\nn={len(class_data)}',
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved per-emotion class metrics plot to: {output_path}")
    plt.close()


def plot_per_quadrant_metrics(df: pd.DataFrame,
                               metric: str = 'waiting_time',
                               scheduler_name: str = None,
                               output_path: str = 'per_quadrant_metrics.png'):
    """
    Plot metrics broken down by Russell quadrant with semantic coloring.

    Uses violin plot to show distribution within each quadrant.
    Depression quadrant is highlighted as the target for priority boost.

    Args:
        df: DataFrame with job logs including 'russell_quadrant' column
        metric: Metric to plot ('waiting_time', 'turnaround_time', or 'actual_serving_time')
        scheduler_name: Name of scheduler for title
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Collect data per quadrant
    data = []
    colors = []
    valid_quadrants = []
    for quadrant in RUSSELL_QUADRANTS:
        quadrant_data = df[df['russell_quadrant'] == quadrant][metric].dropna()
        if len(quadrant_data) > 0:
            data.append(quadrant_data)
            colors.append(QUADRANT_COLORS[quadrant])
            valid_quadrants.append(quadrant)
        else:
            data.append(pd.Series([0]))
            colors.append(QUADRANT_COLORS[quadrant])
            valid_quadrants.append(quadrant)

    # Create violin plot
    parts = ax.violinplot(data, positions=range(len(RUSSELL_QUADRANTS)),
                          showmeans=True, showmedians=True)

    # Apply quadrant colors
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
        # Highlight depression quadrant with thicker edge
        if RUSSELL_QUADRANTS[i] == 'depression':
            pc.set_edgecolor('#8B0000')
            pc.set_linewidth(2)

    ax.set_xticks(range(len(RUSSELL_QUADRANTS)))
    ax.set_xticklabels([q.capitalize() for q in RUSSELL_QUADRANTS])
    ax.set_ylabel(metric.replace('_', ' ').title())

    title = f'{metric.replace("_", " ").title()} by Russell Quadrant'
    if scheduler_name:
        title += f'\n({scheduler_name})'
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)

    # Add statistical annotations
    for i, quadrant in enumerate(RUSSELL_QUADRANTS):
        quadrant_data = df[df['russell_quadrant'] == quadrant][metric].dropna()
        if len(quadrant_data) > 0:
            mean_val = quadrant_data.mean()
            annotation = f'Mean: {mean_val:.2f}\nn={len(quadrant_data)}'
            if quadrant == 'depression':
                annotation += '\n(TARGET)'
            ax.text(i, ax.get_ylim()[1] * 0.95, annotation,
                    ha='center', va='top', fontsize=9,
                    bbox=dict(boxstyle='round',
                              facecolor='lightyellow' if quadrant == 'depression' else 'white',
                              alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved per-quadrant metrics plot to: {output_path}")
    plt.close()


def plot_completion_curves(dfs: Dict[str, pd.DataFrame],
                            output_path: str = 'completion_curves.png'):
    """
    Plot cumulative task completion curves for multiple schedulers

    Args:
        dfs: Dictionary mapping scheduler_name -> DataFrame with job logs
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = sns.color_palette("husl", len(dfs))

    for i, (scheduler_name, df) in enumerate(dfs.items()):
        # Sort by completion time
        sorted_times = df['finish_time'].dropna().sort_values()
        cumulative_count = range(1, len(sorted_times) + 1)

        ax.plot(sorted_times, cumulative_count,
                label=scheduler_name, linewidth=2.5, color=colors[i], alpha=0.8)

    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative Number of Completed Jobs')
    ax.set_title('Task Completion Curves\n(Higher/Left is Better)')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved completion curves plot to: {output_path}")
    plt.close()


def plot_latency_cdf(dfs: Dict[str, pd.DataFrame],
                      metric: str = 'turnaround_time',
                      output_path: str = 'latency_cdf.png'):
    """
    Plot cumulative distribution function of latency for multiple schedulers

    Args:
        dfs: Dictionary mapping scheduler_name -> DataFrame with job logs
        metric: Metric to plot ('waiting_time' or 'turnaround_time')
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = sns.color_palette("husl", len(dfs))

    for i, (scheduler_name, df) in enumerate(dfs.items()):
        data = df[metric].dropna().sort_values()
        cdf = np.arange(1, len(data) + 1) / len(data)

        ax.plot(data, cdf,
                label=scheduler_name, linewidth=2.5, color=colors[i], alpha=0.8)

        # Mark p99
        p99_idx = int(len(data) * 0.99)
        if p99_idx < len(data):
            p99_val = data.iloc[p99_idx]
            ax.plot(p99_val, 0.99, 'o', color=colors[i], markersize=8)
            ax.text(p99_val, 0.99, f'  P99: {p99_val:.2f}',
                    va='center', fontsize=9, color=colors[i])

    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_ylabel('Cumulative Probability')
    ax.set_title(f'{metric.replace("_", " ").title()} CDF\n(Lower/Left is Better)')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved latency CDF plot to: {output_path}")
    plt.close()


def plot_emotion_class_comparison_heatmap_legacy(dfs: Dict[str, pd.DataFrame],
                                                  metric: str = 'waiting_time',
                                                  output_path: str = 'emotion_heatmap.png'):
    """
    Legacy: Create heatmap comparing average metric across schedulers and emotion classes

    Args:
        dfs: Dictionary mapping scheduler_name -> DataFrame with job logs
        metric: Metric to plot
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data matrix
    schedulers = list(dfs.keys())
    short_names = [_shorten_scheduler_name(s) for s in schedulers]
    emotion_classes = LEGACY_EMOTION_CLASSES

    data_matrix = []
    for scheduler_name in schedulers:
        df = dfs[scheduler_name]
        row = []
        for emotion_class in emotion_classes:
            class_data = df[df['emotion_class'] == emotion_class][metric].dropna()
            avg_val = class_data.mean() if len(class_data) > 0 else 0
            row.append(avg_val)
        data_matrix.append(row)

    data_matrix = np.array(data_matrix)

    # Create heatmap
    im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto')

    # Set ticks
    ax.set_xticks(range(len(emotion_classes)))
    ax.set_yticks(range(len(schedulers)))
    ax.set_xticklabels([f'{ec.capitalize()} Arousal' for ec in emotion_classes])
    ax.set_yticklabels(short_names)

    # Add colorbar
    plt.colorbar(im, ax=ax, label=metric.replace('_', ' ').title())

    # Add text annotations
    for i in range(len(schedulers)):
        for j in range(len(emotion_classes)):
            text = ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)

    ax.set_title(f'Average {metric.replace("_", " ").title()} by Scheduler and Emotion Class')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved emotion class comparison heatmap to: {output_path}")
    plt.close()


def plot_quadrant_comparison_heatmap(dfs: Dict[str, pd.DataFrame],
                                      metric: str = 'waiting_time',
                                      output_path: str = 'quadrant_heatmap.png'):
    """
    Create heatmap comparing average metric across schedulers and Russell quadrants.

    Layout follows Russell's Circumplex Model:
    - Rows: High Arousal (top), Low Arousal (bottom)
    - Cols: Negative Valence (left), Positive Valence (right)

    Args:
        dfs: Dictionary mapping scheduler_name -> DataFrame with job logs
        metric: Metric to plot
        output_path: Path to save the figure
    """
    schedulers = list(dfs.keys())
    short_names = [_shorten_scheduler_name(s) for s in schedulers]
    num_schedulers = len(schedulers)

    # Create subplot grid: one 2x2 heatmap per scheduler
    cols = min(num_schedulers, 2)
    rows = (num_schedulers + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))

    if num_schedulers == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    # Collect all values for common color scale
    all_values = []
    for scheduler_name, df in dfs.items():
        for quadrant in RUSSELL_QUADRANTS:
            if 'russell_quadrant' in df.columns:
                val = df[df['russell_quadrant'] == quadrant][metric].dropna().mean()
                if pd.notna(val):
                    all_values.append(val)

    vmin, vmax = (min(all_values), max(all_values)) if all_values else (0, 1)

    im = None
    for idx, (scheduler_name, df) in enumerate(dfs.items()):
        row_idx = idx // cols
        col_idx = idx % cols
        ax = axes[row_idx, col_idx]
        short_name = short_names[idx]

        # Build 2x2 matrix following Russell layout
        matrix = np.zeros((2, 2))
        for quadrant in RUSSELL_QUADRANTS:
            grid_row, grid_col = QUADRANT_GRID_LAYOUT[quadrant]
            if 'russell_quadrant' in df.columns:
                val = df[df['russell_quadrant'] == quadrant][metric].dropna().mean()
                matrix[grid_row, grid_col] = val if pd.notna(val) else 0
            else:
                matrix[grid_row, grid_col] = 0

        # Plot heatmap
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='equal', vmin=vmin, vmax=vmax)

        # Labels
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Neg V', 'Pos V'])
        ax.set_yticklabels(['High A', 'Low A'])
        ax.set_title(short_name, fontweight='bold')

        # Add quadrant labels and values
        for quadrant in RUSSELL_QUADRANTS:
            grid_row, grid_col = QUADRANT_GRID_LAYOUT[quadrant]
            val = matrix[grid_row, grid_col]
            text_color = 'white' if val > (vmax - vmin) * 0.6 + vmin else 'black'
            # Highlight depression quadrant
            fontweight = 'bold' if quadrant == 'depression' else 'normal'
            label = f'{quadrant[:3].upper()}\n{val:.2f}'
            ax.text(grid_col, grid_row, label, ha='center', va='center',
                    color=text_color, fontsize=10, fontweight=fontweight)

    # Hide unused subplots
    for idx in range(num_schedulers, rows * cols):
        row_idx = idx // cols
        col_idx = idx % cols
        axes[row_idx, col_idx].axis('off')

    # Add colorbar
    if im is not None:
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax, label=metric.replace('_', ' ').title())

    fig.suptitle(f'Russell Quadrant {metric.replace("_", " ").title()} Comparison\n'
                 f'(Layout: High/Low Arousal x Neg/Pos Valence)',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved quadrant comparison heatmap to: {output_path}")
    plt.close()


def plot_affect_weight_distribution(dfs: Dict[str, pd.DataFrame],
                                     output_path: str = 'affect_weight_distribution.png'):
    """
    Plot affect weight distribution by Russell quadrant across schedulers.

    Uses box plots to compare how weights vary across emotional states.
    Only meaningful for AW-SSJF and Weight-Only schedulers.

    Args:
        dfs: Dictionary mapping scheduler_name -> DataFrame
        output_path: Path to save figure
    """
    schedulers = list(dfs.keys())
    short_names = [_shorten_scheduler_name(s) for s in schedulers]
    num_dfs = len(dfs)
    cols = min(num_dfs, 2)
    rows = (num_dfs + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 6*rows), sharey=True)

    if num_dfs == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (scheduler_name, df) in enumerate(dfs.items()):
        row_idx = idx // cols
        col_idx = idx % cols
        ax = axes[row_idx, col_idx]
        short_name = short_names[idx]

        if 'affect_weight' not in df.columns or 'russell_quadrant' not in df.columns:
            ax.text(0.5, 0.5, 'No affect_weight data', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(short_name)
            continue

        # Prepare data for box plot
        data_by_quadrant = []
        for quadrant in RUSSELL_QUADRANTS:
            weights = df[df['russell_quadrant'] == quadrant]['affect_weight'].dropna()
            data_by_quadrant.append(weights if len(weights) > 0 else pd.Series([1.0]))

        bp = ax.boxplot(data_by_quadrant, labels=[q[:3].upper() for q in RUSSELL_QUADRANTS],
                        patch_artist=True)

        # Color by quadrant
        for patch, quadrant in zip(bp['boxes'], RUSSELL_QUADRANTS):
            patch.set_facecolor(QUADRANT_COLORS[quadrant])
            patch.set_alpha(0.7)

        ax.set_ylabel('Affect Weight' if col_idx == 0 else '')
        ax.set_title(short_name)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline (w=1)')
        ax.grid(axis='y', alpha=0.3)

    # Hide unused subplots
    for idx in range(num_dfs, rows * cols):
        row_idx = idx // cols
        col_idx = idx % cols
        axes[row_idx, col_idx].axis('off')

    fig.suptitle('Affect Weight Distribution by Quadrant', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved affect weight distribution to: {output_path}")
    plt.close()


def plot_urgency_vs_waiting_time(df: pd.DataFrame,
                                  scheduler_name: str = None,
                                  output_path: str = 'urgency_waiting_scatter.png'):
    """
    Scatter plot of urgency vs waiting time, colored by Russell quadrant.

    Shows whether high-urgency (depression) jobs receive lower waiting times.

    Args:
        df: DataFrame with urgency and waiting_time columns
        scheduler_name: Name of scheduler for title
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    if 'urgency' not in df.columns or 'russell_quadrant' not in df.columns:
        ax.text(0.5, 0.5, 'No urgency data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return

    for quadrant in RUSSELL_QUADRANTS:
        quadrant_df = df[df['russell_quadrant'] == quadrant]
        ax.scatter(
            quadrant_df['urgency'],
            quadrant_df['waiting_time'],
            c=QUADRANT_COLORS[quadrant],
            label=quadrant.capitalize(),
            alpha=0.6,
            s=50
        )

    ax.set_xlabel('Urgency Score')
    ax.set_ylabel('Waiting Time')
    title = 'Urgency vs Waiting Time'
    if scheduler_name:
        title += f' ({scheduler_name})'
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    # Add trend line
    valid_df = df.dropna(subset=['urgency', 'waiting_time'])
    if len(valid_df) > 2:
        z = np.polyfit(valid_df['urgency'], valid_df['waiting_time'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, 1, 100)
        ax.plot(x_line, p(x_line), 'k--', alpha=0.5, linewidth=2)
        ax.annotate(f'Slope: {z[0]:.2f}', xy=(0.7, 0.9), xycoords='axes fraction',
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved urgency vs waiting time plot to: {output_path}")
    plt.close()


def plot_depression_first_effectiveness(dfs: Dict[str, pd.DataFrame],
                                         metric: str = 'waiting_time',
                                         output_path: str = 'depression_first_effectiveness.png'):
    """
    Compare depression quadrant performance relative to other quadrants.

    Shows how effectively each scheduler prioritizes depression-state users.
    Lower ratio means depression users wait less relative to others.

    Args:
        dfs: Dictionary mapping scheduler_name -> DataFrame
        metric: Metric to compare
        output_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    scheduler_names = []
    short_names = []
    depression_avgs = []
    other_avgs = []
    depression_ratios = []

    for scheduler_name, df in dfs.items():
        if 'russell_quadrant' not in df.columns:
            continue

        dep_data = df[df['russell_quadrant'] == 'depression'][metric].dropna()
        other_data = df[df['russell_quadrant'] != 'depression'][metric].dropna()

        if len(dep_data) == 0 or len(other_data) == 0:
            continue

        dep_avg = dep_data.mean()
        other_avg = other_data.mean()
        ratio = dep_avg / other_avg if other_avg > 0 else 1.0

        scheduler_names.append(scheduler_name)
        short_names.append(_shorten_scheduler_name(scheduler_name))
        depression_avgs.append(dep_avg)
        other_avgs.append(other_avg)
        depression_ratios.append(ratio)

    if not scheduler_names:
        ax.text(0.5, 0.5, 'No quadrant data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return

    x = np.arange(len(scheduler_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, depression_avgs, width, label='Depression Quadrant',
                   color=QUADRANT_COLORS['depression'], alpha=0.8)
    bars2 = ax.bar(x + width/2, other_avgs, width, label='Other Quadrants',
                   color='#888888', alpha=0.8)

    ax.set_ylabel(f'Average {metric.replace("_", " ").title()}')
    ax.set_xlabel('Scheduler')
    ax.set_title('Depression-First Effectiveness\n(Lower depression bar = better prioritization)')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add ratio annotations
    for i, ratio in enumerate(depression_ratios):
        max_val = max(depression_avgs[i], other_avgs[i])
        color = 'green' if ratio < 1.0 else 'red'
        ax.annotate(f'Ratio: {ratio:.2f}',
                    xy=(i, max_val + 0.05 * max_val),
                    ha='center', va='bottom', fontsize=9, color=color,
                    fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved depression-first effectiveness plot to: {output_path}")
    plt.close()


def plot_quadrant_jct_improvement(dfs: Dict[str, pd.DataFrame],
                                   baseline_key: str = None,
                                   output_path: str = 'quadrant_jct_improvement.png'):
    """
    Plot per-quadrant JCT improvement relative to FCFS baseline.

    Args:
        dfs: Dictionary mapping scheduler_name -> DataFrame
        baseline_key: Key for baseline scheduler (auto-detected if None)
        output_path: Path to save figure
    """
    # Auto-detect baseline
    if baseline_key is None:
        for key in dfs.keys():
            if 'FCFS' in key.upper():
                baseline_key = key
                break
        if baseline_key is None:
            baseline_key = list(dfs.keys())[0]

    if baseline_key not in dfs:
        print(f"Warning: Baseline '{baseline_key}' not found.")
        return

    baseline_df = dfs[baseline_key]

    # Compute baseline per-quadrant JCT
    baseline_jct = {}
    for quadrant in RUSSELL_QUADRANTS:
        if 'russell_quadrant' in baseline_df.columns:
            q_data = baseline_df[baseline_df['russell_quadrant'] == quadrant]['turnaround_time'].dropna()
            baseline_jct[quadrant] = q_data.mean() if len(q_data) > 0 else 0
        else:
            baseline_jct[quadrant] = 0

    # Compute improvement for each scheduler
    fig, ax = plt.subplots(figsize=(12, 6))

    other_schedulers = [k for k in dfs.keys() if k != baseline_key]
    short_names = [_shorten_scheduler_name(s) for s in other_schedulers]
    baseline_short = _shorten_scheduler_name(baseline_key)

    if not other_schedulers:
        ax.text(0.5, 0.5, 'Only baseline scheduler available', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return

    x = np.arange(len(RUSSELL_QUADRANTS))
    width = 0.8 / len(other_schedulers)
    colors = sns.color_palette("husl", len(other_schedulers))

    for scheduler_idx, scheduler_name in enumerate(other_schedulers):
        df = dfs[scheduler_name]
        short_name = short_names[scheduler_idx]
        improvements = []

        for quadrant in RUSSELL_QUADRANTS:
            if 'russell_quadrant' in df.columns:
                q_data = df[df['russell_quadrant'] == quadrant]['turnaround_time'].dropna()
                scheduler_jct = q_data.mean() if len(q_data) > 0 else 0
            else:
                scheduler_jct = 0

            if baseline_jct[quadrant] > 0:
                improvement = (baseline_jct[quadrant] - scheduler_jct) / baseline_jct[quadrant] * 100
            else:
                improvement = 0
            improvements.append(improvement)

        offset = (scheduler_idx - (len(other_schedulers) - 1) / 2) * width
        bars = ax.bar(x + offset, improvements, width, label=short_name,
                      color=colors[scheduler_idx], alpha=0.8)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel(f'JCT Improvement vs {baseline_short} (%)')
    ax.set_xlabel('Russell Quadrant')
    ax.set_title(f'Per-Quadrant JCT Improvement Relative to {baseline_short}\n(Positive = Better)')
    ax.set_xticks(x)
    ax.set_xticklabels([q.capitalize() for q in RUSSELL_QUADRANTS])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved quadrant JCT improvement plot to: {output_path}")
    plt.close()


def plot_serving_time_comparison_heatmap(dfs: Dict[str, pd.DataFrame],
                                          output_path: str = 'serving_time_heatmap.png',
                                          use_quadrants: bool = None):
    """
    Create side-by-side heatmaps comparing predicted vs actual serving time
    across schedulers and Russell quadrants (or legacy emotion classes).

    Args:
        dfs: Dictionary mapping scheduler_name -> DataFrame with job logs
        output_path: Path to save the figure
        use_quadrants: If None, auto-detect. If True, use Russell quadrants. If False, use legacy.
    """
    # Prepare data
    schedulers = list(dfs.keys())
    # Shorten scheduler names for display
    short_names = [_shorten_scheduler_name(s) for s in schedulers]

    # Calculate figure width based on longest scheduler name
    max_name_len = max(len(s) for s in short_names) if short_names else 10
    left_margin = max(0.15, min(0.35, max_name_len * 0.012))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(6, len(schedulers) * 0.8)))

    # Auto-detect format
    sample_df = list(dfs.values())[0]
    if use_quadrants is None:
        use_quadrants = 'russell_quadrant' in sample_df.columns

    if use_quadrants:
        categories = RUSSELL_QUADRANTS
        category_col = 'russell_quadrant'
        x_labels = [q.capitalize() for q in categories]
        subtitle = 'by Scheduler and Russell Quadrant'
    else:
        categories = LEGACY_EMOTION_CLASSES
        category_col = 'emotion_class'
        x_labels = [f'{ec.capitalize()} Arousal' for ec in categories]
        subtitle = 'by Scheduler and Emotion Class'

    # Build data matrices for predicted and actual serving time
    predicted_matrix = []
    actual_matrix = []

    for scheduler_name in schedulers:
        df = dfs[scheduler_name]
        predicted_row = []
        actual_row = []

        for category in categories:
            if category_col in df.columns:
                class_data = df[df[category_col] == category]
            else:
                class_data = pd.DataFrame()

            # Predicted service time
            if 'predicted_serving_time' in df.columns:
                predicted_data = class_data['predicted_serving_time'].dropna()
                predicted_avg = predicted_data.mean() if len(predicted_data) > 0 else 0
            else:
                predicted_avg = 0
            predicted_row.append(predicted_avg)

            # Actual execution duration
            if 'actual_serving_time' in df.columns:
                actual_data = class_data['actual_serving_time'].dropna()
                actual_avg = actual_data.mean() if len(actual_data) > 0 else 0
            else:
                actual_avg = 0
            actual_row.append(actual_avg)

        predicted_matrix.append(predicted_row)
        actual_matrix.append(actual_row)

    predicted_matrix = np.array(predicted_matrix)
    actual_matrix = np.array(actual_matrix)

    # Use same color scale for both heatmaps
    all_vals = np.concatenate([predicted_matrix.flatten(), actual_matrix.flatten()])
    vmin = all_vals.min() if all_vals.size > 0 else 0
    vmax = all_vals.max() if all_vals.size > 0 else 1

    # Plot predicted serving time
    im1 = ax1.imshow(predicted_matrix, cmap='YlOrRd', aspect='auto', vmin=vmin, vmax=vmax)
    ax1.set_xticks(range(len(categories)))
    ax1.set_yticks(range(len(schedulers)))
    ax1.set_xticklabels(x_labels)
    ax1.set_yticklabels(short_names, fontsize=9)
    ax1.set_title('Predicted Service Time', fontsize=12, fontweight='bold')

    # Add text annotations for predicted
    for i in range(len(schedulers)):
        for j in range(len(categories)):
            val = predicted_matrix[i, j]
            text_color = 'white' if val > (vmax - vmin) * 0.6 + vmin else 'black'
            ax1.text(j, i, f'{val:.2f}', ha="center", va="center", color=text_color, fontsize=10)

    # Plot actual serving time
    im2 = ax2.imshow(actual_matrix, cmap='YlOrRd', aspect='auto', vmin=vmin, vmax=vmax)
    ax2.set_xticks(range(len(categories)))
    ax2.set_yticks(range(len(schedulers)))
    ax2.set_xticklabels(x_labels)
    ax2.set_yticklabels(short_names, fontsize=9)
    ax2.set_title('Actual Execution Duration', fontsize=12, fontweight='bold')

    # Add text annotations for actual
    for i in range(len(schedulers)):
        for j in range(len(categories)):
            val = actual_matrix[i, j]
            text_color = 'white' if val > (vmax - vmin) * 0.6 + vmin else 'black'
            ax2.text(j, i, f'{val:.2f}', ha="center", va="center", color=text_color, fontsize=10)

    # Adjust layout to prevent overlap
    plt.subplots_adjust(left=left_margin, right=0.85, wspace=0.3)

    # Add shared colorbar
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax, label='Time (seconds)')

    # Overall title
    fig.suptitle(f'Serving Time Comparison: Predicted vs Actual\n{subtitle}',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved serving time comparison heatmap to: {output_path}")
    plt.close()


def generate_comprehensive_report(result_dirs: Dict[str, str],
                                   output_dir: str = 'results/plots/'):
    """
    Generate comprehensive visualization report from multiple experiment results.

    Auto-detects data format (Russell quadrants vs legacy emotion classes) and
    generates appropriate visualizations.

    Args:
        result_dirs: Dictionary mapping scheduler_name -> path_to_job_logs_csv
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load all data
    dfs = {}
    for scheduler_name, csv_path in result_dirs.items():
        try:
            dfs[scheduler_name] = load_job_logs(csv_path)
        except Exception as e:
            print(f"Warning: Could not load {csv_path}: {e}")

    if not dfs:
        print("Error: No data loaded")
        return

    # Check data structure and determine if using new format
    sample_df = list(dfs.values())[0]
    using_quadrants = 'russell_quadrant' in sample_df.columns

    print(f"\nGenerating comprehensive visualization report...")
    print(f"Data format: {'Russell Quadrants' if using_quadrants else 'Legacy Emotion Classes'}")
    print(f"Output directory: {output_dir}")
    print(f"Schedulers: {list(dfs.keys())}")

    # 1. Completion curves (works for both formats)
    plot_completion_curves(dfs, os.path.join(output_dir, 'completion_curves.png'))

    # 2. Latency CDF (works for both formats)
    plot_latency_cdf(dfs, metric='turnaround_time',
                     output_path=os.path.join(output_dir, 'latency_cdf.png'))

    if using_quadrants:
        # New quadrant-based visualizations

        # 3. Per-quadrant metrics for each scheduler
        for scheduler_name, df in dfs.items():
            plot_per_quadrant_metrics(
                df, metric='waiting_time', scheduler_name=scheduler_name,
                output_path=os.path.join(output_dir, f'quadrant_metrics_{scheduler_name}.png')
            )

        # 4. Quadrant comparison heatmap (Russell 2x2 layout)
        plot_quadrant_comparison_heatmap(
            dfs, metric='waiting_time',
            output_path=os.path.join(output_dir, 'quadrant_heatmap.png')
        )

        # 5. Affect weight distribution
        plot_affect_weight_distribution(
            dfs, output_path=os.path.join(output_dir, 'affect_weight_distribution.png')
        )

        # 6. Urgency vs waiting time (for affect-aware schedulers)
        for scheduler_name, df in dfs.items():
            if 'urgency' in df.columns:
                plot_urgency_vs_waiting_time(
                    df, scheduler_name=scheduler_name,
                    output_path=os.path.join(output_dir, f'urgency_waiting_{scheduler_name}.png')
                )

        # 7. Depression-first effectiveness
        plot_depression_first_effectiveness(
            dfs, metric='waiting_time',
            output_path=os.path.join(output_dir, 'depression_first_effectiveness.png')
        )

        # 8. Per-quadrant JCT improvement
        plot_quadrant_jct_improvement(
            dfs, output_path=os.path.join(output_dir, 'quadrant_jct_improvement.png')
        )
    else:
        # Legacy visualizations (kept for backward compatibility)
        for scheduler_name, df in dfs.items():
            plot_per_emotion_class_metrics_legacy(
                df, metric='waiting_time', scheduler_name=scheduler_name,
                output_path=os.path.join(output_dir, f'emotion_metrics_{scheduler_name}.png')
            )

        plot_emotion_class_comparison_heatmap_legacy(
            dfs, metric='waiting_time',
            output_path=os.path.join(output_dir, 'emotion_heatmap.png')
        )

    # 9. Serving time comparison (auto-detects format)
    plot_serving_time_comparison_heatmap(
        dfs,
        output_path=os.path.join(output_dir, 'serving_time_heatmap.png')
    )

    print(f"\nComprehensive report generated successfully!")


if __name__ == '__main__':
    # Allow specifying the scan directory from command line
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Generate visualization plots for emotion-aware LLM scheduling results.\n"
                    "Supports both Russell quadrant format (new) and legacy emotion class format.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="results/llm_runs",
        help="Directory to scan for '*_jobs.csv' files (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for plots (default: runs_dir/plots).",
    )
    args = parser.parse_args()

    # Auto-scan given (or default) llm_runs folder and generate all plots
    scan_llm_runs_and_generate_plots(runs_dir=args.runs_dir, output_dir=args.output_dir)
