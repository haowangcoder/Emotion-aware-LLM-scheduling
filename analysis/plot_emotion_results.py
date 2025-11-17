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
        metrics = ['avg_waiting_time', 'p99_waiting_time', 'avg_turnaround_time', 'throughput']

    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(5*num_metrics, 5))

    if num_metrics == 1:
        axes = [axes]

    schedulers = list(results.keys())

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        values = [results[sched].get(metric, 0) for sched in schedulers]
        colors = sns.color_palette("husl", len(schedulers))

        bars = ax.bar(range(len(schedulers)), values, color=colors, alpha=0.8)
        ax.set_xticks(range(len(schedulers)))
        ax.set_xticklabels(schedulers, rotation=45, ha='right')
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


def plot_fairness_comparison(fairness_results: Dict[str, float],
                              output_path: str = 'fairness_comparison.png'):
    """
    Plot Jain Fairness Index comparison across schedulers

    Args:
        fairness_results: Dictionary mapping scheduler_name -> jain_index
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    schedulers = list(fairness_results.keys())
    jain_indices = list(fairness_results.values())

    colors = sns.color_palette("RdYlGn", len(schedulers))
    bars = ax.barh(range(len(schedulers)), jain_indices, color=colors, alpha=0.8)

    ax.set_yticks(range(len(schedulers)))
    ax.set_yticklabels(schedulers)
    ax.set_xlabel('Jain Fairness Index')
    ax.set_title('Fairness Comparison Across Schedulers\n(Higher is Better, 1.0 = Perfect Fairness)')
    ax.set_xlim(0, 1.05)
    ax.axvline(x=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect Fairness')
    ax.grid(axis='x', alpha=0.3)
    ax.legend()

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                f'{width:.4f}',
                ha='left', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved fairness comparison plot to: {output_path}")
    plt.close()


def _compute_scheduler_metrics_from_df(df: pd.DataFrame) -> Dict[str, float]:
    """Compute core metrics from a jobs CSV DataFrame."""
    waiting = df['waiting_time'].dropna().to_numpy()
    turnaround = df['turnaround_time'].dropna().to_numpy()
    finish_times = df['finish_time'].dropna().to_numpy()

    metrics = {
        'avg_waiting_time': float(np.mean(waiting)) if waiting.size else 0.0,
        'p99_waiting_time': float(np.percentile(waiting, 99)) if waiting.size else 0.0,
        'avg_turnaround_time': float(np.mean(turnaround)) if turnaround.size else 0.0,
        'throughput': float(len(df) / finish_times.max()) if finish_times.size and finish_times.max() > 0 else 0.0,
    }
    return metrics


def _compute_jain_fairness_from_df(df: pd.DataFrame) -> float:
    """Compute Jain Fairness Index over per-emotion-class average waiting time."""
    if 'emotion_class' not in df.columns:
        return 0.0
    grp = df.dropna(subset=['waiting_time']).groupby('emotion_class')['waiting_time'].mean()
    vals = grp.to_numpy()
    if vals.size == 0:
        return 0.0
    numerator = (vals.sum() ** 2)
    denominator = (len(vals) * (vals ** 2).sum())
    return float(numerator / denominator) if denominator > 0 else 0.0


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
        fairness_results[key] = _compute_jain_fairness_from_df(df)

    # Generate plots
    plot_scheduler_comparison_barplot(
        scheduler_results,
        output_path=os.path.join(plots_dir, 'scheduler_comparison.png')
    )
    plot_fairness_comparison(
        fairness_results,
        output_path=os.path.join(plots_dir, 'fairness_comparison.png')
    )
    print(f"Comparison plots saved to: {plots_dir}")

    # Generate comprehensive report plots (completion curves, CDF, heatmaps, etc.)
    result_dirs = {
        os.path.splitext(os.path.basename(path))[0]: path
        for path in csv_paths
    }
    generate_comprehensive_report(result_dirs, output_dir=plots_dir)


def plot_per_emotion_class_metrics(df: pd.DataFrame,
                                    metric: str = 'waiting_time',
                                    scheduler_name: str = None,
                                    output_path: str = 'per_emotion_metrics.png'):
    """
    Plot metrics broken down by emotion class

    Args:
        df: DataFrame with job logs including emotion_class column
        metric: Metric to plot ('waiting_time', 'turnaround_time', or 'service_time')
        scheduler_name: Name of scheduler for title
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by emotion class
    emotion_classes = ['high', 'medium', 'low']
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


def plot_emotion_class_comparison_heatmap(dfs: Dict[str, pd.DataFrame],
                                           metric: str = 'waiting_time',
                                           output_path: str = 'emotion_heatmap.png'):
    """
    Create heatmap comparing average metric across schedulers and emotion classes

    Args:
        dfs: Dictionary mapping scheduler_name -> DataFrame with job logs
        metric: Metric to plot
        output_path: Path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data matrix
    schedulers = list(dfs.keys())
    emotion_classes = ['high', 'medium', 'low']

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
    ax.set_yticklabels(schedulers)

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


def plot_serving_time_comparison_heatmap(dfs: Dict[str, pd.DataFrame],
                                          output_path: str = 'serving_time_heatmap.png'):
    """
    Create side-by-side heatmaps comparing predicted vs actual serving time
    across schedulers and emotion classes

    Args:
        dfs: Dictionary mapping scheduler_name -> DataFrame with job logs
        output_path: Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Prepare data
    schedulers = list(dfs.keys())
    emotion_classes = ['high', 'medium', 'low']

    # Build data matrices for predicted and actual serving time
    predicted_matrix = []
    actual_matrix = []

    for scheduler_name in schedulers:
        df = dfs[scheduler_name]
        predicted_row = []
        actual_row = []

        for emotion_class in emotion_classes:
            class_data = df[df['emotion_class'] == emotion_class]

            # Predicted service time
            predicted_data = class_data['service_time'].dropna()
            predicted_avg = predicted_data.mean() if len(predicted_data) > 0 else 0
            predicted_row.append(predicted_avg)

            # Actual execution duration
            actual_data = class_data['actual_execution_duration'].dropna()
            actual_avg = actual_data.mean() if len(actual_data) > 0 else 0
            actual_row.append(actual_avg)

        predicted_matrix.append(predicted_row)
        actual_matrix.append(actual_row)

    predicted_matrix = np.array(predicted_matrix)
    actual_matrix = np.array(actual_matrix)

    # Use same color scale for both heatmaps
    vmin = min(predicted_matrix.min(), actual_matrix.min())
    vmax = max(predicted_matrix.max(), actual_matrix.max())

    # Plot predicted serving time
    im1 = ax1.imshow(predicted_matrix, cmap='YlOrRd', aspect='auto', vmin=vmin, vmax=vmax)
    ax1.set_xticks(range(len(emotion_classes)))
    ax1.set_yticks(range(len(schedulers)))
    ax1.set_xticklabels([f'{ec.capitalize()} Arousal' for ec in emotion_classes])
    ax1.set_yticklabels(schedulers)
    ax1.set_title('Predicted Service Time', fontsize=12, fontweight='bold')

    # Add text annotations for predicted
    for i in range(len(schedulers)):
        for j in range(len(emotion_classes)):
            text = ax1.text(j, i, f'{predicted_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)

    # Plot actual serving time
    im2 = ax2.imshow(actual_matrix, cmap='YlOrRd', aspect='auto', vmin=vmin, vmax=vmax)
    ax2.set_xticks(range(len(emotion_classes)))
    ax2.set_yticks(range(len(schedulers)))
    ax2.set_xticklabels([f'{ec.capitalize()} Arousal' for ec in emotion_classes])
    ax2.set_yticklabels(schedulers)
    ax2.set_title('Actual Execution Duration', fontsize=12, fontweight='bold')

    # Add text annotations for actual
    for i in range(len(schedulers)):
        for j in range(len(emotion_classes)):
            text = ax2.text(j, i, f'{actual_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=10)

    # Add shared colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax, label='Time (seconds)')

    # Overall title
    fig.suptitle('Serving Time Comparison: Predicted vs Actual\nby Scheduler and Emotion Class',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved serving time comparison heatmap to: {output_path}")
    plt.close()


def generate_comprehensive_report(result_dirs: Dict[str, str],
                                   output_dir: str = 'results/plots/'):
    """
    Generate comprehensive visualization report from multiple experiment results

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

    print(f"\nGenerating comprehensive visualization report...")
    print(f"Output directory: {output_dir}")
    print(f"Schedulers: {list(dfs.keys())}")

    # 1. Completion curves
    plot_completion_curves(dfs, os.path.join(output_dir, 'completion_curves.png'))

    # 2. Latency CDF
    plot_latency_cdf(dfs, metric='turnaround_time',
                     output_path=os.path.join(output_dir, 'latency_cdf.png'))

    # 3. Per-emotion class metrics for each scheduler
    for scheduler_name, df in dfs.items():
        plot_per_emotion_class_metrics(
            df, metric='waiting_time', scheduler_name=scheduler_name,
            output_path=os.path.join(output_dir, f'emotion_metrics_{scheduler_name}.png')
        )

    # 4. Emotion class comparison heatmap
    plot_emotion_class_comparison_heatmap(
        dfs, metric='waiting_time',
        output_path=os.path.join(output_dir, 'emotion_heatmap.png')
    )

    # 5. Serving time comparison heatmap (predicted vs actual)
    plot_serving_time_comparison_heatmap(
        dfs,
        output_path=os.path.join(output_dir, 'serving_time_heatmap.png')
    )

    print(f"\nComprehensive report generated successfully!")


if __name__ == '__main__':
    # Allow specifying the scan directory from command line
    import argparse

    parser = argparse.ArgumentParser(
        description="Scan LLM scheduling results and generate visualization plots."
    )
    parser.add_argument(
        "--runs-dir",
        type=str,
        default="results/llm_runs",
        help="Directory to scan for '*_jobs.csv' files (default: %(default)s).",
    )
    args = parser.parse_args()

    # Auto-scan given (or default) llm_runs folder and generate all plots
    scan_llm_runs_and_generate_plots(runs_dir=args.runs_dir)
