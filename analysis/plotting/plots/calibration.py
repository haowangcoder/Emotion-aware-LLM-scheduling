"""
Calibration Plots (Predicted vs Actual)

These plots validate the prediction model's reliability.
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..constants import AROUSAL_ORDER, AROUSAL_LABELS, AROUSAL_COLORS
from ..utils import save_figure


def plot_calibration_scatter(dfs: Dict[str, pd.DataFrame], output_path: str,
                              title: str = None, formats: List[str] = None):
    """
    Scatter plot of predicted vs actual serving time.

    Points are colored by arousal class with y=x reference line.

    Args:
        dfs: Dict mapping scheduler name -> DataFrame
        output_path: Output file path
        title: Plot title
    """
    # Use any scheduler's data (predictions should be similar)
    df = None
    for sched in ['FCFS', 'SSJF-Emotion', 'SSJF-Combined', 'SSJF-Valence']:
        if sched in dfs:
            df = dfs[sched]
            break

    if df is None or 'predicted_serving_time' not in df.columns:
        print("  Warning: No data for calibration scatter plot")
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    for arousal in AROUSAL_ORDER:
        mask = df['emotion_class'] == arousal
        subset = df[mask]

        if len(subset) == 0:
            continue

        ax.scatter(
            subset['predicted_serving_time'],
            subset['actual_serving_time'],
            c=AROUSAL_COLORS.get(arousal, '#888888'),
            s=80,
            alpha=0.7,
            edgecolors='white',
            linewidths=0.5,
            label=AROUSAL_LABELS[arousal],
            marker='o'
        )

    # Add y=x reference line
    lims = [
        max(ax.get_xlim()[0], ax.get_ylim()[0]),
        min(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='Perfect Calibration')

    ax.set_xlabel('Predicted Serving Time (s)')
    ax.set_ylabel('Actual Serving Time (s)')

    if title:
        ax.set_title(title, fontweight='bold', pad=10)

    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    save_figure(fig, output_path, formats)
    plt.close()


def plot_calibration_binned(dfs: Dict[str, pd.DataFrame], output_path: str,
                             title: str = None, formats: List[str] = None):
    """
    Binned calibration plot: mean actual vs predicted bins.

    Groups predictions into bins and shows mean actual +/- std.

    Args:
        dfs: Dict mapping scheduler name -> DataFrame
        output_path: Output file path
        title: Plot title
    """
    df = None
    for sched in ['FCFS', 'SSJF-Emotion', 'SSJF-Combined', 'SSJF-Valence']:
        if sched in dfs:
            df = dfs[sched]
            break

    if df is None or 'predicted_serving_time' not in df.columns:
        print("  Warning: No data for binned calibration plot")
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    df_valid = df.dropna(subset=['predicted_serving_time', 'actual_serving_time'])
    unique_pred = sorted(df_valid['predicted_serving_time'].unique())

    if len(unique_pred) < 2:
        print("  Warning: Not enough unique predicted values for binning")
        plt.close()
        return

    grouped = df_valid.groupby('predicted_serving_time')['actual_serving_time'].agg(['mean', 'std', 'count'])

    x = grouped.index
    y_mean = grouped['mean']
    y_std = grouped['std'].fillna(0)

    ax.errorbar(
        x, y_mean,
        yerr=y_std,
        fmt='o-',
        color='#0072B2',
        markersize=10,
        markerfacecolor='#0072B2',
        markeredgecolor='white',
        markeredgewidth=1.5,
        capsize=5,
        capthick=2,
        elinewidth=2,
        label='Mean Actual (+/- Std)',
        alpha=0.9
    )

    # Add y=x reference line
    lims = [min(x.min(), y_mean.min()) * 0.9, max(x.max(), y_mean.max()) * 1.1]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=2, label='Perfect Calibration')

    ax.set_xlabel('Predicted Serving Time (s)')
    ax.set_ylabel('Actual Serving Time (s)')

    if title:
        ax.set_title(title, fontweight='bold', pad=10)

    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_path, formats)
    plt.close()


def plot_error_distribution(dfs: Dict[str, pd.DataFrame], output_path: str,
                             title: str = None, formats: List[str] = None):
    """
    Violin plot of prediction errors by arousal class.

    Shows the distribution of relative errors for each arousal level.

    Args:
        dfs: Dict mapping scheduler name -> DataFrame
        output_path: Output file path
        title: Plot title
    """
    df = None
    for sched in ['FCFS', 'SSJF-Emotion', 'SSJF-Combined', 'SSJF-Valence']:
        if sched in dfs:
            df = dfs[sched]
            break

    if df is None or 'predicted_serving_time' not in df.columns:
        print("  Warning: No data for error distribution plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    df_valid = df.dropna(subset=['predicted_serving_time', 'actual_serving_time']).copy()
    df_valid['relative_error'] = (
        (df_valid['actual_serving_time'] - df_valid['predicted_serving_time'])
        / df_valid['predicted_serving_time']
    ) * 100  # As percentage

    data_by_class = []
    labels = []
    colors = []

    for arousal in AROUSAL_ORDER:
        mask = df_valid['emotion_class'] == arousal
        if mask.sum() > 0:
            data_by_class.append(df_valid.loc[mask, 'relative_error'].values)
            labels.append(AROUSAL_LABELS[arousal])
            colors.append(AROUSAL_COLORS.get(arousal, '#888888'))

    if not data_by_class:
        print("  Warning: No valid data for error distribution plot")
        plt.close()
        return

    parts = ax.violinplot(data_by_class, positions=range(len(data_by_class)),
                          showmeans=True, showmedians=True, showextrema=True)

    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
        if partname in parts:
            parts[partname].set_edgecolor('black')
            parts[partname].set_linewidth(1)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel('Arousal Class')
    ax.set_ylabel('Relative Prediction Error (%)')

    if title:
        ax.set_title(title, fontweight='bold', pad=10)

    ax.grid(axis='y', alpha=0.3)

    for i, data in enumerate(data_by_class):
        mean_val = np.mean(data)
        ax.annotate(
            f'Mean: {mean_val:.1f}%',
            xy=(i, ax.get_ylim()[1]),
            xytext=(0, -5),
            textcoords='offset points',
            ha='center',
            va='top',
            fontsize=9,
            fontweight='bold'
        )

    plt.tight_layout()
    save_figure(fig, output_path, formats)
    plt.close()


def generate_calibration_plots(dfs: Dict[str, pd.DataFrame], output_dir: str,
                                formats: List[str] = None):
    """Generate all calibration plots."""
    print("\n[E] Generating Calibration Plots...")
    cal_dir = Path(output_dir) / 'calibration'

    plot_calibration_scatter(dfs, str(cal_dir / 'calibration_scatter'),
                             title='Predicted vs Actual Serving Time', formats=formats)

    plot_calibration_binned(dfs, str(cal_dir / 'calibration_binned'),
                            title='Binned Calibration: Mean Actual by Predicted', formats=formats)

    plot_error_distribution(dfs, str(cal_dir / 'error_distribution'),
                            title='Prediction Error Distribution by Arousal Class', formats=formats)
