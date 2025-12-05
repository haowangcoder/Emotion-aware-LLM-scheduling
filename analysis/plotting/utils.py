"""
Utility functions for publication-quality plotting.

Includes style setup, figure saving, and label helpers.
"""

from pathlib import Path
from typing import List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def setup_publication_style(dpi: int = 300):
    """
    Configure matplotlib for publication-quality figures.

    Args:
        dpi: Default DPI for saved figures (applied via rcParams['savefig.dpi']).
    """
    plt.rcParams.update({
        # Figure
        'figure.figsize': (8, 5),
        'figure.dpi': 150,
        'figure.facecolor': 'white',

        # Font
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,

        # Lines
        'lines.linewidth': 2,
        'lines.markersize': 8,

        # Axes
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'axes.axisbelow': True,

        # Grid
        'grid.alpha': 0.3,
        'grid.linestyle': '-',
        'grid.linewidth': 0.5,

        # Legend
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',

        # Savefig
        'savefig.dpi': dpi,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })
    sns.set_style("whitegrid")


def save_figure(fig, output_path: str, formats: List[str] = None):
    """
    Save figure in multiple formats.

    Args:
        fig: Matplotlib figure object
        output_path: Base path for output (without extension)
        formats: List of formats to save (default: ['pdf', 'png'])
    """
    if formats is None:
        formats = ['pdf', 'png']

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        save_path = output_path.with_suffix(f'.{fmt}')
        # Use rcParams['savefig.dpi'] configured via setup_publication_style.
        fig.savefig(save_path, format=fmt, bbox_inches='tight')
        print(f"  Saved: {save_path}")


def add_direct_labels(ax, lines, labels, x_offset=0.02, fontsize=10):
    """
    Add direct labels at line ends instead of using legend.

    This is the "top-conference style" approach where labels appear
    at the end of each line rather than in a separate legend box.

    Args:
        ax: Matplotlib axes object
        lines: List of line objects
        labels: List of label strings
        x_offset: Horizontal offset for labels (in fontsize units)
        fontsize: Font size for labels
    """
    for line, label in zip(lines, labels):
        # Get last point
        xdata, ydata = line.get_xdata(), line.get_ydata()
        if len(xdata) > 0:
            x_end = xdata[-1]
            y_end = ydata[-1]
            color = line.get_color()

            # Add text label
            ax.annotate(
                label,
                xy=(x_end, y_end),
                xytext=(x_offset, 0),
                textcoords='offset fontsize',
                fontsize=fontsize,
                color=color,
                va='center',
                fontweight='bold'
            )
