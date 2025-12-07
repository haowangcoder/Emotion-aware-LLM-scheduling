"""
Plot generation module (compatibility layer).

This module re-exports all plot functions from the plots/ subpackage
for backward compatibility.

For new code, prefer importing directly from the subpackage:
    from plotting.plots import forest, distribution, pareto, slopegraph, calibration
"""

import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .plots import (  # noqa: F401 - subpackage exports
    # Forest
    generate_forest_plots,
    plot_forest_metric,
    plot_forest_delta_vs_baseline,
    # Distribution
    generate_distribution_plots,
    plot_ecdf,
    plot_ccdf_log,
    # Pareto
    generate_pareto_plots,
    plot_pareto_tradeoff,
    plot_starvation_pareto,
    generate_starvation_sweep_plots,
    # Slopegraph
    generate_group_analysis_plots,
    plot_slopegraph_arousal,
    plot_slopegraph_valence,
    plot_dumbbell_arousal,
    # Calibration
    generate_calibration_plots,
    plot_calibration_scatter,
    plot_calibration_binned,
    plot_error_distribution,
    # Heatmap
    plot_param_heatmap,
    plot_param_dual_line,
    generate_param_sweep_plots,
    # Ablation
    plot_shuffle_comparison,
    plot_shuffle_delta,
    generate_shuffle_experiment_plots,
)
from .utils import setup_publication_style
from .data_loader import (
    load_aggregated_results,
    load_job_csvs,
    load_single_summaries,
    load_starvation_sweep_results
)


def generate_all_publication_plots(aggregated_path: str, runs_dir: str,
                                    output_dir: str, formats: List[str] = None,
                                    dpi: int = 300):
    """
    Generate all publication-quality plots.

    Args:
        aggregated_path: Path to aggregated_results.json
        runs_dir: Path to directory with *_jobs.csv and *_summary.json files
        output_dir: Output directory for plots
        formats: Output formats (default: ['pdf', 'png'])
        dpi: DPI for saved figures (default: 300)
    """
    setup_publication_style(dpi=dpi)

    if formats is None:
        formats = ['pdf', 'png']

    print(f"Publication Plot Generator")
    print(f"=" * 50)
    print(f"Aggregated data: {aggregated_path}")
    print(f"Runs directory:  {runs_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Output formats:  {formats}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\nLoading data...")

    agg_data = None
    if aggregated_path and os.path.exists(aggregated_path):
        agg_data = load_aggregated_results(aggregated_path)
        print(f"  Loaded aggregated results with {len(agg_data.get('schedulers', {}))} schedulers")
    else:
        print(f"  Warning: Aggregated results not found at {aggregated_path}")

    dfs: Dict[str, pd.DataFrame] = {}
    summaries: Dict[str, dict] = {}
    if runs_dir and os.path.exists(runs_dir):
        dfs = load_job_csvs(runs_dir)
        print(f"  Loaded {len(dfs)} job CSV files")
        summaries = load_single_summaries(runs_dir)
        print(f"  Loaded {len(summaries)} summary JSON files")
    else:
        print(f"  Warning: Runs directory not found at {runs_dir}")

    if agg_data:
        generate_forest_plots(agg_data, output_dir, formats)
        generate_pareto_plots(agg_data, output_dir, formats)
        generate_group_analysis_plots(agg_data, summaries, output_dir, formats)

    if dfs:
        generate_distribution_plots(dfs, output_dir, formats)
        generate_calibration_plots(dfs, output_dir, formats)

    print(f"\n{'=' * 50}")
    print(f"All plots generated successfully!")
    print(f"Output directory: {output_dir}")


# Re-export everything for backward compatibility
__all__ = [
    'generate_all_publication_plots',
    # Forest
    'generate_forest_plots',
    'plot_forest_metric',
    'plot_forest_delta_vs_baseline',
    # Distribution
    'generate_distribution_plots',
    'plot_ecdf',
    'plot_ccdf_log',
    # Pareto
    'generate_pareto_plots',
    'plot_pareto_tradeoff',
    'plot_starvation_pareto',
    'generate_starvation_sweep_plots',
    # Slopegraph
    'generate_group_analysis_plots',
    'plot_slopegraph_arousal',
    'plot_slopegraph_valence',
    'plot_dumbbell_arousal',
    # Calibration
    'generate_calibration_plots',
    'plot_calibration_scatter',
    'plot_calibration_binned',
    'plot_error_distribution',
    # Heatmap
    'plot_param_heatmap',
    'plot_param_dual_line',
    'generate_param_sweep_plots',
    # Ablation
    'plot_shuffle_comparison',
    'plot_shuffle_delta',
    'generate_shuffle_experiment_plots',
]
