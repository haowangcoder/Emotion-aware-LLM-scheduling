"""
Publication-Quality Plotting Module for Emotion-aware LLM Scheduling

This package provides tools for generating top-conference quality figures.

Usage:
    from analysis.plotting import generate_all_publication_plots

    generate_all_publication_plots(
        aggregated_path='results/multi_seed_runs/aggregated_results.json',
        runs_dir='results/llm_runs_job54_load1.2',
        output_dir='results/publication_plots'
    )
"""

from .constants import COLORS, MARKERS, SCHEDULER_ORDER, AROUSAL_ORDER, VALENCE_ORDER
from .data_loader import load_aggregated_results, load_job_csvs, load_single_summaries
from .utils import setup_publication_style, save_figure
from .main import (
    generate_forest_plots,
    generate_distribution_plots,
    generate_pareto_plots,
    generate_group_analysis_plots,
    generate_calibration_plots,
    generate_all_publication_plots,
)

__all__ = [
    # Constants
    'COLORS',
    'MARKERS',
    'SCHEDULER_ORDER',
    'AROUSAL_ORDER',
    'VALENCE_ORDER',
    # Data loading
    'load_aggregated_results',
    'load_job_csvs',
    'load_single_summaries',
    # Utils
    'setup_publication_style',
    'save_figure',
    # Plot generators
    'generate_forest_plots',
    'generate_distribution_plots',
    'generate_pareto_plots',
    'generate_group_analysis_plots',
    'generate_calibration_plots',
    'generate_all_publication_plots',
]
