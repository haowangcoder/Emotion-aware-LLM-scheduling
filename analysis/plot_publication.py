#!/usr/bin/env python3
"""
Publication-Quality Visualization Script for Emotion-aware LLM Scheduling

This is the CLI entry point. The actual plotting logic is in the
analysis.plotting package.

Usage:
    python analysis/plot_publication.py \
        --aggregated results/multi_seed_runs/aggregated_results.json \
        --runs-dir results/llm_runs_job54_load1.2 \
        --output-dir results/publication_plots
"""

import argparse
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plotting import generate_all_publication_plots


def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-quality plots for emotion-aware scheduling results.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analysis/plot_publication.py \\
      --aggregated results/multi_seed_runs/aggregated_results.json \\
      --runs-dir results/llm_runs_job54_load1.2 \\
      --output-dir results/publication_plots

  python analysis/plot_publication.py \\
      --runs-dir results/llm_runs_job54_load1.2 \\
      --output-dir results/publication_plots \\
      --format png
        """
    )

    parser.add_argument(
        '--aggregated',
        type=str,
        default='results/multi_seed_runs/aggregated_results.json',
        help='Path to aggregated_results.json (default: %(default)s)'
    )

    parser.add_argument(
        '--runs-dir',
        type=str,
        default='results/llm_runs_job54_load1.2',
        help='Directory with *_jobs.csv and *_summary.json files (default: %(default)s)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/publication_plots',
        help='Output directory for plots (default: %(default)s)'
    )

    parser.add_argument(
        '--format',
        type=str,
        default='pdf,png',
        help='Output format(s), comma-separated (default: %(default)s)'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='DPI for PNG output (default: %(default)s)'
    )

    args = parser.parse_args()

    formats = [f.strip() for f in args.format.split(',')]

    generate_all_publication_plots(
        aggregated_path=args.aggregated,
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
        formats=formats,
        dpi=args.dpi,
    )


if __name__ == '__main__':
    main()
