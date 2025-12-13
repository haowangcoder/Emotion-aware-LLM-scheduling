"""
Result aggregation utilities for experiment analysis.

This module provides functions to:
- Load experiment results from CSV/JSON files
- Aggregate metrics across multiple runs
- Extract per-quadrant and per-scheduler statistics
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union


def load_csv_results(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load job-level results from CSV file.

    Args:
        csv_path: Path to the CSV file

    Returns:
        DataFrame with job-level data
    """
    return pd.read_csv(csv_path)


def load_json_summary(json_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load experiment summary from JSON file.

    Args:
        json_path: Path to the JSON file

    Returns:
        Dictionary with experiment summary
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def find_result_files(result_dir: Union[str, Path], pattern: str = "*_summary.json") -> List[Path]:
    """
    Find all result files matching a pattern in a directory.

    Args:
        result_dir: Directory to search
        pattern: Glob pattern to match

    Returns:
        List of matching file paths
    """
    result_dir = Path(result_dir)
    return sorted(result_dir.glob(pattern))


def aggregate_scheduler_results(
    result_dirs: List[Union[str, Path]],
    schedulers: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate results across multiple schedulers.

    Args:
        result_dirs: List of result directories
        schedulers: Optional list of scheduler names to filter

    Returns:
        Dictionary mapping scheduler name to aggregated metrics
    """
    results = {}

    for result_dir in result_dirs:
        result_dir = Path(result_dir)
        json_files = find_result_files(result_dir, "*_summary.json")

        for json_file in json_files:
            summary = load_json_summary(json_file)
            scheduler = summary.get('metadata', {}).get('scheduler', 'unknown')

            if schedulers is None or scheduler in schedulers:
                if scheduler not in results:
                    results[scheduler] = []
                results[scheduler].append(summary)

    return results


def extract_metrics(
    summary: Dict[str, Any],
    metrics: List[str]
) -> Dict[str, float]:
    """
    Extract specific metrics from a summary dictionary.

    Args:
        summary: Experiment summary dictionary
        metrics: List of metric names to extract

    Returns:
        Dictionary mapping metric names to values
    """
    extracted = {}

    # Check different locations for metrics
    run_metrics = summary.get('run_metrics', {})
    overall_metrics = summary.get('overall_metrics', {})

    for metric in metrics:
        if metric in run_metrics:
            extracted[metric] = run_metrics[metric]
        elif metric in overall_metrics:
            extracted[metric] = overall_metrics[metric]
        elif metric in summary:
            extracted[metric] = summary[metric]

    return extracted


def extract_quadrant_metrics(
    summary: Dict[str, Any],
    quadrant: str,
    metric: str = 'avg_waiting_time'
) -> Optional[float]:
    """
    Extract a metric for a specific Russell quadrant.

    Args:
        summary: Experiment summary dictionary
        quadrant: Russell quadrant name ('depression', 'panic', 'excited', 'calm')
        metric: Metric name to extract

    Returns:
        Metric value or None if not found
    """
    per_quadrant = summary.get('per_quadrant_metrics', {})
    quadrant_data = per_quadrant.get(quadrant, {})
    return quadrant_data.get(metric)


def compute_depression_speedup(
    summary: Dict[str, Any],
    baseline_depression_wait: float
) -> float:
    """
    Compute depression speedup relative to a baseline.

    Args:
        summary: Experiment summary dictionary
        baseline_depression_wait: Baseline depression waiting time (e.g., from FCFS)

    Returns:
        Speedup factor (baseline / current)
    """
    depression_wait = extract_quadrant_metrics(summary, 'depression', 'avg_waiting_time')
    if depression_wait is None or depression_wait == 0:
        return float('inf')
    return baseline_depression_wait / depression_wait


def aggregate_sweep_results(
    sweep_dir: Union[str, Path],
    param_name: str,
    param_values: List[Any]
) -> Dict[Any, Dict[str, Any]]:
    """
    Aggregate results from a parameter sweep experiment.

    Args:
        sweep_dir: Base directory containing sweep results
        param_name: Name of the swept parameter (e.g., 'k', 'load', 'gamma')
        param_values: List of parameter values

    Returns:
        Dictionary mapping parameter values to result summaries
    """
    sweep_dir = Path(sweep_dir)
    results = {}

    for value in param_values:
        # Try common directory naming patterns
        patterns = [
            f"{param_name}{value}",
            f"{param_name}_{value}",
            f"{param_name}={value}",
            f"load{value}",
            f"k{value}",
            f"gamma{value}",
        ]

        for pattern in patterns:
            subdir = sweep_dir / pattern
            if subdir.exists():
                json_files = find_result_files(subdir)
                if json_files:
                    results[value] = load_json_summary(json_files[0])
                    break

    return results


def compute_summary_statistics(values: List[float]) -> Dict[str, float]:
    """
    Compute summary statistics for a list of values.

    Args:
        values: List of numeric values

    Returns:
        Dictionary with mean, std, min, max, median, p95, p99
    """
    arr = np.array(values)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'median': float(np.median(arr)),
        'p95': float(np.percentile(arr, 95)),
        'p99': float(np.percentile(arr, 99)),
    }


def create_comparison_table(
    results: Dict[str, Dict[str, Any]],
    metrics: List[str]
) -> pd.DataFrame:
    """
    Create a comparison table of metrics across schedulers/configurations.

    Args:
        results: Dictionary mapping names to result summaries
        metrics: List of metric names to include

    Returns:
        DataFrame with comparison table
    """
    rows = []
    for name, summary in results.items():
        row = {'name': name}
        row.update(extract_metrics(summary, metrics))
        rows.append(row)

    return pd.DataFrame(rows)
