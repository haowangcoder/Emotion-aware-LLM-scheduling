"""
Service Time Distribution Analysis

Extract and analyze service time distributions from simulation data.
Supports per-class (arousal/valence) analysis.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .mg1_formulas import compute_service_time_moments


# Default arousal bins for 3-class system
DEFAULT_AROUSAL_BINS = {
    'low': (-1.0, -0.4),
    'medium': (-0.4, 0.4),
    'high': (0.4, 1.0)
}

# Default valence bins
DEFAULT_VALENCE_BINS = {
    'negative': (-1.0, -0.4),
    'neutral': (-0.4, 0.4),
    'positive': (0.4, 1.0)
}


def load_jobs_dataframe(jobs_csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load jobs CSV file into a DataFrame.

    Args:
        jobs_csv_path: Path to *_jobs.csv file

    Returns:
        DataFrame with job data
    """
    df = pd.read_csv(jobs_csv_path)

    # Ensure required columns exist
    required_cols = ['job_id', 'arrival_time']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    return df


def extract_service_time_distribution(
    jobs_csv_path: Union[str, Path],
    service_time_col: str = 'execution_duration',
    arousal_bins: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, dict]:
    """
    Extract per-arousal-class service time distribution from job CSV.

    Args:
        jobs_csv_path: Path to *_jobs.csv file
        service_time_col: Column name for service time
                         Options: 'execution_duration', 'predicted_service_time',
                                 'actual_execution_duration'
        arousal_bins: Custom bins for arousal classes
                     Default: {'low': (-1, -0.4), 'medium': (-0.4, 0.4), 'high': (0.4, 1)}

    Returns:
        Dict[class_name, distribution] where distribution = {
            'count': int,
            'proportion': float,
            'E_S': float,
            'E_S2': float,
            'Var_S': float,
            'C_s': float,
            'service_times': np.ndarray
        }
    """
    df = load_jobs_dataframe(jobs_csv_path)

    if arousal_bins is None:
        arousal_bins = DEFAULT_AROUSAL_BINS

    # Check for arousal column
    if 'arousal' not in df.columns:
        raise ValueError("Jobs CSV must have 'arousal' column for class analysis")

    # Check for service time column
    if service_time_col not in df.columns:
        # Try alternatives
        alternatives = ['execution_duration', 'predicted_service_time', 'actual_execution_duration']
        for alt in alternatives:
            if alt in df.columns:
                service_time_col = alt
                break
        else:
            raise ValueError(f"No service time column found. Tried: {alternatives}")

    results = {}
    total_jobs = len(df)

    for class_name, (lower, upper) in arousal_bins.items():
        # Filter jobs in this arousal range
        if class_name == 'high':
            # Include upper bound for 'high' class
            mask = (df['arousal'] >= lower) & (df['arousal'] <= upper)
        else:
            mask = (df['arousal'] >= lower) & (df['arousal'] < upper)

        class_df = df[mask]

        if len(class_df) == 0:
            results[class_name] = {
                'count': 0,
                'proportion': 0.0,
                'E_S': np.nan,
                'E_S2': np.nan,
                'Var_S': np.nan,
                'std_S': np.nan,
                'C_s': np.nan,
                'min': np.nan,
                'max': np.nan,
                'service_times': np.array([])
            }
            continue

        service_times = class_df[service_time_col].values
        moments = compute_service_time_moments(service_times)

        results[class_name] = {
            'count': len(class_df),
            'proportion': len(class_df) / total_jobs,
            **moments,
            'service_times': service_times
        }

    return results


def extract_valence_distribution(
    jobs_csv_path: Union[str, Path],
    service_time_col: str = 'execution_duration',
    valence_bins: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, dict]:
    """
    Extract per-valence-class distribution from job CSV.

    Similar to extract_service_time_distribution but groups by valence.

    Args:
        jobs_csv_path: Path to *_jobs.csv file
        service_time_col: Column name for service time
        valence_bins: Custom bins for valence classes

    Returns:
        Dict[class_name, distribution]
    """
    df = load_jobs_dataframe(jobs_csv_path)

    if valence_bins is None:
        valence_bins = DEFAULT_VALENCE_BINS

    if 'valence' not in df.columns:
        raise ValueError("Jobs CSV must have 'valence' column for valence analysis")

    if service_time_col not in df.columns:
        alternatives = ['execution_duration', 'predicted_service_time']
        for alt in alternatives:
            if alt in df.columns:
                service_time_col = alt
                break

    results = {}
    total_jobs = len(df)

    for class_name, (lower, upper) in valence_bins.items():
        if class_name == 'positive':
            mask = (df['valence'] >= lower) & (df['valence'] <= upper)
        else:
            mask = (df['valence'] >= lower) & (df['valence'] < upper)

        class_df = df[mask]

        if len(class_df) == 0:
            results[class_name] = {
                'count': 0,
                'proportion': 0.0,
                'E_S': np.nan,
                'service_times': np.array([])
            }
            continue

        service_times = class_df[service_time_col].values
        moments = compute_service_time_moments(service_times)

        results[class_name] = {
            'count': len(class_df),
            'proportion': len(class_df) / total_jobs,
            **moments,
            'service_times': service_times
        }

    return results


def extract_waiting_times_by_class(
    jobs_csv_path: Union[str, Path],
    waiting_time_col: str = 'waiting_time',
    arousal_bins: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, dict]:
    """
    Extract per-class waiting time statistics from job CSV.

    Args:
        jobs_csv_path: Path to *_jobs.csv file
        waiting_time_col: Column name for waiting time
        arousal_bins: Custom bins for arousal classes

    Returns:
        Dict[class_name, stats] where stats = {
            'count': int,
            'mean': float (W_q),
            'std': float,
            'p50': float,
            'p95': float,
            'p99': float,
            'waiting_times': np.ndarray
        }
    """
    df = load_jobs_dataframe(jobs_csv_path)

    if arousal_bins is None:
        arousal_bins = DEFAULT_AROUSAL_BINS

    # Try different column names for waiting time
    if waiting_time_col not in df.columns:
        alternatives = ['waiting_time', 'waiting_duration', 'wait_time']
        for alt in alternatives:
            if alt in df.columns:
                waiting_time_col = alt
                break
        else:
            # Compute from start_time - arrival_time if available
            if 'start_time' in df.columns and 'arrival_time' in df.columns:
                df['waiting_time'] = df['start_time'] - df['arrival_time']
                waiting_time_col = 'waiting_time'
            else:
                raise ValueError("No waiting time column found")

    results = {}

    for class_name, (lower, upper) in arousal_bins.items():
        if class_name == 'high':
            mask = (df['arousal'] >= lower) & (df['arousal'] <= upper)
        else:
            mask = (df['arousal'] >= lower) & (df['arousal'] < upper)

        class_df = df[mask]

        if len(class_df) == 0:
            results[class_name] = {
                'count': 0,
                'mean': np.nan,
                'std': np.nan,
                'p50': np.nan,
                'p95': np.nan,
                'p99': np.nan,
                'waiting_times': np.array([])
            }
            continue

        waiting_times = class_df[waiting_time_col].values

        results[class_name] = {
            'count': len(class_df),
            'mean': float(np.mean(waiting_times)),
            'std': float(np.std(waiting_times)),
            'p50': float(np.percentile(waiting_times, 50)),
            'p95': float(np.percentile(waiting_times, 95)),
            'p99': float(np.percentile(waiting_times, 99)),
            'waiting_times': waiting_times
        }

    return results


def extract_arrival_rate(
    jobs_csv_path: Optional[Union[str, Path]] = None,
    summary_json_path: Optional[Union[str, Path]] = None,
    jobs_df: Optional[pd.DataFrame] = None
) -> float:
    """
    Extract overall arrival rate from experiment data.

    Args:
        jobs_csv_path: Path to jobs CSV (compute from data)
        summary_json_path: Path to summary JSON (read stored value)
        jobs_df: Pre-loaded DataFrame

    Returns:
        Arrival rate λ (jobs/second)
    """
    # Try summary JSON first
    if summary_json_path is not None:
        with open(summary_json_path) as f:
            summary = json.load(f)

        # Try different keys for total time
        total_time = None
        total_jobs = None

        if 'run_metrics' in summary:
            total_time = summary['run_metrics'].get('total_time')
            total_jobs = summary['run_metrics'].get('total_jobs')
        elif 'total_time' in summary:
            total_time = summary['total_time']
            total_jobs = summary.get('total_jobs') or summary.get('num_jobs')

        if total_time and total_jobs:
            return total_jobs / total_time

    # Fall back to computing from job data
    if jobs_df is None:
        if jobs_csv_path is None:
            raise ValueError("Must provide either jobs_csv_path or jobs_df")
        jobs_df = load_jobs_dataframe(jobs_csv_path)

    # Compute from arrival times
    arrival_times = jobs_df['arrival_time'].values
    total_time = arrival_times[-1] - arrival_times[0]

    if total_time <= 0:
        return 0.0

    return len(jobs_df) / total_time


def extract_per_class_arrival_rates(
    jobs_csv_path: Union[str, Path],
    class_distributions: Optional[Dict[str, dict]] = None,
    arousal_bins: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, float]:
    """
    Extract per-class arrival rates.

    Per-class arrival rate λ_c = λ_total × proportion_c

    Args:
        jobs_csv_path: Path to jobs CSV
        class_distributions: Pre-computed distributions from extract_service_time_distribution()
        arousal_bins: Arousal class bins

    Returns:
        Dict[class_name, λ_c]
    """
    if class_distributions is None:
        class_distributions = extract_service_time_distribution(
            jobs_csv_path, arousal_bins=arousal_bins
        )

    lambda_total = extract_arrival_rate(jobs_csv_path=jobs_csv_path)

    arrival_rates = {
        class_name: lambda_total * dist['proportion']
        for class_name, dist in class_distributions.items()
    }

    return arrival_rates


def summarize_service_time_distribution(
    class_distributions: Dict[str, dict]
) -> pd.DataFrame:
    """
    Create a summary DataFrame of per-class service time distributions.

    Args:
        class_distributions: Output from extract_service_time_distribution()

    Returns:
        DataFrame with columns: [class, count, proportion, E_S, E_S2, Var_S, C_s]
    """
    rows = []
    for class_name, dist in class_distributions.items():
        rows.append({
            'class': class_name,
            'count': dist['count'],
            'proportion': dist['proportion'],
            'E_S': dist.get('E_S', np.nan),
            'E_S2': dist.get('E_S2', np.nan),
            'Var_S': dist.get('Var_S', np.nan),
            'std_S': dist.get('std_S', np.nan),
            'C_s': dist.get('C_s', np.nan),
            'min': dist.get('min', np.nan),
            'max': dist.get('max', np.nan)
        })

    return pd.DataFrame(rows)
