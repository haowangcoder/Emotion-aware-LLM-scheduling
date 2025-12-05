"""
Data loading functions for publication plotting.

Handles loading of:
- Aggregated multi-seed results (JSON)
- Job-level data (CSV)
- Single-seed summary files (JSON)
"""

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from .constants import SCHEDULER_ORDER


def load_aggregated_results(path: str) -> dict:
    """
    Load multi-seed aggregated results JSON.

    Args:
        path: Path to aggregated_results.json

    Returns:
        Dict containing aggregated statistics with CI
    """
    with open(path, 'r') as f:
        return json.load(f)


def load_job_csvs(runs_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load job-level CSV files from a runs directory.

    Args:
        runs_dir: Directory containing *_jobs.csv files

    Returns:
        Dict mapping scheduler name -> DataFrame
    """
    runs_dir = Path(runs_dir)
    dfs = {}

    for csv_path in sorted(runs_dir.glob('*_jobs.csv')):
        # Extract scheduler name from filename
        # e.g., "FCFS_54jobs_load1.20_fixed_jobs.csv" -> "FCFS"
        name = csv_path.stem
        for sched in SCHEDULER_ORDER:
            if name.startswith(sched):
                dfs[sched] = pd.read_csv(csv_path)
                break

    return dfs


def load_single_summaries(runs_dir: str) -> Dict[str, dict]:
    """
    Load single-seed summary JSON files from a runs directory.

    Args:
        runs_dir: Directory containing *_summary.json files

    Returns:
        Dict mapping scheduler name -> summary dict
    """
    runs_dir = Path(runs_dir)
    summaries = {}

    for json_path in sorted(runs_dir.glob('*_summary.json')):
        name = json_path.stem
        for sched in SCHEDULER_ORDER:
            if name.startswith(sched):
                with open(json_path, 'r') as f:
                    summaries[sched] = json.load(f)
                break

    return summaries
