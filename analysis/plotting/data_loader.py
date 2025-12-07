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


def load_starvation_sweep_results(sweep_dir: str) -> dict:
    """
    Load starvation coefficient sweep results.

    Expected directory structure:
        sweep_dir/
        ├── baseline/           # FCFS baseline
        │   └── FCFS_*_summary.json
        ├── coeff_2/
        │   ├── SSJF-Emotion_*_summary.json
        │   └── SSJF-Combined_*_summary.json
        ├── coeff_5/
        └── ...

    Args:
        sweep_dir: Root directory of starvation sweep results

    Returns:
        Dict with structure:
        {
            'baseline': {'p99': float, 'jain': float, 'avg_wait': float},
            'schedulers': {
                'SSJF-Emotion': {
                    2: {'p99': float, 'jain': float, 'avg_wait': float},
                    5: {...},
                    ...
                },
                ...
            },
            'coefficients': [2, 5, 10, 20]
        }
    """
    sweep_dir = Path(sweep_dir)
    result = {
        'baseline': None,
        'schedulers': {},
        'coefficients': []
    }

    # Load baseline (FCFS)
    baseline_dir = sweep_dir / 'baseline'
    if baseline_dir.exists():
        for json_path in baseline_dir.glob('*_summary.json'):
            if json_path.stem.startswith('FCFS'):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                result['baseline'] = {
                    'p99': data['overall_metrics']['p99_waiting_time'],
                    'jain': data['fairness_analysis']['waiting_time_fairness']['jain_index'],
                    'avg_wait': data['overall_metrics']['avg_waiting_time']
                }
                break

    # Load coefficient sweep results
    coeff_dirs = sorted(sweep_dir.glob('coeff_*'))
    for coeff_dir in coeff_dirs:
        # Extract coefficient value from directory name
        coeff_str = coeff_dir.name.replace('coeff_', '')
        try:
            coeff = int(coeff_str)
        except ValueError:
            continue

        result['coefficients'].append(coeff)

        # Load each scheduler's results for this coefficient
        for json_path in coeff_dir.glob('*_summary.json'):
            name = json_path.stem
            for sched in SCHEDULER_ORDER:
                if name.startswith(sched) and sched != 'FCFS':
                    with open(json_path, 'r') as f:
                        data = json.load(f)

                    if sched not in result['schedulers']:
                        result['schedulers'][sched] = {}

                    result['schedulers'][sched][coeff] = {
                        'p99': data['overall_metrics']['p99_waiting_time'],
                        'jain': data['fairness_analysis']['waiting_time_fairness']['jain_index'],
                        'avg_wait': data['overall_metrics']['avg_waiting_time']
                    }
                    break

    return result


def load_param_sweep_results(sweep_dir: str) -> dict:
    """
    Load α × β parameter sweep results.

    Expected directory structure:
        sweep_dir/
        ├── baseline/                    # FCFS baseline
        │   └── FCFS_*_summary.json
        ├── alpha_0.0_beta_0.0/
        │   └── SSJF-Combined_*_summary.json
        ├── alpha_0.0_beta_0.25/
        └── ...

    Args:
        sweep_dir: Root directory of parameter sweep results

    Returns:
        Dict with structure:
        {
            'baseline': {'p99': float, 'jain': float, 'avg_wait': float},
            'grid': {
                (alpha, beta): {'p99': float, 'jain': float, 'avg_wait': float},
                ...
            },
            'alphas': [0.0, 0.25, 0.5, 0.75, 1.0],
            'betas': [0.0, 0.25, 0.5, 0.75, 1.0]
        }
    """
    import re

    sweep_dir = Path(sweep_dir)
    result = {
        'baseline': None,
        'grid': {},
        'alphas': set(),
        'betas': set()
    }

    # Load baseline (FCFS)
    baseline_dir = sweep_dir / 'baseline'
    if baseline_dir.exists():
        for json_path in baseline_dir.glob('*_summary.json'):
            if json_path.stem.startswith('FCFS'):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                result['baseline'] = {
                    'p99': data['overall_metrics']['p99_waiting_time'],
                    'jain': data['fairness_analysis']['waiting_time_fairness']['jain_index'],
                    'avg_wait': data['overall_metrics']['avg_waiting_time']
                }
                break

    # Load parameter sweep results
    pattern = re.compile(r'alpha_([\d.]+)_beta_([\d.]+)')
    param_dirs = sorted(sweep_dir.glob('alpha_*_beta_*'))

    for param_dir in param_dirs:
        match = pattern.match(param_dir.name)
        if not match:
            continue

        alpha = float(match.group(1))
        beta = float(match.group(2))

        result['alphas'].add(alpha)
        result['betas'].add(beta)

        # Load summary JSON
        for json_path in param_dir.glob('*_summary.json'):
            if json_path.stem.startswith('SSJF'):
                with open(json_path, 'r') as f:
                    data = json.load(f)

                result['grid'][(alpha, beta)] = {
                    'p99': data['overall_metrics']['p99_waiting_time'],
                    'jain': data['fairness_analysis']['waiting_time_fairness']['jain_index'],
                    'avg_wait': data['overall_metrics']['avg_waiting_time']
                }
                break

    # Convert sets to sorted lists
    result['alphas'] = sorted(result['alphas'])
    result['betas'] = sorted(result['betas'])

    return result
