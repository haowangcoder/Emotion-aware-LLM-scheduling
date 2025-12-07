"""
Theory vs Simulation Validation Workflow

Main validation module that compares M/G/1 theoretical predictions
with simulation results.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .mg1_formulas import pollaczek_khinchin, compute_service_time_moments
from .multiclass_priority import (
    kleinrock_multiclass_priority,
    weighted_average_response_time,
    ssjf_emotion_priority_order
)
from .service_time_analysis import (
    load_jobs_dataframe,
    extract_service_time_distribution,
    extract_waiting_times_by_class,
    extract_arrival_rate,
    extract_per_class_arrival_rates
)
from .error_analysis import (
    remove_transient_period,
    finite_sample_correction,
    compare_theory_vs_simulation,
    bootstrap_confidence_interval
)


def validate_fcfs_baseline(
    summary_json_path: Union[str, Path],
    jobs_csv_path: Union[str, Path],
    apply_transient_removal: bool = True,
    transient_warmup: float = 0.1,
    transient_cooldown: float = 0.1,
    apply_finite_correction: bool = False,
    service_time_col: str = 'execution_duration'
) -> dict:
    """
    Validate FCFS scheduler using Pollaczek-Khinchin formula.

    Computes theoretical waiting time and compares with simulation.

    Args:
        summary_json_path: Path to *_summary.json file
        jobs_csv_path: Path to *_jobs.csv file
        apply_transient_removal: Remove warm-up/cool-down periods
        transient_warmup: Fraction to remove from start
        transient_cooldown: Fraction to remove from end
        apply_finite_correction: Apply finite sample correction
        service_time_col: Column name for service time

    Returns:
        dict with:
            - theory: {rho, W_q, W, ...}
            - simulation: {W_q, W, n_jobs, ...}
            - errors: {abs_error_W_q, rel_error_W_q, ...}
            - parameters: {lambda, E_S, E_S2, ...}
    """
    # Load data
    jobs_df = load_jobs_dataframe(jobs_csv_path)

    # Remove transient period if requested
    if apply_transient_removal:
        jobs_df = remove_transient_period(
            jobs_df,
            warmup_fraction=transient_warmup,
            cooldown_fraction=transient_cooldown
        )

    # Extract service time distribution
    if service_time_col not in jobs_df.columns:
        # Try alternatives
        for alt in ['actual_serving_time', 'execution_duration', 'predicted_service_time', 'actual_execution_duration']:
            if alt in jobs_df.columns:
                service_time_col = alt
                break

    service_times = jobs_df[service_time_col].values
    moments = compute_service_time_moments(service_times)

    # Extract arrival rate
    # Compute from filtered data
    if 'arrival_time' in jobs_df.columns:
        arrival_times = jobs_df['arrival_time'].values
        total_time = arrival_times[-1] - arrival_times[0]
        lambda_arrival = len(jobs_df) / total_time if total_time > 0 else 0
    else:
        lambda_arrival = extract_arrival_rate(summary_json_path=summary_json_path)

    # Theory prediction
    theory = pollaczek_khinchin(lambda_arrival, moments['E_S'], moments['E_S2'])

    # Apply finite sample correction if requested
    if apply_finite_correction:
        theory['W_q_corrected'] = finite_sample_correction(
            theory['W_q'], len(jobs_df), theory['rho']
        )
        theory['W_corrected'] = theory['W_q_corrected'] + moments['E_S']
        theory_W_q = theory['W_q_corrected']
        theory_W = theory['W_corrected']
    else:
        theory_W_q = theory['W_q']
        theory_W = theory['W']

    # Simulation results
    # Try different column names for waiting time
    waiting_col = None
    for col in ['waiting_time', 'waiting_duration', 'wait_time']:
        if col in jobs_df.columns:
            waiting_col = col
            break

    if waiting_col is None and 'start_time' in jobs_df.columns:
        jobs_df['waiting_time'] = jobs_df['start_time'] - jobs_df['arrival_time']
        waiting_col = 'waiting_time'

    if waiting_col:
        sim_W_q = jobs_df[waiting_col].mean()
        sim_W_q_std = jobs_df[waiting_col].std()
    else:
        sim_W_q = np.nan
        sim_W_q_std = np.nan

    # Turnaround time
    if 'turnaround_time' in jobs_df.columns:
        sim_W = jobs_df['turnaround_time'].mean()
    elif 'completion_time' in jobs_df.columns and 'arrival_time' in jobs_df.columns:
        sim_W = (jobs_df['completion_time'] - jobs_df['arrival_time']).mean()
    else:
        sim_W = sim_W_q + moments['E_S'] if not np.isnan(sim_W_q) else np.nan

    # Compare
    errors = compare_theory_vs_simulation(theory_W_q, sim_W_q, theory_W, sim_W)

    return {
        'theory': theory,
        'simulation': {
            'W_q': sim_W_q,
            'W_q_std': sim_W_q_std,
            'W': sim_W,
            'n_jobs': len(jobs_df),
        },
        'errors': errors,
        'parameters': {
            'lambda': lambda_arrival,
            **moments
        },
        'config': {
            'apply_transient_removal': apply_transient_removal,
            'transient_warmup': transient_warmup,
            'transient_cooldown': transient_cooldown,
            'apply_finite_correction': apply_finite_correction
        }
    }


def validate_ssjf_emotion(
    summary_json_path: Union[str, Path],
    jobs_csv_path: Union[str, Path],
    apply_transient_removal: bool = True,
    transient_warmup: float = 0.1,
    transient_cooldown: float = 0.1,
    service_time_col: str = 'execution_duration'
) -> dict:
    """
    Validate SSJF-Emotion scheduler using Kleinrock multi-class priority formula.

    Computes per-class theoretical waiting times and compares with simulation.

    Args:
        summary_json_path: Path to *_summary.json file
        jobs_csv_path: Path to *_jobs.csv file
        apply_transient_removal: Remove warm-up/cool-down periods
        transient_warmup: Fraction to remove from start
        transient_cooldown: Fraction to remove from end
        service_time_col: Column name for service time

    Returns:
        dict with:
            - theory: {class_name: {W_q, W, priority, ...}}
            - simulation: {class_name: {W_q, W, n_jobs, ...}}
            - comparison: {class_name: {abs_error, rel_error, ...}}
            - overall: {theory_W_q_avg, sim_W_q_avg, ...}
    """
    # Load data
    jobs_df = load_jobs_dataframe(jobs_csv_path)

    # Remove transient period if requested
    if apply_transient_removal:
        jobs_df = remove_transient_period(
            jobs_df,
            warmup_fraction=transient_warmup,
            cooldown_fraction=transient_cooldown
        )

    # Check for arousal column
    if 'arousal' not in jobs_df.columns:
        raise ValueError("Jobs CSV must have 'arousal' column for SSJF-Emotion validation")

    # Resolve service time column against available alternatives
    if service_time_col not in jobs_df.columns:
        for alt in ['actual_serving_time', 'execution_duration', 'predicted_service_time', 'actual_execution_duration']:
            if alt in jobs_df.columns:
                service_time_col = alt
                break

    # Extract per-class service time distributions
    class_dists = extract_service_time_distribution(
        jobs_csv_path, service_time_col=service_time_col
    )

    # For filtered data, recompute distributions
    # (extract_service_time_distribution reads from file, not filtered df)
    # So we manually compute here
    class_dists_filtered = _compute_class_distributions(
        jobs_df, service_time_col
    )

    # Extract arrival rate and per-class rates
    if 'arrival_time' in jobs_df.columns:
        arrival_times = jobs_df['arrival_time'].values
        total_time = arrival_times[-1] - arrival_times[0]
        lambda_total = len(jobs_df) / total_time if total_time > 0 else 0
    else:
        lambda_total = extract_arrival_rate(summary_json_path=summary_json_path)

    # Per-class arrival rates
    arrival_rates = {
        c: lambda_total * d['proportion']
        for c, d in class_dists_filtered.items()
    }

    # Per-class service times
    mean_service_times = {c: d['E_S'] for c, d in class_dists_filtered.items()}
    second_moments = {c: d['E_S2'] for c, d in class_dists_filtered.items()}

    # Theory: Kleinrock formula
    priority_order = ssjf_emotion_priority_order()

    # Filter to only classes that exist in data
    valid_classes = [c for c in priority_order if c in arrival_rates and arrival_rates[c] > 0]

    if len(valid_classes) < len(priority_order):
        # Some classes missing, use available ones
        priority_order = valid_classes

    theory_metrics = kleinrock_multiclass_priority(
        {c: arrival_rates[c] for c in priority_order},
        {c: mean_service_times[c] for c in priority_order},
        {c: second_moments[c] for c in priority_order},
        priority_order
    )

    # Simulation: per-class metrics
    simulation_metrics = _compute_simulation_metrics(jobs_df, priority_order)

    # Compare per-class
    comparison = {}
    for class_name in priority_order:
        if class_name in theory_metrics and class_name in simulation_metrics:
            comparison[class_name] = compare_theory_vs_simulation(
                theory_metrics[class_name]['W_q'],
                simulation_metrics[class_name]['W_q'],
                theory_metrics[class_name]['W'],
                simulation_metrics[class_name]['W']
            )

    # Overall weighted average
    theory_overall = weighted_average_response_time(theory_metrics)

    sim_W_q_avg = np.average(
        [simulation_metrics[c]['W_q'] for c in priority_order],
        weights=[class_dists_filtered[c]['proportion'] for c in priority_order]
    )

    return {
        'theory': theory_metrics,
        'simulation': simulation_metrics,
        'comparison': comparison,
        'overall': {
            'theory_W_q_avg': theory_overall['W_q_avg'],
            'theory_W_avg': theory_overall['W_avg'],
            'sim_W_q_avg': sim_W_q_avg,
            'lambda_total': lambda_total,
            'n_jobs': len(jobs_df)
        },
        'class_distributions': class_dists_filtered,
        'config': {
            'apply_transient_removal': apply_transient_removal,
            'transient_warmup': transient_warmup,
            'transient_cooldown': transient_cooldown
        }
    }


def _compute_class_distributions(
    jobs_df: pd.DataFrame,
    service_time_col: str
) -> Dict[str, dict]:
    """Compute per-class service time distributions from DataFrame."""
    arousal_bins = {
        'low': (-1.0, -0.4),
        'medium': (-0.4, 0.4),
        'high': (0.4, 1.0)
    }

    results = {}
    total_jobs = len(jobs_df)

    for class_name, (lower, upper) in arousal_bins.items():
        if class_name == 'high':
            mask = (jobs_df['arousal'] >= lower) & (jobs_df['arousal'] <= upper)
        else:
            mask = (jobs_df['arousal'] >= lower) & (jobs_df['arousal'] < upper)

        class_df = jobs_df[mask]

        if len(class_df) == 0:
            results[class_name] = {
                'count': 0,
                'proportion': 0.0,
                'E_S': np.nan,
                'E_S2': np.nan
            }
            continue

        service_times = class_df[service_time_col].values
        moments = compute_service_time_moments(service_times)

        results[class_name] = {
            'count': len(class_df),
            'proportion': len(class_df) / total_jobs,
            **moments
        }

    return results


def _compute_simulation_metrics(
    jobs_df: pd.DataFrame,
    class_names: List[str]
) -> Dict[str, dict]:
    """Compute per-class simulation metrics from DataFrame."""
    arousal_bins = {
        'low': (-1.0, -0.4),
        'medium': (-0.4, 0.4),
        'high': (0.4, 1.0)
    }

    # Determine waiting time column
    waiting_col = None
    for col in ['waiting_time', 'waiting_duration', 'wait_time']:
        if col in jobs_df.columns:
            waiting_col = col
            break

    if waiting_col is None and 'start_time' in jobs_df.columns:
        jobs_df = jobs_df.copy()
        jobs_df['waiting_time'] = jobs_df['start_time'] - jobs_df['arrival_time']
        waiting_col = 'waiting_time'

    results = {}

    for class_name in class_names:
        if class_name not in arousal_bins:
            continue

        lower, upper = arousal_bins[class_name]
        if class_name == 'high':
            mask = (jobs_df['arousal'] >= lower) & (jobs_df['arousal'] <= upper)
        else:
            mask = (jobs_df['arousal'] >= lower) & (jobs_df['arousal'] < upper)

        class_df = jobs_df[mask]

        if len(class_df) == 0:
            results[class_name] = {
                'W_q': np.nan,
                'W': np.nan,
                'n_jobs': 0
            }
            continue

        W_q = class_df[waiting_col].mean() if waiting_col else np.nan

        if 'turnaround_time' in class_df.columns:
            W = class_df['turnaround_time'].mean()
        elif 'completion_time' in class_df.columns:
            W = (class_df['completion_time'] - class_df['arrival_time']).mean()
        else:
            W = np.nan

        results[class_name] = {
            'W_q': W_q,
            'W_q_std': class_df[waiting_col].std() if waiting_col else np.nan,
            'W': W,
            'n_jobs': len(class_df)
        }

    return results


def validate_load_sweep(
    results_dir: Union[str, Path],
    load_levels: List[float],
    schedulers: List[str] = ['FCFS', 'SSJF-Emotion'],
    apply_transient_removal: bool = True
) -> Dict[str, Dict[float, dict]]:
    """
    Validate theory vs simulation across multiple load levels.

    Expects directory structure:
        results_dir/
        ├── load_0.5/
        │   ├── FCFS_*_summary.json
        │   ├── FCFS_*_jobs.csv
        │   ├── SSJF-Emotion_*_summary.json
        │   └── SSJF-Emotion_*_jobs.csv
        ├── load_0.6/
        │   └── ...

    Args:
        results_dir: Base directory containing load subdirectories
        load_levels: List of load levels to validate
        schedulers: List of schedulers to validate
        apply_transient_removal: Remove transient periods

    Returns:
        {scheduler: {load: validation_result}}
    """
    results_dir = Path(results_dir)
    results = {s: {} for s in schedulers}

    for load in load_levels:
        load_dir = results_dir / f"load_{load}"

        if not load_dir.exists():
            print(f"Warning: {load_dir} not found, skipping")
            continue

        for scheduler in schedulers:
            # Find summary and jobs files
            summary_files = list(load_dir.glob(f"{scheduler}_*_summary.json"))
            jobs_files = list(load_dir.glob(f"{scheduler}_*_jobs.csv"))

            if not summary_files or not jobs_files:
                print(f"Warning: {scheduler} files not found in {load_dir}")
                continue

            summary_path = summary_files[0]
            jobs_path = jobs_files[0]

            if scheduler == 'FCFS':
                result = validate_fcfs_baseline(
                    summary_path, jobs_path,
                    apply_transient_removal=apply_transient_removal
                )
            elif scheduler == 'SSJF-Emotion':
                result = validate_ssjf_emotion(
                    summary_path, jobs_path,
                    apply_transient_removal=apply_transient_removal
                )
            else:
                print(f"Warning: Unknown scheduler {scheduler}")
                continue

            results[scheduler][load] = result

    return results


def generate_validation_report(
    validation_results: Dict[str, Dict[float, dict]],
    output_path: Optional[Union[str, Path]] = None
) -> str:
    """
    Generate a text report of validation results.

    Args:
        validation_results: Output from validate_load_sweep()
        output_path: Optional path to save report

    Returns:
        Report as string
    """
    lines = [
        "# M/G/1 Queueing Theory Validation Report",
        "",
        "## Summary",
        ""
    ]

    for scheduler, load_results in validation_results.items():
        lines.append(f"### {scheduler}")
        lines.append("")
        lines.append("| Load | Theory W_q | Sim W_q | Error (%) |")
        lines.append("|------|-----------|---------|-----------|")

        for load in sorted(load_results.keys()):
            result = load_results[load]

            if scheduler == 'FCFS':
                theory_wq = result['theory']['W_q']
                sim_wq = result['simulation']['W_q']
                error = result['errors']['rel_error_W_q']
            else:  # SSJF-Emotion
                theory_wq = result['overall']['theory_W_q_avg']
                sim_wq = result['overall']['sim_W_q_avg']
                error = abs(theory_wq - sim_wq) / theory_wq * 100 if theory_wq > 0 else np.nan

            lines.append(f"| {load:.2f} | {theory_wq:.3f}s | {sim_wq:.3f}s | {error:.1f}% |")

        lines.append("")

    report = "\n".join(lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)

    return report
