"""
Load Sweep Theory Predictions

Generate theoretical predictions across multiple system load levels
for comparison with simulation results.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from .mg1_formulas import pollaczek_khinchin, theoretical_service_time_moments
from .multiclass_priority import (
    kleinrock_multiclass_priority,
    weighted_average_response_time,
    ssjf_emotion_priority_order
)


# Default parameters matching the emotion-aware scheduling system
DEFAULT_AROUSAL_VALUES = {
    'low': -0.8,
    'medium': 0.0,
    'high': 0.8
}


def predict_theory_load_sweep(
    base_service_time: float,
    alpha: float,
    load_levels: List[float],
    arousal_values: Optional[Dict[str, float]] = None,
    class_proportions: Optional[Dict[str, float]] = None,
    scheduler: str = 'FCFS'
) -> pd.DataFrame:
    """
    Predict waiting times across load levels using M/G/1 theory.

    For FCFS: Uses standard P-K formula (same waiting time for all classes)
    For SSJF-Emotion: Uses Kleinrock multi-class priority formula

    Args:
        base_service_time: L_0 (base service time in seconds)
        alpha: α coefficient for arousal-service time mapping
        load_levels: List of target utilization levels ρ (e.g., [0.5, 0.6, 0.7, 0.8, 0.9])
        arousal_values: {class_name: arousal_value}
                       Default: {'low': -0.8, 'medium': 0.0, 'high': 0.8}
        class_proportions: {class_name: proportion}
                          Defaults to uniform (1/3 each)
        scheduler: 'FCFS' or 'SSJF-Emotion'

    Returns:
        DataFrame with columns:
            - load: System utilization ρ
            - scheduler: Scheduler type
            - class: Arousal class (for per-class analysis)
            - W_q: Theoretical waiting time
            - W: Theoretical response time
            - lambda_total: Total arrival rate
            - E_S: Mean service time
            - stable: Whether system is stable at this load

    Example:
        >>> loads = [0.5, 0.6, 0.7, 0.8, 0.9]
        >>> df = predict_theory_load_sweep(
        ...     base_service_time=2.0, alpha=0.8, load_levels=loads
        ... )
        >>> print(df[df['class'] == 'overall'][['load', 'W_q']])
    """
    if arousal_values is None:
        arousal_values = DEFAULT_AROUSAL_VALUES

    if class_proportions is None:
        n_classes = len(arousal_values)
        class_proportions = {c: 1.0 / n_classes for c in arousal_values}

    # Compute service time moments from model parameters
    moments = theoretical_service_time_moments(
        base_service_time, alpha, arousal_values, class_proportions
    )

    E_S_overall = moments['overall']['E_S']
    E_S2_overall = moments['overall']['E_S2']

    results = []

    for rho_target in load_levels:
        # Check stability
        if rho_target >= 1.0:
            # Unstable system
            _add_unstable_rows(results, rho_target, scheduler, arousal_values)
            continue

        # Compute arrival rate to achieve target load
        # ρ = λ × E[S] => λ = ρ / E[S]
        lambda_total = rho_target / E_S_overall

        if scheduler == 'FCFS':
            # Standard P-K formula (same W_q for all classes)
            pk_result = pollaczek_khinchin(lambda_total, E_S_overall, E_S2_overall)

            # Add overall result
            results.append({
                'load': rho_target,
                'scheduler': 'FCFS',
                'class': 'overall',
                'W_q': pk_result['W_q'],
                'W': pk_result['W'],
                'lambda_total': lambda_total,
                'E_S': E_S_overall,
                'stable': pk_result['stable']
            })

            # Add per-class results (same W_q for FCFS)
            for class_name in arousal_values:
                E_Sc = moments['per_class'][class_name]['E_S']
                results.append({
                    'load': rho_target,
                    'scheduler': 'FCFS',
                    'class': class_name,
                    'W_q': pk_result['W_q'],
                    'W': pk_result['W_q'] + E_Sc,  # Response time = W_q + E[S_c]
                    'lambda_total': lambda_total,
                    'E_S': E_Sc,
                    'stable': pk_result['stable']
                })

        elif scheduler == 'SSJF-Emotion':
            # Kleinrock multi-class priority formula
            # Per-class arrival rates
            arrival_rates = {
                c: lambda_total * class_proportions[c]
                for c in arousal_values
            }

            # Per-class service times and moments
            mean_service_times = {
                c: moments['per_class'][c]['E_S']
                for c in arousal_values
            }
            second_moments = {
                c: moments['per_class'][c]['E_S2']
                for c in arousal_values
            }

            # Priority order: low < medium < high (shortest first)
            priority_order = ssjf_emotion_priority_order()

            class_metrics = kleinrock_multiclass_priority(
                arrival_rates, mean_service_times, second_moments, priority_order
            )

            # Add per-class results
            for class_name, metrics in class_metrics.items():
                results.append({
                    'load': rho_target,
                    'scheduler': 'SSJF-Emotion',
                    'class': class_name,
                    'W_q': metrics['W_q'],
                    'W': metrics['W'],
                    'lambda_total': lambda_total,
                    'E_S': metrics['E_S'],
                    'stable': metrics['stable'],
                    'priority': metrics['priority']
                })

            # Add overall weighted average
            overall = weighted_average_response_time(class_metrics)
            results.append({
                'load': rho_target,
                'scheduler': 'SSJF-Emotion',
                'class': 'overall',
                'W_q': overall['W_q_avg'],
                'W': overall['W_avg'],
                'lambda_total': lambda_total,
                'E_S': E_S_overall,
                'stable': all(m['stable'] for m in class_metrics.values())
            })

        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")

    return pd.DataFrame(results)


def _add_unstable_rows(
    results: list,
    load: float,
    scheduler: str,
    arousal_values: Dict[str, float]
):
    """Add rows for unstable (overloaded) system."""
    for class_name in list(arousal_values.keys()) + ['overall']:
        results.append({
            'load': load,
            'scheduler': scheduler,
            'class': class_name,
            'W_q': float('inf'),
            'W': float('inf'),
            'lambda_total': float('nan'),
            'E_S': float('nan'),
            'stable': False
        })


def predict_both_schedulers(
    base_service_time: float,
    alpha: float,
    load_levels: List[float],
    arousal_values: Optional[Dict[str, float]] = None,
    class_proportions: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Predict theory for both FCFS and SSJF-Emotion schedulers.

    Convenience function that calls predict_theory_load_sweep for both
    schedulers and concatenates results.

    Args:
        base_service_time: L_0
        alpha: α coefficient
        load_levels: List of load levels to predict
        arousal_values: Arousal values per class
        class_proportions: Class proportions

    Returns:
        Combined DataFrame with both schedulers
    """
    df_fcfs = predict_theory_load_sweep(
        base_service_time, alpha, load_levels,
        arousal_values, class_proportions, scheduler='FCFS'
    )

    df_ssjf = predict_theory_load_sweep(
        base_service_time, alpha, load_levels,
        arousal_values, class_proportions, scheduler='SSJF-Emotion'
    )

    return pd.concat([df_fcfs, df_ssjf], ignore_index=True)


def compute_theoretical_improvement(
    theory_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute theoretical improvement of SSJF-Emotion over FCFS.

    Args:
        theory_df: Output from predict_both_schedulers()

    Returns:
        DataFrame with columns:
            - load, class
            - fcfs_W_q, ssjf_W_q
            - improvement_abs: FCFS W_q - SSJF W_q
            - improvement_pct: (improvement_abs / fcfs_W_q) × 100
    """
    fcfs = theory_df[theory_df['scheduler'] == 'FCFS'].copy()
    ssjf = theory_df[theory_df['scheduler'] == 'SSJF-Emotion'].copy()

    merged = fcfs.merge(
        ssjf,
        on=['load', 'class'],
        suffixes=('_fcfs', '_ssjf')
    )

    merged['improvement_abs'] = merged['W_q_fcfs'] - merged['W_q_ssjf']
    merged['improvement_pct'] = (
        merged['improvement_abs'] / merged['W_q_fcfs'] * 100
    ).replace([np.inf, -np.inf], np.nan)

    return merged[[
        'load', 'class',
        'W_q_fcfs', 'W_q_ssjf',
        'improvement_abs', 'improvement_pct'
    ]]


def generate_theory_table(
    base_service_time: float,
    alpha: float,
    load_levels: List[float],
    scheduler: str = 'SSJF-Emotion'
) -> pd.DataFrame:
    """
    Generate a formatted table of theoretical predictions.

    Args:
        base_service_time: L_0
        alpha: α coefficient
        load_levels: List of load levels
        scheduler: Scheduler to generate table for

    Returns:
        Pivoted DataFrame with loads as rows and classes as columns
    """
    df = predict_theory_load_sweep(
        base_service_time, alpha, load_levels,
        scheduler=scheduler
    )

    # Filter to per-class results (not overall)
    df_classes = df[df['class'] != 'overall'].copy()

    # Pivot: rows=load, columns=class, values=W_q
    pivoted = df_classes.pivot(
        index='load', columns='class', values='W_q'
    )

    # Reorder columns
    column_order = ['low', 'medium', 'high']
    pivoted = pivoted[[c for c in column_order if c in pivoted.columns]]

    return pivoted
