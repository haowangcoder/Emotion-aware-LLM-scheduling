"""
Multi-Class Priority Queue Formulas (Kleinrock)

Implements Kleinrock's formulas for non-preemptive priority scheduling
in M/G/1 queues with multiple job classes.

Reference:
    Kleinrock, L. (1975). Queueing Systems, Volume I: Theory.
    Section 3.4: Priority Queueing
"""

from typing import Dict, List, Optional
import numpy as np


def kleinrock_multiclass_priority(
    arrival_rates: Dict[str, float],
    mean_service_times: Dict[str, float],
    second_moments: Dict[str, float],
    priority_order: List[str]
) -> Dict[str, dict]:
    """
    Compute per-class waiting times using Kleinrock's formula for
    non-preemptive priority scheduling in M/G/1 queues.

    Formula:
        E[W_q,c] = R_0 / [(1 - σ_{c-1})(1 - σ_c)]

    where:
        R_0 = (1/2) Σ_m λ_m E[S_m²]  (residual work)
        σ_c = Σ_{m: priority ≥ c} ρ_m  (cumulative utilization)

    Args:
        arrival_rates: {class_name: λ_c} - arrival rate for each class
        mean_service_times: {class_name: E[S_c]} - mean service time
        second_moments: {class_name: E[S_c²]} - second moment of service time
        priority_order: List of class names in priority order
                       [0] = highest priority (served first)

    Returns:
        Dict[class_name, metrics] where metrics = {
            'lambda': λ_c,
            'E_S': E[S_c],
            'E_S2': E[S_c²],
            'rho': ρ_c = λ_c × E[S_c],
            'rho_cumulative': σ_c,
            'W_q': E[W_q,c],
            'W': E[R_c] = E[S_c] + E[W_q,c],
            'priority': priority rank (1 = highest)
        }

    Example:
        >>> # SSJF-Emotion: low arousal = shortest = highest priority
        >>> arrival_rates = {'low': 0.15, 'medium': 0.15, 'high': 0.15}
        >>> mean_times = {'low': 0.72, 'medium': 2.0, 'high': 3.28}
        >>> second_moments = {c: t**2 for c, t in mean_times.items()}
        >>> priority_order = ['low', 'medium', 'high']
        >>> results = kleinrock_multiclass_priority(
        ...     arrival_rates, mean_times, second_moments, priority_order
        ... )
        >>> for c in priority_order:
        ...     print(f"{c}: W_q = {results[c]['W_q']:.2f}s")
    """
    # Validate inputs
    classes = set(arrival_rates.keys())
    assert classes == set(mean_service_times.keys()), "Class mismatch in mean_service_times"
    assert classes == set(second_moments.keys()), "Class mismatch in second_moments"
    assert set(priority_order) == classes, "priority_order must contain all classes"

    # Compute per-class utilization
    rho = {c: arrival_rates[c] * mean_service_times[c] for c in classes}

    # Compute total utilization
    rho_total = sum(rho.values())

    # Compute residual work R_0 = (1/2) Σ λ_m E[S_m²]
    R_0 = 0.5 * sum(
        arrival_rates[c] * second_moments[c]
        for c in classes
    )

    results = {}

    for i, class_c in enumerate(priority_order):
        # Cumulative utilization for higher-priority classes
        # σ_{c-1} = Σ_{m: priority > c} ρ_m
        sigma_higher = sum(rho[priority_order[j]] for j in range(i))

        # Cumulative utilization up to and including current class
        # σ_c = Σ_{m: priority ≥ c} ρ_m
        sigma_upto_c = sigma_higher + rho[class_c]

        # Check stability conditions
        if sigma_higher >= 1.0 or sigma_upto_c >= 1.0:
            W_q_c = float('inf')
            W_c = float('inf')
            stable = False
        else:
            # Kleinrock's formula for non-preemptive priority
            W_q_c = R_0 / ((1 - sigma_higher) * (1 - sigma_upto_c))
            W_c = mean_service_times[class_c] + W_q_c
            stable = True

        results[class_c] = {
            'lambda': arrival_rates[class_c],
            'E_S': mean_service_times[class_c],
            'E_S2': second_moments[class_c],
            'rho': rho[class_c],
            'rho_cumulative': sigma_upto_c,
            'rho_higher': sigma_higher,
            'R_0': R_0,
            'W_q': W_q_c,
            'W': W_c,
            'priority': i + 1,  # 1-indexed priority rank
            'stable': stable
        }

    return results


def weighted_average_response_time(
    class_metrics: Dict[str, dict]
) -> dict:
    """
    Compute overall weighted average metrics across all classes.

    Weights are proportional to arrival rates (λ_c / Σλ).

    Args:
        class_metrics: Output from kleinrock_multiclass_priority()

    Returns:
        dict with:
            - lambda_total: Total arrival rate
            - rho_total: Total utilization
            - W_q_avg: Weighted average waiting time
            - W_avg: Weighted average response time
    """
    lambda_total = sum(m['lambda'] for m in class_metrics.values())

    if lambda_total == 0:
        return {
            'lambda_total': 0,
            'rho_total': 0,
            'W_q_avg': 0,
            'W_avg': 0
        }

    W_q_avg = sum(
        m['lambda'] * m['W_q'] for m in class_metrics.values()
    ) / lambda_total

    W_avg = sum(
        m['lambda'] * m['W'] for m in class_metrics.values()
    ) / lambda_total

    rho_total = sum(m['rho'] for m in class_metrics.values())

    return {
        'lambda_total': lambda_total,
        'rho_total': rho_total,
        'W_q_avg': W_q_avg,
        'W_avg': W_avg
    }


def compute_priority_improvement(
    fcfs_W_q: float,
    priority_class_metrics: Dict[str, dict]
) -> Dict[str, dict]:
    """
    Compute improvement of priority scheduling over FCFS for each class.

    Args:
        fcfs_W_q: FCFS waiting time (same for all classes)
        priority_class_metrics: Output from kleinrock_multiclass_priority()

    Returns:
        Dict[class_name, improvement_metrics] where improvement_metrics = {
            'W_q_fcfs': FCFS waiting time,
            'W_q_priority': Priority scheduling waiting time,
            'improvement_abs': W_q_fcfs - W_q_priority,
            'improvement_pct': (W_q_fcfs - W_q_priority) / W_q_fcfs * 100
        }
    """
    results = {}

    for class_name, metrics in priority_class_metrics.items():
        W_q_priority = metrics['W_q']

        if np.isinf(fcfs_W_q) or np.isinf(W_q_priority):
            improvement_abs = float('nan')
            improvement_pct = float('nan')
        else:
            improvement_abs = fcfs_W_q - W_q_priority
            improvement_pct = (improvement_abs / fcfs_W_q * 100) if fcfs_W_q > 0 else 0

        results[class_name] = {
            'W_q_fcfs': fcfs_W_q,
            'W_q_priority': W_q_priority,
            'improvement_abs': improvement_abs,
            'improvement_pct': improvement_pct
        }

    return results


def ssjf_emotion_priority_order() -> List[str]:
    """
    Return the priority order for SSJF-Emotion scheduler.

    In SSJF (Shortest-Service-Job-First), jobs with shorter
    expected service times have higher priority.

    For emotion-aware scheduling:
        - Low arousal → shortest service time → highest priority
        - High arousal → longest service time → lowest priority

    Returns:
        ['low', 'medium', 'high'] - priority order (high to low)
    """
    return ['low', 'medium', 'high']


def ssjf_valence_priority_order() -> List[str]:
    """
    Return the priority order for SSJF-Valence scheduler.

    In valence-based scheduling, negative emotions get higher priority
    (ethical prioritization).

    For valence-based scheduling:
        - Negative valence → highest priority (sad, angry users first)
        - Positive valence → lowest priority

    Note: This is based on valence groups, not arousal.

    Returns:
        ['negative', 'neutral', 'positive'] - priority order
    """
    return ['negative', 'neutral', 'positive']


def predict_class_waiting_times(
    base_service_time: float,
    alpha: float,
    target_load: float,
    arousal_values: Dict[str, float],
    class_proportions: Optional[Dict[str, float]] = None,
    scheduler: str = 'SSJF-Emotion'
) -> Dict[str, dict]:
    """
    Predict per-class waiting times for a given scheduler and load.

    This is a convenience function that combines:
    1. Service time calculation from model parameters
    2. Arrival rate calculation from target load
    3. Kleinrock formula application

    Args:
        base_service_time: L_0 (base service time in seconds)
        alpha: α coefficient for arousal-service time mapping
        target_load: Target system utilization ρ
        arousal_values: {class_name: arousal_value}
        class_proportions: {class_name: proportion} (defaults to uniform)
        scheduler: 'SSJF-Emotion' or 'FCFS'

    Returns:
        Per-class metrics from kleinrock_multiclass_priority()
    """
    if class_proportions is None:
        n_classes = len(arousal_values)
        class_proportions = {c: 1.0 / n_classes for c in arousal_values}

    # Compute per-class service times
    service_times = {
        c: base_service_time * (1 + alpha * a)
        for c, a in arousal_values.items()
    }

    # Compute overall mean service time
    E_S_overall = sum(
        class_proportions[c] * service_times[c]
        for c in arousal_values
    )

    # Compute total arrival rate to achieve target load
    # ρ = λ × E[S] => λ = ρ / E[S]
    lambda_total = target_load / E_S_overall

    # Per-class arrival rates (proportional)
    arrival_rates = {
        c: lambda_total * class_proportions[c]
        for c in arousal_values
    }

    # Second moments (deterministic per class)
    second_moments = {c: s ** 2 for c, s in service_times.items()}

    # Apply Kleinrock formula
    if scheduler == 'SSJF-Emotion':
        priority_order = ssjf_emotion_priority_order()
    elif scheduler == 'FCFS':
        # For FCFS, order doesn't matter but we still compute per-class
        priority_order = list(arousal_values.keys())
    else:
        raise ValueError(f"Unknown scheduler: {scheduler}")

    return kleinrock_multiclass_priority(
        arrival_rates,
        service_times,
        second_moments,
        priority_order
    )
