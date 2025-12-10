"""
Fairness Metrics for Affect-Aware LLM Scheduling

This module provides fairness analysis tools for evaluating scheduling strategies
across different emotional states using Russell's Circumplex Model quadrants.

Key Metrics:
- Jain Fairness Index: Quantifies resource allocation fairness across groups
- Russell Quadrant Analysis: Compares performance across emotional quadrants

Russell Quadrants (Depression-First Strategy Target):
- excited: High valence + High arousal
- calm: High valence + Low arousal
- panic: Low valence + High arousal
- depression: Low valence + Low arousal (target for priority boost)

Jain Fairness Index: J = (sum x_i)^2 / (n * sum x_i^2)
where x_i is the resource allocation (e.g., average latency) for group i
J ranges from 1/n (worst) to 1 (perfect fairness)
"""

import numpy as np
from typing import List, Dict
from collections import defaultdict
from core.job import Job


def calculate_jain_fairness_index(values: List[float]) -> float:
    """
    Calculate Jain Fairness Index for a list of values.

    Formula: J = (sum x_i)^2 / (n * sum x_i^2)

    Args:
        values: List of resource allocation values for each group

    Returns:
        Fairness index in range [1/n, 1] where 1 is perfectly fair
        Returns 0 if values is empty or all values are 0

    Example:
        >>> calculate_jain_fairness_index([1, 1, 1])  # Perfect fairness
        1.0
        >>> calculate_jain_fairness_index([1, 2, 3])  # Some unfairness
        0.857
    """
    if not values or len(values) == 0:
        return 0.0

    values = np.array(values, dtype=float)

    # Handle edge case where all values are zero
    if np.all(values == 0):
        return 1.0  # Perfectly fair if everyone gets nothing

    n = len(values)
    sum_x = np.sum(values)
    sum_x_squared = np.sum(values ** 2)

    if sum_x_squared == 0:
        return 1.0

    jain_index = (sum_x ** 2) / (n * sum_x_squared)

    return jain_index


def calculate_weighted_jain_index(values: List[float], weights: List[float]) -> float:
    """
    Calculate weighted Jain Fairness Index.

    J_w = (sum w_i x_i)^2 / (n sum (w_i^2 x_i^2))
    Falls back to 1.0 when all values are zero; returns 0 if input is empty.
    """
    if not values or not weights or len(values) != len(weights):
        return 0.0

    x = np.array(values, dtype=float)
    w = np.array(weights, dtype=float)

    if np.all(x == 0):
        return 1.0

    numerator = (np.sum(w * x)) ** 2
    denominator = len(values) * np.sum((w ** 2) * (x ** 2))

    if denominator == 0:
        return 1.0

    return float(numerator / denominator)


def group_jobs_by_russell_quadrant(job_list: List[Job]) -> Dict[str, List[Job]]:
    """
    Group jobs by their Russell quadrant.

    Args:
        job_list: List of Job objects with russell_quadrant attribute

    Returns:
        Dictionary mapping quadrant -> list of jobs
    """
    grouped = defaultdict(list)

    for job in job_list:
        quadrant = getattr(job, 'russell_quadrant', None)
        if quadrant is not None:
            grouped[quadrant].append(job)
        else:
            grouped['unknown'].append(job)

    return dict(grouped)


def group_jobs_by_emotion_class(job_list: List[Job]) -> Dict[str, List[Job]]:
    """
    Group jobs by their emotion class (legacy, for backward compatibility).

    Args:
        job_list: List of Job objects with emotion_class attribute

    Returns:
        Dictionary mapping emotion_class -> list of jobs
    """
    grouped = defaultdict(list)

    for job in job_list:
        if job.emotion_class is not None:
            grouped[job.emotion_class].append(job)
        else:
            grouped['unknown'].append(job)

    return dict(grouped)


def group_jobs_by_emotion_label(job_list: List[Job]) -> Dict[str, List[Job]]:
    """
    Group jobs by their specific emotion label.

    Args:
        job_list: List of Job objects with emotion_label attribute

    Returns:
        Dictionary mapping emotion_label -> list of jobs
    """
    grouped = defaultdict(list)

    for job in job_list:
        if job.emotion_label is not None:
            grouped[job.emotion_label].append(job)
        else:
            grouped['unknown'].append(job)

    return dict(grouped)


def group_jobs_by_valence_class(job_list: List[Job]) -> Dict[str, List[Job]]:
    """
    Group jobs by valence_class (negative/neutral/positive).
    Jobs without valence info are grouped under 'unknown'.
    """
    grouped = defaultdict(list)

    for job in job_list:
        valence_class = getattr(job, "valence_class", None)
        if valence_class is not None:
            grouped[valence_class].append(job)
        else:
            grouped["unknown"].append(job)

    return dict(grouped)


def calculate_per_quadrant_metrics(job_list: List[Job], completed_only: bool = True) -> Dict:
    """
    Calculate performance metrics for each Russell quadrant.

    Args:
        job_list: List of Job objects
        completed_only: Only include completed jobs in calculations

    Returns:
        Dictionary with per-quadrant metrics:
        {
            'excited': {'count': X, 'avg_waiting': Y, ...},
            'calm': {...},
            'panic': {...},
            'depression': {...},
            'overall': {...}
        }
    """
    # Filter to completed jobs if requested
    if completed_only:
        jobs = [j for j in job_list if j.completion_time is not None]
    else:
        jobs = job_list

    if not jobs:
        return {}

    # Group by Russell quadrant
    grouped = group_jobs_by_russell_quadrant(jobs)

    metrics = {}

    for quadrant, quadrant_jobs in grouped.items():
        if not quadrant_jobs:
            continue

        # Calculate waiting times
        waiting_times = []
        turnaround_times = []
        affect_weights = []

        for job in quadrant_jobs:
            if job.waiting_duration is not None:
                waiting_times.append(job.waiting_duration)
            elif job.completion_time is not None:
                waiting = job.completion_time - job.arrival_time - job.execution_duration
                waiting_times.append(waiting)

            if job.completion_time is not None:
                turnaround = job.completion_time - job.arrival_time
                turnaround_times.append(turnaround)

            if hasattr(job, 'affect_weight') and job.affect_weight is not None:
                affect_weights.append(job.affect_weight)

        # Collect measured execution times
        measured_exec_times = [
            j.actual_execution_duration
            for j in quadrant_jobs
            if hasattr(j, 'actual_execution_duration') and j.actual_execution_duration is not None
        ]

        metrics[quadrant] = {
            'count': len(quadrant_jobs),
            'avg_waiting_time': np.mean(waiting_times) if waiting_times else 0,
            'std_waiting_time': np.std(waiting_times) if waiting_times else 0,
            'avg_turnaround_time': np.mean(turnaround_times) if turnaround_times else 0,
            'std_turnaround_time': np.std(turnaround_times) if turnaround_times else 0,
            'avg_predicted_service_time': np.mean([j.execution_duration for j in quadrant_jobs]),
            'avg_measured_execution_time': np.mean(measured_exec_times) if measured_exec_times else None,
            'avg_affect_weight': np.mean(affect_weights) if affect_weights else 1.0,
            'p50_waiting_time': np.percentile(waiting_times, 50) if waiting_times else 0,
            'p95_waiting_time': np.percentile(waiting_times, 95) if waiting_times else 0,
            'p99_waiting_time': np.percentile(waiting_times, 99) if waiting_times else 0,
        }

    # Calculate overall metrics
    all_waiting = []
    all_turnaround = []
    for quadrant, quadrant_jobs in grouped.items():
        for job in quadrant_jobs:
            if job.waiting_duration is not None:
                all_waiting.append(job.waiting_duration)
            if job.completion_time is not None:
                all_turnaround.append(job.completion_time - job.arrival_time)

    metrics['overall'] = {
        'count': len(jobs),
        'avg_waiting_time': np.mean(all_waiting) if all_waiting else 0,
        'avg_turnaround_time': np.mean(all_turnaround) if all_turnaround else 0,
    }

    return metrics


def calculate_per_class_metrics(job_list: List[Job], completed_only: bool = True) -> Dict:
    """
    Calculate performance metrics for each emotion class (legacy).
    Wrapper for backward compatibility - now prefers Russell quadrants.

    Args:
        job_list: List of Job objects
        completed_only: Only include completed jobs in calculations

    Returns:
        Dictionary with per-class metrics
    """
    # Try to use Russell quadrants if available
    jobs = [j for j in job_list if j.completion_time is not None] if completed_only else job_list
    if jobs and hasattr(jobs[0], 'russell_quadrant') and jobs[0].russell_quadrant is not None:
        return calculate_per_quadrant_metrics(job_list, completed_only)

    # Fall back to emotion_class for legacy support
    if not jobs:
        return {}

    grouped = group_jobs_by_emotion_class(jobs)
    metrics = {}

    for emotion_class, class_jobs in grouped.items():
        if not class_jobs:
            continue

        waiting_times = []
        turnaround_times = []

        for job in class_jobs:
            if job.waiting_duration is not None:
                waiting_times.append(job.waiting_duration)
            elif job.completion_time is not None:
                waiting = job.completion_time - job.arrival_time - job.execution_duration
                waiting_times.append(waiting)

            if job.completion_time is not None:
                turnaround = job.completion_time - job.arrival_time
                turnaround_times.append(turnaround)

        measured_exec_times = [
            j.actual_execution_duration
            for j in class_jobs
            if hasattr(j, 'actual_execution_duration') and j.actual_execution_duration is not None
        ]

        metrics[emotion_class] = {
            'count': len(class_jobs),
            'avg_waiting_time': np.mean(waiting_times) if waiting_times else 0,
            'std_waiting_time': np.std(waiting_times) if waiting_times else 0,
            'avg_turnaround_time': np.mean(turnaround_times) if turnaround_times else 0,
            'std_turnaround_time': np.std(turnaround_times) if turnaround_times else 0,
            'avg_predicted_service_time': np.mean([j.execution_duration for j in class_jobs]),
            'avg_measured_execution_time': np.mean(measured_exec_times) if measured_exec_times else None,
            'p50_waiting_time': np.percentile(waiting_times, 50) if waiting_times else 0,
            'p95_waiting_time': np.percentile(waiting_times, 95) if waiting_times else 0,
            'p99_waiting_time': np.percentile(waiting_times, 99) if waiting_times else 0,
        }

    all_waiting = []
    all_turnaround = []
    for emotion_class, class_jobs in grouped.items():
        for job in class_jobs:
            if job.waiting_duration is not None:
                all_waiting.append(job.waiting_duration)
            if job.completion_time is not None:
                all_turnaround.append(job.completion_time - job.arrival_time)

    metrics['overall'] = {
        'count': len(jobs),
        'avg_waiting_time': np.mean(all_waiting) if all_waiting else 0,
        'avg_turnaround_time': np.mean(all_turnaround) if all_turnaround else 0,
    }

    return metrics


def calculate_fairness_across_quadrants(job_list: List[Job],
                                        metric: str = 'waiting_time') -> Dict:
    """
    Calculate fairness metrics across Russell quadrants.

    Args:
        job_list: List of completed Job objects
        metric: Metric to use for fairness ('waiting_time' or 'turnaround_time')

    Returns:
        Dictionary with:
        - 'jain_index': Jain Fairness Index
        - 'per_quadrant_values': Average metric value per quadrant
        - 'coefficient_of_variation': CV of metric across quadrants
        - 'max_min_ratio': Ratio of max to min average metric
        - 'depression_vs_others': Ratio of depression to others
    """
    per_quadrant = calculate_per_quadrant_metrics(job_list, completed_only=True)

    if not per_quadrant or len(per_quadrant) <= 1:
        return {
            'jain_index': 1.0,
            'per_quadrant_values': {},
            'coefficient_of_variation': 0.0,
            'max_min_ratio': 1.0,
            'depression_vs_others': 1.0,
        }

    # Extract metric values per quadrant
    metric_key = f'avg_{metric}'
    quadrant_values = {}

    for quadrant, metrics in per_quadrant.items():
        if quadrant != 'overall' and metric_key in metrics:
            quadrant_values[quadrant] = metrics[metric_key]

    if not quadrant_values:
        return {
            'jain_index': 1.0,
            'per_quadrant_values': {},
            'coefficient_of_variation': 0.0,
            'max_min_ratio': 1.0,
            'depression_vs_others': 1.0,
        }

    # Calculate Jain Fairness Index
    values_list = list(quadrant_values.values())
    jain_index = calculate_jain_fairness_index(values_list)

    # Calculate coefficient of variation
    mean_val = np.mean(values_list)
    std_val = np.std(values_list)
    cv = std_val / mean_val if mean_val > 0 else 0

    # Calculate max/min ratio
    max_val = max(values_list)
    min_val = min(values_list)
    max_min_ratio = max_val / min_val if min_val > 0 else float('inf')

    # Calculate depression vs others ratio
    depression_val = quadrant_values.get('depression', 0)
    other_vals = [v for k, v in quadrant_values.items() if k != 'depression']
    other_mean = np.mean(other_vals) if other_vals else 0
    depression_vs_others = depression_val / other_mean if other_mean > 0 else 1.0

    return {
        'jain_index': jain_index,
        'per_quadrant_values': quadrant_values,
        'coefficient_of_variation': cv,
        'max_min_ratio': max_min_ratio,
        'depression_vs_others': depression_vs_others,
        'mean': mean_val,
        'std': std_val
    }


def calculate_fairness_across_emotions(job_list: List[Job],
                                        metric: str = 'waiting_time') -> Dict:
    """
    Calculate fairness metrics across emotion classes (legacy wrapper).

    Args:
        job_list: List of completed Job objects
        metric: Metric to use for fairness

    Returns:
        Dictionary with fairness metrics
    """
    # Try Russell quadrants first
    jobs = [j for j in job_list if j.completion_time is not None]
    if jobs and hasattr(jobs[0], 'russell_quadrant') and jobs[0].russell_quadrant is not None:
        return calculate_fairness_across_quadrants(job_list, metric)

    # Fall back to emotion_class
    per_class = calculate_per_class_metrics(job_list, completed_only=True)

    if not per_class or len(per_class) <= 1:
        return {
            'jain_index': 1.0,
            'per_class_values': {},
            'coefficient_of_variation': 0.0,
            'max_min_ratio': 1.0
        }

    metric_key = f'avg_{metric}'
    class_values = {}

    for emotion_class, metrics in per_class.items():
        if emotion_class != 'overall' and metric_key in metrics:
            class_values[emotion_class] = metrics[metric_key]

    if not class_values:
        return {
            'jain_index': 1.0,
            'per_class_values': {},
            'coefficient_of_variation': 0.0,
            'max_min_ratio': 1.0
        }

    values_list = list(class_values.values())
    jain_index = calculate_jain_fairness_index(values_list)

    mean_val = np.mean(values_list)
    std_val = np.std(values_list)
    cv = std_val / mean_val if mean_val > 0 else 0

    max_val = max(values_list)
    min_val = min(values_list)
    max_min_ratio = max_val / min_val if min_val > 0 else float('inf')

    return {
        'jain_index': jain_index,
        'per_class_values': class_values,
        'coefficient_of_variation': cv,
        'max_min_ratio': max_min_ratio,
        'mean': mean_val,
        'std': std_val
    }


def calculate_valence_fairness(job_list: List[Job],
                               beta: float = 0.0,
                               metric: str = 'waiting_time') -> Dict:
    """
    Calculate weighted fairness across valence classes.
    Legacy function for backward compatibility.
    """
    if not job_list:
        return {
            'weighted_jain_index': 1.0,
            'per_valence_values': {},
            'weights': {},
            'mean': 0.0,
            'std': 0.0,
        }

    grouped = group_jobs_by_valence_class(job_list)
    if not grouped:
        return {
            'weighted_jain_index': 1.0,
            'per_valence_values': {},
            'weights': {},
            'mean': 0.0,
            'std': 0.0,
        }

    per_valence = {}
    weights = {}
    for valence_class, jobs in grouped.items():
        if not jobs:
            continue
        metric_values = []
        valences = []
        for job in jobs:
            if metric == 'waiting_time' and job.waiting_duration is not None:
                metric_values.append(job.waiting_duration)
            elif metric == 'turnaround_time' and job.completion_time is not None:
                metric_values.append(job.completion_time - job.arrival_time)
            val_val = getattr(job, "valence", None)
            if val_val is not None:
                valences.append(val_val)

        if not metric_values:
            continue

        avg_metric = float(np.mean(metric_values))
        if valences:
            avg_valence = float(np.mean(valences))
        else:
            if valence_class == 'negative':
                avg_valence = -0.8
            elif valence_class == 'positive':
                avg_valence = 0.8
            else:
                avg_valence = 0.0

        weight = 1.0 + beta * (-avg_valence)
        if weight <= 0:
            weight = 1e-6
        per_valence[valence_class] = avg_metric
        weights[valence_class] = weight

    if not per_valence:
        return {
            'weighted_jain_index': 1.0,
            'per_valence_values': {},
            'weights': {},
            'mean': 0.0,
            'std': 0.0,
        }

    values_list = list(per_valence.values())
    weights_list = [weights[k] for k in per_valence.keys()]

    wj = calculate_weighted_jain_index(values_list, weights_list)
    mean_val = float(np.mean(values_list)) if values_list else 0.0
    std_val = float(np.std(values_list)) if values_list else 0.0

    return {
        'weighted_jain_index': wj,
        'per_valence_values': per_valence,
        'weights': weights,
        'mean': mean_val,
        'std': std_val,
    }


def calculate_russell_quadrant_distribution(job_list: List[Job]) -> Dict:
    """
    Calculate the distribution of jobs across Russell quadrants.

    Args:
        job_list: List of Job objects with russell_quadrant attribute

    Returns:
        Dictionary with counts and percentages for each quadrant
    """
    distribution = {
        'excited': 0, 'calm': 0, 'panic': 0, 'depression': 0,
    }

    for job in job_list:
        quadrant = getattr(job, 'russell_quadrant', None)
        if quadrant in distribution:
            distribution[quadrant] += 1

    total = sum(distribution.values())
    distribution['total'] = total

    # Add percentages
    for quadrant in ['excited', 'calm', 'panic', 'depression']:
        pct_key = f'{quadrant}_pct'
        distribution[pct_key] = distribution[quadrant] / total * 100 if total > 0 else 0

    return distribution


def calculate_arousal_valence_distribution(job_list: List[Job]) -> Dict:
    """
    Calculate the joint distribution of arousal and valence classes.
    Legacy function for backward compatibility.
    """
    distribution = {
        'high_positive': 0, 'high_neutral': 0, 'high_negative': 0,
        'medium_positive': 0, 'medium_neutral': 0, 'medium_negative': 0,
        'low_positive': 0, 'low_neutral': 0, 'low_negative': 0,
    }

    for job in job_list:
        arousal_class = getattr(job, 'emotion_class', None)
        valence_class = getattr(job, 'valence_class', None)

        if arousal_class and valence_class:
            key = f"{arousal_class}_{valence_class}"
            if key in distribution:
                distribution[key] += 1

    distribution['total'] = sum(v for k, v in distribution.items() if k != 'total')

    return distribution


def analyze_fairness_comprehensive(job_list: List[Job]) -> Dict:
    """
    Comprehensive fairness analysis across multiple metrics.

    Args:
        job_list: List of completed Job objects

    Returns:
        Dictionary with fairness analysis for multiple metrics
    """
    analysis = {}

    # Analyze waiting time fairness
    analysis['waiting_time_fairness'] = calculate_fairness_across_quadrants(
        job_list, metric='waiting_time'
    )

    # Analyze turnaround time fairness
    analysis['turnaround_time_fairness'] = calculate_fairness_across_quadrants(
        job_list, metric='turnaround_time'
    )

    # Per-quadrant detailed metrics
    analysis['per_quadrant_metrics'] = calculate_per_quadrant_metrics(job_list)

    # Russell quadrant distribution
    analysis['russell_quadrant_distribution'] = calculate_russell_quadrant_distribution(job_list)

    # Legacy arousal-valence distribution (for backward compatibility)
    analysis['arousal_valence_distribution'] = calculate_arousal_valence_distribution(job_list)

    return analysis


def compare_scheduler_fairness(job_lists: Dict[str, List[Job]]) -> Dict:
    """
    Compare fairness metrics across different schedulers.

    Args:
        job_lists: Dictionary mapping scheduler_name -> list of completed jobs

    Returns:
        Dictionary with fairness comparison results
    """
    comparison = {}

    for scheduler_name, job_list in job_lists.items():
        comparison[scheduler_name] = analyze_fairness_comprehensive(job_list)

    # Summary comparison
    summary = {
        'jain_index_waiting_time': {},
        'jain_index_turnaround_time': {},
        'depression_vs_others': {},
    }

    for scheduler_name, analysis in comparison.items():
        summary['jain_index_waiting_time'][scheduler_name] = \
            analysis['waiting_time_fairness']['jain_index']
        summary['jain_index_turnaround_time'][scheduler_name] = \
            analysis['turnaround_time_fairness']['jain_index']
        summary['depression_vs_others'][scheduler_name] = \
            analysis['waiting_time_fairness'].get('depression_vs_others', 1.0)

    comparison['summary'] = summary

    return comparison
