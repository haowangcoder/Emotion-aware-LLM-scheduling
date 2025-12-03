"""
Fairness Metrics for Emotion-aware LLM Scheduling

This module provides fairness analysis tools for evaluating scheduling strategies
across different emotion categories. The primary metric is the Jain Fairness Index,
which quantifies how fairly resources are allocated across different groups.

Jain Fairness Index: J = (Σx_i)² / (n * Σx_i²)
where x_i is the resource allocation (e.g., average latency) for group i

J ranges from 1/n (worst) to 1 (perfect fairness)
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
from core.job import Job


def calculate_jain_fairness_index(values: List[float]) -> float:
    """
    Calculate Jain Fairness Index for a list of values

    Formula: J = (Σx_i)² / (n * Σx_i²)

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

    J_w = (Σ w_i x_i)^2 / (n Σ (w_i^2 x_i^2))
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


def group_jobs_by_emotion_class(job_list: List[Job]) -> Dict[str, List[Job]]:
    """
    Group jobs by their emotion class

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
    Group jobs by their specific emotion label

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


def calculate_per_class_metrics(job_list: List[Job], completed_only: bool = True) -> Dict:
    """
    Calculate performance metrics for each emotion class

    Args:
        job_list: List of Job objects
        completed_only: Only include completed jobs in calculations

    Returns:
        Dictionary with per-class metrics:
        {
            'high': {'count': X, 'avg_waiting': Y, 'avg_completion': Z, ...},
            'medium': {...},
            'low': {...},
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

    # Group by emotion class
    grouped = group_jobs_by_emotion_class(jobs)

    metrics = {}

    for emotion_class, class_jobs in grouped.items():
        if not class_jobs:
            continue

        # Calculate waiting times (completion_time - arrival_time - execution_duration)
        waiting_times = []
        completion_times = []
        turnaround_times = []

        for job in class_jobs:
            if job.waiting_duration is not None:
                waiting_times.append(job.waiting_duration)
            elif job.completion_time is not None:
                # Calculate waiting time if not explicitly set
                waiting = job.completion_time - job.arrival_time - job.execution_duration
                waiting_times.append(waiting)

            if job.completion_time is not None:
                turnaround = job.completion_time - job.arrival_time
                turnaround_times.append(turnaround)

        # Collect measured execution times (actual LLM inference times)
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
            # Predicted service time based on arousal mapping (used for scheduling decisions)
            'avg_predicted_service_time': np.mean([j.execution_duration for j in class_jobs]),
            # Measured execution time from actual LLM inference (if available)
            'avg_measured_execution_time': np.mean(measured_exec_times) if measured_exec_times else None,
            'p50_waiting_time': np.percentile(waiting_times, 50) if waiting_times else 0,
            'p95_waiting_time': np.percentile(waiting_times, 95) if waiting_times else 0,
            'p99_waiting_time': np.percentile(waiting_times, 99) if waiting_times else 0,
        }

    # Calculate overall metrics
    all_waiting = []
    all_turnaround = []
    for class_metrics in metrics.values():
        if metrics[emotion_class]['count'] > 0:
            all_waiting.extend(waiting_times)
            all_turnaround.extend(turnaround_times)

    metrics['overall'] = {
        'count': len(jobs),
        'avg_waiting_time': np.mean(all_waiting) if all_waiting else 0,
        'avg_turnaround_time': np.mean(all_turnaround) if all_turnaround else 0,
    }

    return metrics


def calculate_fairness_across_emotions(job_list: List[Job],
                                        metric: str = 'waiting_time') -> Dict:
    """
    Calculate fairness metrics across emotion classes

    Args:
        job_list: List of completed Job objects
        metric: Metric to use for fairness ('waiting_time', 'turnaround_time', or 'completion_time')

    Returns:
        Dictionary with:
        - 'jain_index': Jain Fairness Index
        - 'per_class_values': Average metric value per class
        - 'coefficient_of_variation': CV of metric across classes
        - 'max_min_ratio': Ratio of max to min average metric
    """
    # Calculate per-class metrics
    per_class = calculate_per_class_metrics(job_list, completed_only=True)

    if not per_class or len(per_class) <= 1:
        return {
            'jain_index': 1.0,
            'per_class_values': {},
            'coefficient_of_variation': 0.0,
            'max_min_ratio': 1.0
        }

    # Extract metric values per class
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

    # Calculate Jain Fairness Index
    values_list = list(class_values.values())
    jain_index = calculate_jain_fairness_index(values_list)

    # Calculate coefficient of variation
    mean_val = np.mean(values_list)
    std_val = np.std(values_list)
    cv = std_val / mean_val if mean_val > 0 else 0

    # Calculate max/min ratio
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
    Calculate weighted fairness across valence classes using π_i = 1 + β(-v_i).

    - Groups by valence_class (negative/neutral/positive).
    - Uses group average valence to derive weight.
    - metric: 'waiting_time' or 'turnaround_time'.
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
        # If no valence values on jobs, fall back to representative constants
        if valences:
            avg_valence = float(np.mean(valences))
        else:
            # Defaults align with discrete mapping used elsewhere
            if valence_class == 'negative':
                avg_valence = -0.8
            elif valence_class == 'positive':
                avg_valence = 0.8
            else:
                avg_valence = 0.0

        weight = 1.0 + beta * (-avg_valence)
        if weight <= 0:
            weight = 1e-6  # avoid zero/negative weights
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


def calculate_arousal_valence_distribution(job_list: List[Job]) -> Dict:
    """
    Calculate the joint distribution of arousal and valence classes.

    Returns a 3x3 matrix counting jobs in each (arousal_class, valence_class) combination.
    This helps validate assumptions about correlation between arousal and valence
    (e.g., "high arousal is usually negative").

    Args:
        job_list: List of Job objects with emotion_class and valence_class attributes

    Returns:
        Dictionary with counts for each combination:
        {
            'high_positive': count, 'high_neutral': count, 'high_negative': count,
            'medium_positive': count, 'medium_neutral': count, 'medium_negative': count,
            'low_positive': count, 'low_neutral': count, 'low_negative': count,
            'total': total_count
        }
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
    Comprehensive fairness analysis across multiple metrics

    Args:
        job_list: List of completed Job objects

    Returns:
        Dictionary with fairness analysis for multiple metrics
    """
    analysis = {}

    # Analyze waiting time fairness
    analysis['waiting_time_fairness'] = calculate_fairness_across_emotions(
        job_list, metric='waiting_time'
    )

    # Analyze turnaround time fairness
    analysis['turnaround_time_fairness'] = calculate_fairness_across_emotions(
        job_list, metric='turnaround_time'
    )

    # Per-class detailed metrics
    analysis['per_class_metrics'] = calculate_per_class_metrics(job_list)

    # Arousal-Valence joint distribution (validates correlation assumptions)
    analysis['arousal_valence_distribution'] = calculate_arousal_valence_distribution(job_list)

    return analysis


def compare_scheduler_fairness(job_lists: Dict[str, List[Job]]) -> Dict:
    """
    Compare fairness metrics across different schedulers

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
        'jain_index_turnaround_time': {}
    }

    for scheduler_name, analysis in comparison.items():
        summary['jain_index_waiting_time'][scheduler_name] = \
            analysis['waiting_time_fairness']['jain_index']
        summary['jain_index_turnaround_time'][scheduler_name] = \
            analysis['turnaround_time_fairness']['jain_index']

    comparison['summary'] = summary

    return comparison


# Example usage and testing
if __name__ == '__main__':
    from job import Job

    print("=" * 70)
    print("Fairness Metrics Test")
    print("=" * 70)

    # Test 1: Jain Fairness Index calculation
    print("\n1. Jain Fairness Index Calculation")
    test_cases = [
        [1, 1, 1],  # Perfect fairness
        [1, 2, 3],  # Some unfairness
        [1, 1, 10],  # High unfairness
        [5, 5, 5, 5, 5]  # Perfect fairness, more groups
    ]

    for values in test_cases:
        jain = calculate_jain_fairness_index(values)
        print(f"   Values: {values} -> Jain Index: {jain:.4f}")

    # Test 2: Create test jobs with emotions
    print("\n2. Emotion-aware Job Fairness Analysis")
    jobs = [
        Job(0, 2.0, 0.0, emotion_label='excited', arousal=0.9, emotion_class='high'),
        Job(1, 1.5, 1.0, emotion_label='sad', arousal=-0.6, emotion_class='low'),
        Job(2, 2.5, 2.0, emotion_label='angry', arousal=0.8, emotion_class='high'),
        Job(3, 1.0, 3.0, emotion_label='calm', arousal=-0.3, emotion_class='low'),
        Job(4, 1.8, 4.0, emotion_label='neutral', arousal=0.0, emotion_class='medium'),
    ]

    # Compute completion metrics
    current_time = 0
    for job in jobs:
        waiting = current_time - job.arrival_time
        job.waiting_duration = max(0, waiting)
        job.completion_time = current_time + job.execution_duration
        current_time = job.completion_time

    # Calculate per-class metrics
    metrics = calculate_per_class_metrics(jobs)
    print("\n   Per-Class Metrics:")
    for emotion_class, class_metrics in metrics.items():
        if emotion_class != 'overall':
            print(f"     {emotion_class}:")
            print(f"       Count: {class_metrics['count']}")
            print(f"       Avg waiting: {class_metrics['avg_waiting_time']:.3f}")
            print(f"       Avg turnaround: {class_metrics['avg_turnaround_time']:.3f}")

    # Calculate fairness
    print("\n3. Fairness Analysis")
    fairness = calculate_fairness_across_emotions(jobs, metric='waiting_time')
    print(f"   Waiting Time Fairness:")
    print(f"     Jain Index: {fairness['jain_index']:.4f}")
    print(f"     Coefficient of Variation: {fairness['coefficient_of_variation']:.4f}")
    print(f"     Max/Min Ratio: {fairness['max_min_ratio']:.4f}")
    print(f"     Per-class values:")
    for cls, val in fairness['per_class_values'].items():
        print(f"       {cls}: {val:.3f}")

    # Test 3: Compare different scenarios
    print("\n4. Scheduler Fairness Comparison")

    # Scenario 1: Fair scheduler (FCFS-like)
    jobs_fair = []
    for i in range(15):
        emotion_class = ['high', 'medium', 'low'][i % 3]
        arousal = [0.8, 0.0, -0.6][i % 3]
        job = Job(i, 2.0, i*1.0, emotion_class=emotion_class, arousal=arousal)
        job.waiting_duration = 0
        job.completion_time = job.arrival_time + job.execution_duration
        jobs_fair.append(job)

    # Scenario 2: Unfair scheduler (prioritizes high arousal)
    jobs_unfair = []
    high_jobs = []
    other_jobs = []
    for i in range(15):
        emotion_class = ['high', 'medium', 'low'][i % 3]
        arousal = [0.8, 0.0, -0.6][i % 3]
        job = Job(i, 2.0, i*1.0, emotion_class=emotion_class, arousal=arousal)
        if emotion_class == 'high':
            high_jobs.append(job)
        else:
            other_jobs.append(job)

    # Schedule high arousal first
    current_time = 0
    for job in high_jobs + other_jobs:
        waiting = max(0, current_time - job.arrival_time)
        job.waiting_duration = waiting
        job.completion_time = current_time + job.execution_duration
        current_time = job.completion_time
    jobs_unfair = high_jobs + other_jobs

    # Compare fairness
    comparison = compare_scheduler_fairness({
        'Fair (FCFS)': jobs_fair,
        'Unfair (Priority)': jobs_unfair
    })

    print("\n   Fairness Comparison:")
    print(f"   {'Scheduler':<20} {'Jain Index (Waiting)':<25}")
    print(f"   {'-'*45}")
    for scheduler, jain in comparison['summary']['jain_index_waiting_time'].items():
        print(f"   {scheduler:<20} {jain:<25.4f}")

    print("\n" + "=" * 70)
