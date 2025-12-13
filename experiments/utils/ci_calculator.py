"""
Confidence interval calculation utilities for statistical validation.

This module provides functions to:
- Compute confidence intervals (95% CI by default)
- Perform statistical tests
- Generate forest plot data
"""

import numpy as np
import scipy.stats as stats
from typing import List, Tuple, Dict, Any, Optional


def compute_95_ci(values: List[float]) -> Tuple[float, float, float]:
    """
    Compute 95% confidence interval using t-distribution.

    Args:
        values: List of sample values

    Returns:
        Tuple of (mean, ci_lower, ci_upper)
    """
    n = len(values)
    if n < 2:
        mean = values[0] if n == 1 else 0.0
        return mean, mean, mean

    mean = np.mean(values)
    se = stats.sem(values)
    ci = stats.t.interval(0.95, n - 1, loc=mean, scale=se)

    return float(mean), float(ci[0]), float(ci[1])


def compute_ci(
    values: List[float],
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute confidence interval at specified confidence level.

    Args:
        values: List of sample values
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (mean, ci_lower, ci_upper)
    """
    n = len(values)
    if n < 2:
        mean = values[0] if n == 1 else 0.0
        return mean, mean, mean

    mean = np.mean(values)
    se = stats.sem(values)
    ci = stats.t.interval(confidence, n - 1, loc=mean, scale=se)

    return float(mean), float(ci[0]), float(ci[1])


def compute_bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute confidence interval using bootstrap resampling.

    Args:
        values: List of sample values
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level

    Returns:
        Tuple of (mean, ci_lower, ci_upper)
    """
    arr = np.array(values)
    n = len(arr)
    mean = float(np.mean(arr))

    if n < 2:
        return mean, mean, mean

    # Bootstrap resampling
    bootstrap_means = np.array([
        np.mean(np.random.choice(arr, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    # Percentile method
    alpha = (1 - confidence) / 2
    ci_lower = float(np.percentile(bootstrap_means, alpha * 100))
    ci_upper = float(np.percentile(bootstrap_means, (1 - alpha) * 100))

    return mean, ci_lower, ci_upper


def perform_t_test(
    values1: List[float],
    values2: List[float],
    alternative: str = 'two-sided'
) -> Tuple[float, float]:
    """
    Perform independent t-test between two groups.

    Args:
        values1: First group values
        values2: Second group values
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        Tuple of (t_statistic, p_value)
    """
    result = stats.ttest_ind(values1, values2, alternative=alternative)
    return float(result.statistic), float(result.pvalue)


def perform_paired_t_test(
    values1: List[float],
    values2: List[float],
    alternative: str = 'two-sided'
) -> Tuple[float, float]:
    """
    Perform paired t-test between two groups.

    Args:
        values1: First group values
        values2: Second group values (must have same length)
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        Tuple of (t_statistic, p_value)
    """
    result = stats.ttest_rel(values1, values2, alternative=alternative)
    return float(result.statistic), float(result.pvalue)


def compute_effect_size(
    values1: List[float],
    values2: List[float]
) -> float:
    """
    Compute Cohen's d effect size.

    Args:
        values1: First group values
        values2: Second group values

    Returns:
        Cohen's d effect size
    """
    n1, n2 = len(values1), len(values2)
    mean1, mean2 = np.mean(values1), np.mean(values2)
    var1, var2 = np.var(values1, ddof=1), np.var(values2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return float((mean1 - mean2) / pooled_std)


def aggregate_multi_seed_results(
    results_per_seed: Dict[int, Dict[str, float]],
    metrics: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate results from multiple random seeds and compute CIs.

    Args:
        results_per_seed: Dictionary mapping seed to metric dictionary
        metrics: List of metric names to aggregate

    Returns:
        Dictionary mapping metric names to {mean, ci_lower, ci_upper, std, n}
    """
    aggregated = {}

    for metric in metrics:
        values = [
            results[metric]
            for seed, results in results_per_seed.items()
            if metric in results
        ]

        if values:
            mean, ci_lower, ci_upper = compute_95_ci(values)
            aggregated[metric] = {
                'mean': mean,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'std': float(np.std(values)),
                'n': len(values),
            }

    return aggregated


def create_forest_plot_data(
    results: Dict[str, Dict[str, float]],
    metric: str
) -> List[Dict[str, Any]]:
    """
    Create data structure for forest plot visualization.

    Args:
        results: Dictionary mapping configuration names to metric results
        metric: Metric name to plot

    Returns:
        List of dictionaries with {name, mean, ci_lower, ci_upper} for plotting
    """
    plot_data = []

    for name, metrics in results.items():
        if metric in metrics or isinstance(metrics, dict):
            metric_data = metrics.get(metric, metrics)
            if isinstance(metric_data, dict):
                plot_data.append({
                    'name': name,
                    'mean': metric_data.get('mean', 0),
                    'ci_lower': metric_data.get('ci_lower', 0),
                    'ci_upper': metric_data.get('ci_upper', 0),
                })

    return plot_data


def is_significant(
    values1: List[float],
    values2: List[float],
    alpha: float = 0.05
) -> bool:
    """
    Test if two groups are significantly different.

    Args:
        values1: First group values
        values2: Second group values
        alpha: Significance level

    Returns:
        True if difference is statistically significant
    """
    _, p_value = perform_t_test(values1, values2)
    return p_value < alpha
