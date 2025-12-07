"""
Error Analysis and Correction Utilities

Implements techniques to reduce error between theoretical predictions
and simulation results:
- Transient period removal (warm-up/cool-down)
- Finite sample corrections
- Bootstrap confidence intervals
- Error comparison metrics
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd


def remove_transient_period(
    jobs_df: pd.DataFrame,
    warmup_fraction: float = 0.1,
    cooldown_fraction: float = 0.1,
    sort_by: str = 'arrival_time'
) -> pd.DataFrame:
    """
    Remove warm-up and cool-down periods from simulation data.

    M/G/1 theory assumes steady-state behavior. Simulation data includes:
    - Warm-up: System starts empty, initial jobs have lower waiting times
    - Cool-down: System drains at end, final jobs may have lower waiting times

    Args:
        jobs_df: DataFrame with job data
        warmup_fraction: Fraction of jobs to remove from start (default 10%)
        cooldown_fraction: Fraction of jobs to remove from end (default 10%)
        sort_by: Column to sort by before removing (default 'arrival_time')

    Returns:
        Filtered DataFrame with steady-state jobs only

    Example:
        >>> df_steady = remove_transient_period(df, warmup_fraction=0.1, cooldown_fraction=0.1)
        >>> # Removes first 10% and last 10% of jobs
    """
    df = jobs_df.copy()

    # Sort by arrival time to ensure correct ordering
    if sort_by in df.columns:
        df = df.sort_values(sort_by).reset_index(drop=True)

    n = len(df)
    warmup_count = int(n * warmup_fraction)
    cooldown_count = int(n * cooldown_fraction)

    start_idx = warmup_count
    end_idx = n - cooldown_count

    if start_idx >= end_idx:
        raise ValueError(
            f"Transient removal too aggressive: "
            f"warmup={warmup_count}, cooldown={cooldown_count}, total={n}"
        )

    return df.iloc[start_idx:end_idx].copy()


def finite_sample_correction(
    theoretical_W_q: float,
    n_jobs: int,
    rho: float,
    method: str = 'simple'
) -> float:
    """
    Apply finite-sample correction to theoretical waiting time.

    M/G/1 theory assumes infinite steady-state. For finite N jobs,
    the effective behavior differs slightly.

    Args:
        theoretical_W_q: Theoretical steady-state waiting time
        n_jobs: Number of jobs in simulation
        rho: System utilization
        method: Correction method
               - 'simple': W_q_finite ≈ W_q_infinite × (N-1)/N
               - 'none': No correction

    Returns:
        Corrected waiting time estimate

    Note:
        These corrections are heuristic. For rigorous analysis,
        use multi-seed experiments with confidence intervals.
    """
    if n_jobs <= 1:
        return theoretical_W_q

    if method == 'none':
        return theoretical_W_q

    elif method == 'simple':
        # Simple correction for finite population effect
        # The first job always has W_q = 0, so average is slightly lower
        correction = (n_jobs - 1) / n_jobs
        return theoretical_W_q * correction

    else:
        raise ValueError(f"Unknown correction method: {method}")


def bootstrap_confidence_interval(
    values: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    statistic: str = 'mean'
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        values: Array of observed values
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (default 0.95 for 95% CI)
        statistic: Statistic to compute ('mean', 'median', 'std')

    Returns:
        (point_estimate, ci_low, ci_high)

    Example:
        >>> waiting_times = np.array([1.2, 1.5, 2.0, 1.8, 1.3])
        >>> mean, ci_low, ci_high = bootstrap_confidence_interval(waiting_times)
        >>> print(f"Mean: {mean:.2f}, 95% CI: [{ci_low:.2f}, {ci_high:.2f}]")
    """
    values = np.asarray(values)
    n = len(values)

    if n == 0:
        return np.nan, np.nan, np.nan

    # Compute statistic function
    if statistic == 'mean':
        stat_func = np.mean
    elif statistic == 'median':
        stat_func = np.median
    elif statistic == 'std':
        stat_func = np.std
    else:
        raise ValueError(f"Unknown statistic: {statistic}")

    # Point estimate
    point_estimate = stat_func(values)

    # Bootstrap resampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_stats.append(stat_func(sample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Confidence interval using percentile method
    alpha = 1 - confidence
    ci_low = np.percentile(bootstrap_stats, alpha / 2 * 100)
    ci_high = np.percentile(bootstrap_stats, (1 - alpha / 2) * 100)

    return float(point_estimate), float(ci_low), float(ci_high)


def compare_theory_vs_simulation(
    theory_W_q: float,
    sim_W_q: float,
    theory_W: Optional[float] = None,
    sim_W: Optional[float] = None
) -> Dict[str, float]:
    """
    Compute error metrics between theory and simulation.

    Args:
        theory_W_q: Theoretical waiting time
        sim_W_q: Simulated waiting time
        theory_W: Theoretical response time (optional)
        sim_W: Simulated response time (optional)

    Returns:
        dict with:
            - abs_error_W_q: |theory - sim| for waiting time
            - rel_error_W_q: Relative error (%) for waiting time
            - abs_error_W: Absolute error for response time (if provided)
            - rel_error_W: Relative error (%) for response time (if provided)
            - direction: 'over' if theory > sim, 'under' if theory < sim
    """
    # Waiting time errors
    abs_error_W_q = abs(theory_W_q - sim_W_q)

    if theory_W_q > 0 and not np.isinf(theory_W_q):
        rel_error_W_q = (abs_error_W_q / theory_W_q) * 100
    else:
        rel_error_W_q = np.nan

    result = {
        'theory_W_q': theory_W_q,
        'sim_W_q': sim_W_q,
        'abs_error_W_q': abs_error_W_q,
        'rel_error_W_q': rel_error_W_q,
        'direction_W_q': 'over' if theory_W_q > sim_W_q else 'under'
    }

    # Response time errors (if provided)
    if theory_W is not None and sim_W is not None:
        abs_error_W = abs(theory_W - sim_W)

        if theory_W > 0 and not np.isinf(theory_W):
            rel_error_W = (abs_error_W / theory_W) * 100
        else:
            rel_error_W = np.nan

        result.update({
            'theory_W': theory_W,
            'sim_W': sim_W,
            'abs_error_W': abs_error_W,
            'rel_error_W': rel_error_W,
            'direction_W': 'over' if theory_W > sim_W else 'under'
        })

    return result


def analyze_error_by_load(
    results_by_load: Dict[float, Dict[str, float]]
) -> pd.DataFrame:
    """
    Analyze how error varies with system load.

    Args:
        results_by_load: {load: comparison_dict} from compare_theory_vs_simulation

    Returns:
        DataFrame with columns: [load, theory_W_q, sim_W_q, rel_error, direction]
    """
    rows = []
    for load, comparison in results_by_load.items():
        rows.append({
            'load': load,
            'theory_W_q': comparison.get('theory_W_q', np.nan),
            'sim_W_q': comparison.get('sim_W_q', np.nan),
            'abs_error': comparison.get('abs_error_W_q', np.nan),
            'rel_error': comparison.get('rel_error_W_q', np.nan),
            'direction': comparison.get('direction_W_q', '')
        })

    df = pd.DataFrame(rows).sort_values('load')
    return df


def quantify_transient_effect(
    jobs_df: pd.DataFrame,
    waiting_time_col: str = 'waiting_time',
    n_bins: int = 10
) -> pd.DataFrame:
    """
    Quantify the transient effect by analyzing waiting time vs arrival order.

    If transient effects are significant:
    - Early jobs have lower waiting times (system starts empty)
    - Late jobs may have lower waiting times (system draining)

    Args:
        jobs_df: DataFrame with job data
        waiting_time_col: Column name for waiting time
        n_bins: Number of bins to divide jobs into

    Returns:
        DataFrame with columns: [bin, mean_wait, std_wait, count]
        where bin=0 is earliest jobs, bin=n_bins-1 is latest
    """
    df = jobs_df.copy()

    if waiting_time_col not in df.columns:
        if 'start_time' in df.columns and 'arrival_time' in df.columns:
            df['waiting_time'] = df['start_time'] - df['arrival_time']
            waiting_time_col = 'waiting_time'
        else:
            raise ValueError(f"Column {waiting_time_col} not found")

    # Sort by arrival and assign bins
    df = df.sort_values('arrival_time').reset_index(drop=True)
    df['bin'] = pd.cut(df.index, bins=n_bins, labels=False)

    # Aggregate by bin
    agg = df.groupby('bin')[waiting_time_col].agg(['mean', 'std', 'count'])
    agg.columns = ['mean_wait', 'std_wait', 'count']
    agg = agg.reset_index()

    return agg


def recommend_transient_removal(
    jobs_df: pd.DataFrame,
    waiting_time_col: str = 'waiting_time',
    threshold: float = 0.2
) -> Tuple[float, float]:
    """
    Recommend warmup/cooldown fractions based on transient analysis.

    Looks for bins where waiting time is significantly below steady-state.

    Args:
        jobs_df: DataFrame with job data
        waiting_time_col: Column name for waiting time
        threshold: Relative threshold (0.2 = 20% below steady-state is transient)

    Returns:
        (warmup_fraction, cooldown_fraction) recommendations
    """
    transient_df = quantify_transient_effect(jobs_df, waiting_time_col, n_bins=10)

    # Use middle bins as steady-state reference
    middle_bins = transient_df[(transient_df['bin'] >= 3) & (transient_df['bin'] <= 6)]
    steady_state_mean = middle_bins['mean_wait'].mean()

    if steady_state_mean <= 0 or np.isnan(steady_state_mean):
        return 0.1, 0.1  # Default recommendations

    # Find warmup bins (significantly below steady-state)
    warmup_bins = 0
    for _, row in transient_df.iterrows():
        if row['mean_wait'] < steady_state_mean * (1 - threshold):
            warmup_bins += 1
        else:
            break

    # Find cooldown bins
    cooldown_bins = 0
    for _, row in transient_df.iloc[::-1].iterrows():
        if row['mean_wait'] < steady_state_mean * (1 - threshold):
            cooldown_bins += 1
        else:
            break

    warmup_fraction = warmup_bins / 10
    cooldown_fraction = cooldown_bins / 10

    # Ensure reasonable bounds
    warmup_fraction = min(max(warmup_fraction, 0.05), 0.2)
    cooldown_fraction = min(max(cooldown_fraction, 0.05), 0.2)

    return warmup_fraction, cooldown_fraction
