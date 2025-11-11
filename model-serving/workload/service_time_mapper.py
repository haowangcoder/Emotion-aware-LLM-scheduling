"""
Service Time Mapping Module for Emotion-aware LLM Scheduling

This module maps emotion arousal values to task service times (execution duration).
The core formula is:
    S_i = L_0 * (1 + α * a_i)

Where:
- S_i: Service time for task i
- L_0: Base service time (baseline duration)
- α (alpha): Sensitivity coefficient controlling arousal impact
- a_i: Arousal value for task i (typically in [-1, 1])

This implements the "Arousal-Token Length Mapping" from the design specification,
allowing high-arousal emotions to generate longer responses and low-arousal emotions
to generate shorter responses.
"""

import numpy as np
from typing import Callable, Optional, Dict


class ServiceTimeConfig:
    """Configuration for service time mapping"""

    def __init__(self,
                 base_service_time: float = 1.0,
                 alpha: float = 0.5,
                 gamma: float = 1.0,
                 rho: float = 1.0,
                 min_service_time: float = 0.1,
                 mapping_func: str = 'linear'):
        """
        Initialize service time configuration

        Args:
            base_service_time (L_0): Base service time when arousal is 0
            alpha (α): Linear coefficient for arousal impact
            gamma (γ): Non-linear exponent (1.0 = linear, >1 = super-linear, <1 = sub-linear)
            rho (ρ): Correlation strength (0 = no correlation, 1 = full correlation)
            min_service_time: Minimum allowed service time (safety bound)
            mapping_func: Type of mapping function ('linear', 'exponential', 'gamma_dist')
        """
        self.base_service_time = base_service_time
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.min_service_time = min_service_time
        self.mapping_func = mapping_func

        # Validate parameters
        if base_service_time <= 0:
            raise ValueError(f"base_service_time must be positive, got {base_service_time}")
        if min_service_time <= 0:
            raise ValueError(f"min_service_time must be positive, got {min_service_time}")
        if not 0 <= rho <= 1:
            raise ValueError(f"rho must be in [0, 1], got {rho}")

    def __repr__(self):
        return (f"ServiceTimeConfig(L0={self.base_service_time}, α={self.alpha}, "
                f"γ={self.gamma}, ρ={self.rho}, mapping={self.mapping_func})")


def map_service_time_linear(arousal: float, config: ServiceTimeConfig) -> float:
    """
    Linear mapping: S_i = L_0 * (1 + α * a_i)

    This is the primary mapping function specified in the design.

    Args:
        arousal: Arousal value (typically in [-1, 1])
        config: ServiceTimeConfig object

    Returns:
        Service time (guaranteed to be >= min_service_time)
    """
    # Apply correlation strength (rho)
    effective_arousal = arousal * config.rho

    # Linear formula
    service_time = config.base_service_time * (1 + config.alpha * effective_arousal)

    # Apply non-linear transformation if gamma != 1
    if config.gamma != 1.0:
        # Preserve sign of the arousal effect
        arousal_effect = config.alpha * effective_arousal
        if arousal_effect >= 0:
            arousal_effect = arousal_effect ** config.gamma
        else:
            arousal_effect = -(abs(arousal_effect) ** config.gamma)
        service_time = config.base_service_time * (1 + arousal_effect)

    # Ensure service time is at least min_service_time
    service_time = max(service_time, config.min_service_time)

    return service_time


def map_service_time_exponential(arousal: float, config: ServiceTimeConfig) -> float:
    """
    Exponential mapping: S_i = L_0 * exp(α * a_i)

    Provides stronger scaling for high arousal values.

    Args:
        arousal: Arousal value
        config: ServiceTimeConfig object

    Returns:
        Service time
    """
    effective_arousal = arousal * config.rho
    service_time = config.base_service_time * np.exp(config.alpha * effective_arousal)
    service_time = max(service_time, config.min_service_time)
    return service_time


def map_service_time_gamma_dist(arousal: float, config: ServiceTimeConfig) -> float:
    """
    Gamma distribution-based mapping: sample from Gamma(shape, scale)
    where parameters depend on arousal

    Args:
        arousal: Arousal value
        config: ServiceTimeConfig object

    Returns:
        Service time sampled from Gamma distribution
    """
    effective_arousal = arousal * config.rho

    # Map arousal to Gamma distribution parameters
    # Higher arousal -> higher mean (scale parameter)
    # gamma parameter controls shape
    mean_multiplier = 1 + config.alpha * effective_arousal
    mean = config.base_service_time * mean_multiplier

    # Shape parameter (controls variance)
    shape = config.gamma
    scale = mean / shape

    # Sample from Gamma distribution
    service_time = np.random.gamma(shape, scale)
    service_time = max(service_time, config.min_service_time)

    return service_time


# Mapping function registry
MAPPING_FUNCTIONS: Dict[str, Callable] = {
    'linear': map_service_time_linear,
    'exponential': map_service_time_exponential,
    'gamma_dist': map_service_time_gamma_dist,
}


def map_service_time(arousal: float, config: ServiceTimeConfig = None) -> float:
    """
    Map arousal value to service time using configured mapping function

    Args:
        arousal: Arousal value (typically in [-1, 1])
        config: ServiceTimeConfig object (uses default if None)

    Returns:
        Service time (execution duration)

    Example:
        >>> config = ServiceTimeConfig(base_service_time=2.0, alpha=0.5)
        >>> service_time = map_service_time(arousal=0.8, config=config)
        >>> print(f"Service time: {service_time:.2f}")
        Service time: 2.80
    """
    if config is None:
        config = ServiceTimeConfig()

    mapping_func = MAPPING_FUNCTIONS.get(config.mapping_func)
    if mapping_func is None:
        raise ValueError(f"Unknown mapping function: {config.mapping_func}")

    return mapping_func(arousal, config)


def map_service_times_batch(arousal_values: list, config: ServiceTimeConfig = None) -> list:
    """
    Map multiple arousal values to service times

    Args:
        arousal_values: List of arousal values
        config: ServiceTimeConfig object (uses default if None)

    Returns:
        List of service times
    """
    return [map_service_time(a, config) for a in arousal_values]


def get_service_time_statistics(arousal_values: list,
                                 config: ServiceTimeConfig = None) -> Dict:
    """
    Calculate statistics about service time distribution given arousal values

    Args:
        arousal_values: List of arousal values
        config: ServiceTimeConfig object

    Returns:
        Dictionary with statistics
    """
    service_times = map_service_times_batch(arousal_values, config)

    return {
        'mean': np.mean(service_times),
        'std': np.std(service_times),
        'min': np.min(service_times),
        'max': np.max(service_times),
        'median': np.median(service_times),
        'p25': np.percentile(service_times, 25),
        'p75': np.percentile(service_times, 75),
        'p95': np.percentile(service_times, 95),
        'p99': np.percentile(service_times, 99),
    }


def validate_config_safety(config: ServiceTimeConfig, arousal_min: float = -1.0,
                           arousal_max: float = 1.0) -> Dict:
    """
    Validate that the configuration produces valid service times across arousal range

    Args:
        config: ServiceTimeConfig object to validate
        arousal_min: Minimum arousal value to test
        arousal_max: Maximum arousal value to test

    Returns:
        Dictionary with validation results
    """
    # Test boundary values
    test_arousal_values = [arousal_min, 0.0, arousal_max]

    results = {
        'valid': True,
        'warnings': [],
        'service_times': {},
    }

    for arousal in test_arousal_values:
        try:
            service_time = map_service_time(arousal, config)
            results['service_times'][arousal] = service_time

            if service_time <= 0:
                results['valid'] = False
                results['warnings'].append(
                    f"Service time <= 0 for arousal={arousal}: {service_time}")

            if service_time < config.min_service_time:
                results['warnings'].append(
                    f"Service time below minimum for arousal={arousal}: {service_time}")

        except Exception as e:
            results['valid'] = False
            results['warnings'].append(f"Error for arousal={arousal}: {str(e)}")

    # Check if range is reasonable (max should be > min)
    if len(results['service_times']) >= 2:
        st_values = list(results['service_times'].values())
        if max(st_values) <= min(st_values):
            results['warnings'].append(
                "Service time range is too narrow - arousal has minimal impact")

    return results


# Example usage and testing
if __name__ == '__main__':
    print("=" * 70)
    print("Service Time Mapping Module Test")
    print("=" * 70)

    # Test linear mapping
    print("\n1. Linear Mapping Test (S_i = L_0 * (1 + α * a_i))")
    config_linear = ServiceTimeConfig(
        base_service_time=2.0,
        alpha=0.5,
        mapping_func='linear'
    )
    print(f"   Config: {config_linear}")

    test_arousal = [-1.0, -0.5, 0.0, 0.5, 1.0]
    print(f"\n   {'Arousal':<10} {'Service Time':<15} {'Change %':<10}")
    print(f"   {'-'*35}")
    for a in test_arousal:
        st = map_service_time(a, config_linear)
        change_pct = ((st / config_linear.base_service_time) - 1) * 100
        print(f"   {a:<10.2f} {st:<15.3f} {change_pct:>7.1f}%")

    # Test exponential mapping
    print("\n2. Exponential Mapping Test (S_i = L_0 * exp(α * a_i))")
    config_exp = ServiceTimeConfig(
        base_service_time=2.0,
        alpha=0.3,
        mapping_func='exponential'
    )
    print(f"   Config: {config_exp}")
    print(f"\n   {'Arousal':<10} {'Service Time':<15} {'Change %':<10}")
    print(f"   {'-'*35}")
    for a in test_arousal:
        st = map_service_time(a, config_exp)
        change_pct = ((st / config_exp.base_service_time) - 1) * 100
        print(f"   {a:<10.2f} {st:<15.3f} {change_pct:>7.1f}%")

    # Validate configuration safety
    print("\n3. Configuration Safety Validation")
    validation = validate_config_safety(config_linear)
    print(f"   Valid: {validation['valid']}")
    print(f"   Service times: {validation['service_times']}")
    if validation['warnings']:
        print(f"   Warnings:")
        for warning in validation['warnings']:
            print(f"     - {warning}")

    # Test with realistic arousal distribution
    print("\n4. Realistic Arousal Distribution Test")
    # Sample arousal from normal distribution (clipped to [-1, 1])
    np.random.seed(42)
    arousal_samples = np.clip(np.random.normal(0, 0.5, 1000), -1, 1)

    stats = get_service_time_statistics(arousal_samples, config_linear)
    print(f"   Number of samples: 1000")
    print(f"   Service Time Statistics:")
    print(f"     Mean:   {stats['mean']:.3f}")
    print(f"     Std:    {stats['std']:.3f}")
    print(f"     Min:    {stats['min']:.3f}")
    print(f"     Max:    {stats['max']:.3f}")
    print(f"     Median: {stats['median']:.3f}")
    print(f"     P95:    {stats['p95']:.3f}")
    print(f"     P99:    {stats['p99']:.3f}")

    # Test impact of different alpha values
    print("\n5. Alpha Parameter Sensitivity (arousal = 1.0)")
    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    print(f"   {'Alpha (α)':<12} {'Service Time':<15} {'Change vs α=0':<15}")
    print(f"   {'-'*42}")
    baseline_st = None
    for alpha in alpha_values:
        config_alpha = ServiceTimeConfig(base_service_time=2.0, alpha=alpha)
        st = map_service_time(1.0, config_alpha)
        if baseline_st is None:
            baseline_st = st
        change = st - baseline_st
        print(f"   {alpha:<12.2f} {st:<15.3f} {change:>+13.3f}")

    print("\n" + "=" * 70)
