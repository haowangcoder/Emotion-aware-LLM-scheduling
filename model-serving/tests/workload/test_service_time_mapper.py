"""
Tests for Service Time Mapping Module

This test file contains tests for the emotion-aware service time mapping functionality.
It also includes helper functions that are only used for testing purposes.
"""

import numpy as np
from typing import Dict
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workload.service_time_mapper import (
    ServiceTimeConfig,
    map_service_time,
    map_service_time_linear
)


# ============================================================================
# Helper Functions (used only in tests)
# ============================================================================

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


# ============================================================================
# Test Functions
# ============================================================================

def test_linear_mapping():
    """Test linear mapping: S_i = L_0 * (1 + α * ρ * a_i)"""
    print("\n" + "=" * 70)
    print("Test 1: Linear Mapping")
    print("=" * 70)

    config = ServiceTimeConfig(
        base_service_time=2.0,
        alpha=0.5
    )
    print(f"Config: {config}")

    test_arousal = [-1.0, -0.5, 0.0, 0.5, 1.0]
    print(f"\n{'Arousal':<10} {'Service Time':<15} {'Change %':<10}")
    print(f"{'-'*35}")
    for a in test_arousal:
        st = map_service_time(a, config)
        change_pct = ((st / config.base_service_time) - 1) * 100
        print(f"{a:<10.2f} {st:<15.3f} {change_pct:>7.1f}%")

        # Assertions
        assert st > 0, f"Service time should be positive, got {st}"
        expected = config.base_service_time * (1 + config.alpha * a)
        assert abs(st - expected) < 0.001, f"Expected {expected}, got {st}"


def test_config_validation():
    """Test configuration safety validation"""
    print("\n" + "=" * 70)
    print("Test 2: Configuration Safety Validation")
    print("=" * 70)

    config = ServiceTimeConfig(
        base_service_time=2.0,
        alpha=0.5
    )

    validation = validate_config_safety(config)
    print(f"Valid: {validation['valid']}")
    print(f"Service times: {validation['service_times']}")

    assert validation['valid'], "Config should be valid"
    assert len(validation['service_times']) == 3, "Should have 3 test points"

    if validation['warnings']:
        print(f"Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")


def test_realistic_arousal_distribution():
    """Test with realistic arousal distribution"""
    print("\n" + "=" * 70)
    print("Test 3: Realistic Arousal Distribution")
    print("=" * 70)

    config = ServiceTimeConfig(
        base_service_time=2.0,
        alpha=0.5
    )

    # Sample arousal from normal distribution (clipped to [-1, 1])
    np.random.seed(42)
    arousal_samples = np.clip(np.random.normal(0, 0.5, 1000), -1, 1)

    stats = get_service_time_statistics(arousal_samples, config)
    print(f"Number of samples: 1000")
    print(f"Service Time Statistics:")
    print(f"  Mean:   {stats['mean']:.3f}")
    print(f"  Std:    {stats['std']:.3f}")
    print(f"  Min:    {stats['min']:.3f}")
    print(f"  Max:    {stats['max']:.3f}")
    print(f"  Median: {stats['median']:.3f}")
    print(f"  P95:    {stats['p95']:.3f}")
    print(f"  P99:    {stats['p99']:.3f}")

    # Assertions
    assert stats['mean'] > 0, "Mean should be positive"
    assert stats['min'] >= config.min_service_time, "Min should be >= min_service_time"
    assert stats['max'] <= 10.0, "Max should be reasonable"


def test_alpha_parameter_sensitivity():
    """Test impact of different alpha values"""
    print("\n" + "=" * 70)
    print("Test 4: Alpha Parameter Sensitivity (arousal = 1.0)")
    print("=" * 70)

    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    print(f"{'Alpha (α)':<12} {'Service Time':<15} {'Change vs α=0':<15}")
    print(f"{'-'*42}")

    baseline_st = None
    for alpha in alpha_values:
        config_alpha = ServiceTimeConfig(base_service_time=2.0, alpha=alpha)
        st = map_service_time(1.0, config_alpha)
        if baseline_st is None:
            baseline_st = st
        change = st - baseline_st
        print(f"{alpha:<12.2f} {st:<15.3f} {change:>+13.3f}")

        # Assertion: higher alpha should give higher service time
        expected = 2.0 * (1 + alpha * 1.0)
        assert abs(st - expected) < 0.001, f"Expected {expected}, got {st}"


def test_rho_parameter_sensitivity():
    """Test impact of rho (correlation strength)"""
    print("\n" + "=" * 70)
    print("Test 5: Rho Parameter Sensitivity (arousal = 1.0, alpha = 0.5)")
    print("=" * 70)

    rho_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    print(f"{'Rho (ρ)':<12} {'Service Time':<15} {'Effective arousal':<20}")
    print(f"{'-'*47}")

    for rho in rho_values:
        config_rho = ServiceTimeConfig(base_service_time=2.0, alpha=0.5, rho=rho)
        st = map_service_time(1.0, config_rho)
        effective = 1.0 * rho
        print(f"{rho:<12.2f} {st:<15.3f} {effective:<20.3f}")

        # Assertion: rho scales the arousal effect
        expected = 2.0 * (1 + 0.5 * effective)
        assert abs(st - expected) < 0.001, f"Expected {expected}, got {st}"


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("SERVICE TIME MAPPER TEST SUITE")
    print("=" * 70)

    # Run all tests
    test_linear_mapping()
    test_config_validation()
    test_realistic_arousal_distribution()
    test_alpha_parameter_sensitivity()
    test_rho_parameter_sensitivity()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
