"""
Service Time Mapping Module for Emotion-aware LLM Scheduling

This module maps emotion arousal values to task service times (execution duration).
The core formula is:
    S_i = L_0 * (1 + α * emotion_correlation * a_i)

Where:
- S_i: Service time for task i
- L_0: Base service time (baseline duration)
- α (alpha): Sensitivity coefficient controlling arousal impact
- emotion_correlation: Correlation strength between emotion and service time [0,1]
- a_i: Arousal value for task i (typically in [-1, 1])

Note: emotion_correlation was previously named 'rho', but renamed to avoid confusion
with system_load (ρ) which represents system utilization in queueing theory.

This implements the "Arousal-Token Length Mapping" from the design specification,
allowing high-arousal emotions to generate longer responses and low-arousal emotions
to generate shorter responses.
"""

import numpy as np


class ServiceTimeConfig:
    """Configuration for service time mapping."""

    def __init__(
        self,
        base_service_time: float = 2.0,
        alpha: float = 0.5,
        emotion_correlation: float = 1.0,
        min_service_time: float = 0.1,
    ):
        """
        Initialize service time configuration.

        Args:
            base_service_time (L_0): Base service time when arousal is 0.
            alpha (α): Linear coefficient for arousal impact.
            emotion_correlation: Correlation strength (0 = no correlation, 1 = full correlation).
                Previously named 'rho' - renamed to avoid confusion with system_load (ρ).
            min_service_time: Minimum allowed service time (safety bound).
        """
        self.base_service_time = base_service_time
        self.alpha = alpha
        self.emotion_correlation = emotion_correlation
        self.min_service_time = min_service_time

        # Validate parameters
        if base_service_time <= 0:
            raise ValueError(f"base_service_time must be positive, got {base_service_time}")
        if min_service_time <= 0:
            raise ValueError(f"min_service_time must be positive, got {min_service_time}")
        if not 0 <= emotion_correlation <= 1:
            raise ValueError(f"emotion_correlation must be in [0, 1], got {emotion_correlation}")

    def __repr__(self):
        return (f"ServiceTimeConfig(L0={self.base_service_time}, α={self.alpha}, "
                f"emotion_correlation={self.emotion_correlation})")


def map_service_time_linear(arousal: float, config: ServiceTimeConfig) -> float:
    """
    Linear mapping: S_i = L_0 * (1 + α * emotion_correlation * a_i)

    This is the primary mapping function specified in the design.

    Args:
        arousal: Arousal value (typically in [-1, 1])
        config: ServiceTimeConfig object

    Returns:
        Service time (guaranteed to be >= min_service_time)
    """
    # Apply correlation strength
    effective_arousal = arousal * config.emotion_correlation

    # Linear formula
    service_time = config.base_service_time * (1 + config.alpha * effective_arousal)

    # Ensure service time is at least min_service_time
    service_time = max(service_time, config.min_service_time)

    return service_time


def map_service_time(arousal: float, config: ServiceTimeConfig = None) -> float:
    """
    Map arousal value to service time using linear mapping:
    S_i = L_0 * (1 + α * emotion_correlation * a_i)

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

    return map_service_time_linear(arousal, config)
