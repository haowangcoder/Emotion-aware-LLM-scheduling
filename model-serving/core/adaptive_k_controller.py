"""
Adaptive K Controller for Online Control (Exp-4)

Implements dynamic adjustment of weight_exponent (k) based on queue length.
Uses a threshold-based policy with hysteresis to prevent oscillation.

Control Strategy:
- High load (queue_len > high_threshold): Increase k to protect vulnerable users
- Low load (queue_len < low_threshold): Decrease k for efficiency
- Hysteresis: Require consecutive readings before switching

Usage:
    controller = AdaptiveKController(AdaptiveKConfig(k_min=1, k_max=4))
    new_k = controller.get_k(queue_length=15, current_time=10.5)
    scheduler.weight_exponent = new_k
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import json


@dataclass
class AdaptiveKConfig:
    """Configuration for adaptive k controller."""
    k_min: float = 1.0
    k_max: float = 4.0
    high_threshold: int = 10  # Queue length threshold to increase k
    low_threshold: int = 3    # Queue length threshold to decrease k
    hysteresis_window: int = 3  # Consecutive readings before switching
    adjustment_interval: float = 5.0  # Seconds between adjustment checks
    initial_k: float = 2.0  # Starting k value


class AdaptiveKController:
    """
    Online controller that adapts weight_exponent (k) based on queue length.

    This implements a threshold-based policy with hysteresis to prevent oscillation:
    - When queue_length consistently exceeds high_threshold, increase k
    - When queue_length consistently stays below low_threshold, decrease k
    - Hysteresis prevents rapid oscillation by requiring consecutive readings

    Attributes:
        config: Controller configuration
        current_k: Current k value being used
        k_history: List of (time, k) tuples for visualization
        load_history: List of (time, queue_len) tuples for visualization
    """

    def __init__(self, config: Optional[AdaptiveKConfig] = None):
        """
        Initialize the adaptive k controller.

        Args:
            config: Controller configuration. Uses defaults if None.
        """
        self.config = config or AdaptiveKConfig()
        self.current_k = self.config.initial_k

        # History tracking for visualization
        self.k_history: List[Tuple[float, float]] = [(0.0, self.current_k)]
        self.load_history: List[Tuple[float, int]] = []

        # Hysteresis state
        self.high_count = 0  # Consecutive high load readings
        self.low_count = 0   # Consecutive low load readings
        self.last_adjustment_time = 0.0

    def get_k(self, queue_length: int, current_time: float) -> float:
        """
        Determine the appropriate k value based on current queue length.

        Args:
            queue_length: Current number of jobs in queue
            current_time: Current simulation time

        Returns:
            The k value to use for scheduling
        """
        # Always record load for visualization
        self.load_history.append((current_time, queue_length))

        # Check if adjustment interval has passed
        if current_time - self.last_adjustment_time < self.config.adjustment_interval:
            return self.current_k

        self.last_adjustment_time = current_time

        # Update hysteresis counters based on current load
        if queue_length > self.config.high_threshold:
            self.high_count += 1
            self.low_count = 0
        elif queue_length < self.config.low_threshold:
            self.low_count += 1
            self.high_count = 0
        else:
            # In middle zone: decay both counters
            self.high_count = max(0, self.high_count - 1)
            self.low_count = max(0, self.low_count - 1)

        # Make adjustment decision based on hysteresis window
        new_k = self.current_k

        if self.high_count >= self.config.hysteresis_window:
            # High load sustained: increase k to protect vulnerable users
            if self.current_k < self.config.k_max:
                new_k = min(self.current_k + 1.0, self.config.k_max)
                self.high_count = 0  # Reset after adjustment

        elif self.low_count >= self.config.hysteresis_window:
            # Low load sustained: decrease k for efficiency
            if self.current_k > self.config.k_min:
                new_k = max(self.current_k - 1.0, self.config.k_min)
                self.low_count = 0  # Reset after adjustment

        # Record k change if it happened
        if new_k != self.current_k:
            self.current_k = new_k
            self.k_history.append((current_time, new_k))

        return self.current_k

    def get_trajectory(self) -> dict:
        """
        Get recorded trajectory data for plotting.

        Returns:
            Dictionary with k_history and load_history
        """
        return {
            'k_history': self.k_history,
            'load_history': self.load_history,
        }

    def get_statistics(self) -> dict:
        """
        Get summary statistics of the controller behavior.

        Returns:
            Dictionary with statistics about k adjustments
        """
        k_values = [k for _, k in self.k_history]
        load_values = [load for _, load in self.load_history]

        return {
            'num_k_transitions': len(self.k_history) - 1,  # Exclude initial value
            'k_values_used': sorted(set(k_values)),
            'final_k': self.current_k,
            'avg_queue_length': sum(load_values) / len(load_values) if load_values else 0,
            'max_queue_length': max(load_values) if load_values else 0,
            'min_queue_length': min(load_values) if load_values else 0,
        }

    def save_trajectory(self, filepath: str) -> None:
        """
        Save trajectory data to JSON file.

        Args:
            filepath: Path to save the trajectory
        """
        data = {
            'config': {
                'k_min': self.config.k_min,
                'k_max': self.config.k_max,
                'high_threshold': self.config.high_threshold,
                'low_threshold': self.config.low_threshold,
                'hysteresis_window': self.config.hysteresis_window,
                'adjustment_interval': self.config.adjustment_interval,
            },
            'trajectory': self.get_trajectory(),
            'statistics': self.get_statistics(),
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def reset(self) -> None:
        """Reset the controller to initial state."""
        self.current_k = self.config.initial_k
        self.k_history = [(0.0, self.current_k)]
        self.load_history = []
        self.high_count = 0
        self.low_count = 0
        self.last_adjustment_time = 0.0


def create_controller_from_config(scheduler_config) -> Optional[AdaptiveKController]:
    """
    Create an AdaptiveKController from scheduler configuration.

    Args:
        scheduler_config: SchedulerConfig object with adaptive_k settings

    Returns:
        AdaptiveKController if adaptive_k is enabled, None otherwise
    """
    if not getattr(scheduler_config, 'adaptive_k', False):
        return None

    k_min = getattr(scheduler_config, 'adaptive_k_min', 1.0)
    k_max = getattr(scheduler_config, 'adaptive_k_max', 4.0)
    start_k = getattr(scheduler_config, 'weight_exponent', None)
    if start_k is None:
        start_k = k_min
    start_k = max(k_min, min(float(start_k), k_max))

    config = AdaptiveKConfig(
        k_min=k_min,
        k_max=k_max,
        high_threshold=getattr(scheduler_config, 'adaptive_k_high_threshold', 10),
        low_threshold=getattr(scheduler_config, 'adaptive_k_low_threshold', 3),
        initial_k=start_k,
    )

    return AdaptiveKController(config)
