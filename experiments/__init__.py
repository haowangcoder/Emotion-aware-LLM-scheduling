"""
Experiment module for Emotion-aware LLM Scheduling.

This module contains experiment scripts for:
- Exp-0: Latency decomposition verification
- Exp-1: k (weight_exponent) sweep
- Exp-2: Load sweep
- Exp-3: gamma_panic sweep
- Exp-4: Online control with dynamic k
- Exp-5: Queueing modeling
- Exp-6: Multi-seed statistical validation
- A1-A3: Defense experiments (starvation prevention, etc.)
"""

from .utils import result_aggregator, ci_calculator

__all__ = [
    "result_aggregator",
    "ci_calculator",
]
