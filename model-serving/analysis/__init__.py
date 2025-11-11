"""
Analysis module for emotion-aware scheduling.

Contains logging and fairness metrics:
- logger: Logs experiment results to CSV/JSON
- fairness_metrics: Fairness analysis tools
"""

from .logger import EmotionAwareLogger
from .fairness_metrics import (
    calculate_jain_fairness_index,
    calculate_per_class_metrics,
    calculate_fairness_across_emotions,
    analyze_fairness_comprehensive
)

__all__ = [
    'EmotionAwareLogger',
    'calculate_jain_fairness_index',
    'calculate_per_class_metrics',
    'calculate_fairness_across_emotions',
    'analyze_fairness_comprehensive',
]
