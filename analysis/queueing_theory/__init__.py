"""
M/G/1 Queueing Theory Analysis Module

This module provides theoretical analysis tools for validating
emotion-aware LLM scheduling against queueing theory predictions.

Key components:
- mg1_formulas: Basic M/G/1 Pollaczek-Khinchin formulas
- multiclass_priority: Kleinrock's multi-class priority queue formulas
- service_time_analysis: Extract service time distribution from data
- load_sweep: Generate theory predictions across load levels
- error_analysis: Finite sample corrections, transient removal
- validation: Compare theory vs simulation
- plotting: Visualization utilities

Usage:
    from analysis.queueing_theory import (
        pollaczek_khinchin,
        kleinrock_multiclass_priority,
        predict_theory_load_sweep,
        validate_fcfs_baseline,
        validate_ssjf_emotion
    )
"""

from .mg1_formulas import (
    pollaczek_khinchin,
    compute_service_time_moments,
    theoretical_service_time_moments,
    pk_sensitivity_analysis,
)

from .multiclass_priority import (
    kleinrock_multiclass_priority,
    weighted_average_response_time,
    compute_priority_improvement,
    ssjf_emotion_priority_order,
    predict_class_waiting_times,
)

from .service_time_analysis import (
    extract_service_time_distribution,
    extract_waiting_times_by_class,
    extract_arrival_rate,
    extract_per_class_arrival_rates,
)

from .error_analysis import (
    remove_transient_period,
    finite_sample_correction,
    bootstrap_confidence_interval,
    compare_theory_vs_simulation,
)

from .load_sweep import (
    predict_theory_load_sweep,
    predict_both_schedulers,
    compute_theoretical_improvement,
)

from .validation import (
    validate_fcfs_baseline,
    validate_ssjf_emotion,
    validate_load_sweep,
    generate_validation_report,
)

__all__ = [
    # mg1_formulas
    'pollaczek_khinchin',
    'compute_service_time_moments',
    'theoretical_service_time_moments',
    'pk_sensitivity_analysis',
    # multiclass_priority
    'kleinrock_multiclass_priority',
    'weighted_average_response_time',
    'compute_priority_improvement',
    'ssjf_emotion_priority_order',
    'predict_class_waiting_times',
    # service_time_analysis
    'extract_service_time_distribution',
    'extract_waiting_times_by_class',
    'extract_arrival_rate',
    'extract_per_class_arrival_rates',
    # error_analysis
    'remove_transient_period',
    'finite_sample_correction',
    'bootstrap_confidence_interval',
    'compare_theory_vs_simulation',
    # load_sweep
    'predict_theory_load_sweep',
    'predict_both_schedulers',
    'compute_theoretical_improvement',
    # validation
    'validate_fcfs_baseline',
    'validate_ssjf_emotion',
    'validate_load_sweep',
    'generate_validation_report',
]
