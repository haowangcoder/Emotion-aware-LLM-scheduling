"""
Core module for Affect-Aware LLM Scheduling (AW-SSJF Framework).

Contains domain models and scheduling algorithms:
- job: Job class with emotion attributes and affect weight
- emotion: Emotion sampling with NRC-VAD values and Russell quadrants
- affect_weight: Depression-First weight computation
- scheduler_base: Base scheduler and FCFS/SJF implementations
- aw_ssjf_scheduler: Affect-Weighted SSJF scheduler
- weight_only_scheduler: Pure affect-based scheduler (baseline)
"""

from .job import Job
from .emotion import (
    EmotionConfig,
    EMOTION_AROUSAL_MAP,
    EMOTION_VALENCE_MAP,
    EMOTION_DOMINANCE_MAP,
    RUSSELL_QUADRANT_MAP,
    RUSSELL_QUADRANTS,
    sample_emotion,
    sample_emotions_batch,
    sample_emotions_batch_stratified_quadrant,
    get_emotion_statistics,
)
from .affect_weight import (
    compute_urgency,
    affect_weight,
    compute_wspt_score,
    compute_urgency_and_weight,
)
from .scheduler_base import SchedulerBase, FCFSScheduler, SJFScheduler
from .aw_ssjf_scheduler import AWSSJFScheduler
from .weight_only_scheduler import WeightOnlyScheduler

# Backward compatibility imports (deprecated)
from .ssjf_emotion import (
    SSJFEmotionScheduler,
    SSJFEmotionPriorityScheduler,
    SSJFValenceScheduler,
    SSJFCombinedScheduler,
)

__all__ = [
    # Job
    'Job',

    # Emotion
    'EmotionConfig',
    'EMOTION_AROUSAL_MAP',
    'EMOTION_VALENCE_MAP',
    'EMOTION_DOMINANCE_MAP',
    'RUSSELL_QUADRANT_MAP',
    'RUSSELL_QUADRANTS',
    'sample_emotion',
    'sample_emotions_batch',
    'sample_emotions_batch_stratified_quadrant',
    'get_emotion_statistics',

    # Affect Weight
    'compute_urgency',
    'affect_weight',
    'compute_wspt_score',
    'compute_urgency_and_weight',

    # Schedulers
    'SchedulerBase',
    'FCFSScheduler',
    'SJFScheduler',
    'AWSSJFScheduler',
    'WeightOnlyScheduler',

    # Deprecated (backward compatibility)
    'SSJFEmotionScheduler',
    'SSJFEmotionPriorityScheduler',
    'SSJFValenceScheduler',
    'SSJFCombinedScheduler',
]
