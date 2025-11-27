"""
Core module for emotion-aware LLM scheduling.

Contains domain models and scheduling algorithms:
- job: Job class with emotion attributes
- emotion: Emotion sampling and arousal mapping
- scheduler_base: Base scheduler and FCFS/SJF implementations
- ssjf_emotion: Emotion-aware SSJF schedulers
"""

from .job import Job
from .emotion import (
    EmotionConfig,
    EMOTION_AROUSAL_MAP,
    EMOTION_VALENCE_MAP,
    EMOTION_CATEGORY_MAP,
    EMOTION_CATEGORIES,
    sample_emotion,
    sample_emotions_batch,
    sample_emotions_batch_stratified_9class,
    get_emotion_statistics
)
from .scheduler_base import SchedulerBase, FCFSScheduler, SJFScheduler
from .ssjf_emotion import SSJFEmotionScheduler, SSJFEmotionPriorityScheduler, SSJFValenceScheduler

__all__ = [
    'Job',
    'EmotionConfig',
    'EMOTION_AROUSAL_MAP',
    'EMOTION_VALENCE_MAP',
    'EMOTION_CATEGORY_MAP',
    'EMOTION_CATEGORIES',
    'sample_emotion',
    'sample_emotions_batch',
    'sample_emotions_batch_stratified_9class',
    'get_emotion_statistics',
    'SchedulerBase',
    'FCFSScheduler',
    'SJFScheduler',
    'SSJFEmotionScheduler',
    'SSJFEmotionPriorityScheduler',
    'SSJFValenceScheduler',
]
