"""
Workload generation module for emotion-aware scheduling.

Contains task generation and service time mapping:
- service_time_mapper: Maps emotion arousal to service times
- task_generator: Creates emotion-aware jobs
"""

from .service_time_mapper import (
    ServiceTimeConfig,
    map_service_time_linear,
    map_service_time_exponential,
    map_service_time_gamma_dist,
    map_service_time
)
from .task_generator import (
    EmotionAwareTaskConfig,
    create_emotion_aware_jobs,
    get_emotion_aware_statistics
)

__all__ = [
    'ServiceTimeConfig',
    'map_service_time_linear',
    'map_service_time_exponential',
    'map_service_time_gamma_dist',
    'map_service_time',
    'EmotionAwareTaskConfig',
    'create_emotion_aware_jobs',
    'get_emotion_aware_statistics',
]
