"""
Workload generation module for affect-aware scheduling.

Contains task generation with emotion attributes:
- task_generator: Creates affect-aware jobs with Russell quadrant mapping
"""

from .task_generator import (
    create_emotion_aware_jobs,
    create_jobs_from_trace,
    generate_job_trace,
    generate_job_on_demand,
    get_emotion_aware_statistics,
)

__all__ = [
    'create_emotion_aware_jobs',
    'create_jobs_from_trace',
    'generate_job_trace',
    'generate_job_on_demand',
    'get_emotion_aware_statistics',
]
