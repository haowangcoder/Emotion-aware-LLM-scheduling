"""
Emotion-aware Task Generator for LLM Scheduling

This module creates jobs/tasks with emotion attributes for the emotion-aware
scheduling simulator. It integrates:
1. Emotion sampling from emotion.py
2. Service time mapping from service_time_mapper.py
3. Poisson arrival process for task generation

Tasks arrive according to a Poisson process with exponential inter-arrival times.
"""

import numpy as np
from typing import List, Dict

from core.job import Job
from core.emotion import EmotionConfig, sample_emotion, sample_emotions_batch
from workload.service_time_mapper import ServiceTimeConfig, map_service_time


def generate_job_on_demand(
        job_id: int,
        arrival_time: float,
        emotion_config: EmotionConfig = None,
        service_time_config: ServiceTimeConfig = None,
        enable_emotion: bool = True) -> Job:
    """
    Generate a single job on-demand for fixed-rate arrival simulation

    Args:
        job_id: Unique job identifier
        arrival_time: When this job arrives in the system
        emotion_config: EmotionConfig object
        service_time_config: ServiceTimeConfig object
        enable_emotion: Whether to enable emotion-aware features

    Returns:
        Single Job object with emotion attributes
    """
    if emotion_config is None:
        emotion_config = EmotionConfig()
    if service_time_config is None:
        service_time_config = ServiceTimeConfig()

    # Sample emotion for this job
    if enable_emotion:
        emotion_label, arousal = sample_emotion(emotion_config)
        emotion_class = emotion_config.classify_arousal(arousal)
        service_time = map_service_time(arousal, service_time_config)
    else:
        emotion_label = 'neutral'
        arousal = 0.0
        emotion_class = 'medium'
        service_time = service_time_config.base_service_time

    # Create job
    job = Job(
        job_id=job_id,
        execution_duration=service_time,
        arrival_time=arrival_time,
        emotion_label=emotion_label,
        arousal=arousal,
        emotion_class=emotion_class
    )

    return job


def create_emotion_aware_jobs(
        num_jobs: int,
        arrival_rate: float = 1.0,
        emotion_config: EmotionConfig = None,
        service_time_config: ServiceTimeConfig = None,
        enable_emotion: bool = True,
        job_configs: List[Dict] = None,
        random_seed: int = None) -> List[Job]:
    """
    Create a list of jobs with emotion attributes using Poisson arrival process

    Args:
        num_jobs: Number of jobs to create
        arrival_rate: Arrival rate (λ) for Poisson process
        emotion_config: EmotionConfig object
        service_time_config: ServiceTimeConfig object
        enable_emotion: Whether to enable emotion-aware features
        job_configs: Optional list of job configurations to load (for reproducibility)
        random_seed: Optional random seed for reproducibility

    Returns:
        List of Job objects with emotion attributes
    """
    if emotion_config is None:
        emotion_config = EmotionConfig()
    if service_time_config is None:
        service_time_config = ServiceTimeConfig()

    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
        import random
        random.seed(random_seed)

    # Load emotions from config or sample new ones
    if job_configs is not None:
        # Use pre-defined job configurations
        emotions_arousal = []
        for job_config in job_configs[:num_jobs]:
            emotion = job_config.get('emotion', 'neutral')
            # Use saved arousal value if available, otherwise recalculate from emotion
            arousal = job_config.get('arousal')
            if arousal is None:
                # Fallback: recalculate arousal from emotion (for old config files)
                if enable_emotion and emotion in emotion_config.emotion_arousal_map:
                    arousal = emotion_config.get_arousal(emotion, add_noise=False)
                else:
                    arousal = 0.0
            emotions_arousal.append((emotion, arousal))
    elif enable_emotion:
        # Sample emotions for all jobs
        emotions_arousal = sample_emotions_batch(num_jobs, emotion_config)
    else:
        # Neutral emotion if disabled
        emotions_arousal = [('neutral', 0.0)] * num_jobs

    # Generate arrival times: use saved values if loading from config, otherwise generate new
    if job_configs is not None:
        # Load arrival times from saved job configurations
        arrival_times = [job_configs[i].get('arrival_time', 0.0) for i in range(num_jobs)]
    else:
        # Generate new arrival times using Poisson process
        arrival_times = _generate_arrival_times(
            num_jobs=num_jobs,
            arrival_rate=arrival_rate
        )

    # Create jobs with emotion attributes
    job_list = []
    for job_id in range(num_jobs):
        emotion_label, arousal = emotions_arousal[job_id]
        emotion_class = emotion_config.classify_arousal(arousal)

        # Get service time: use saved value if loading from config, otherwise calculate
        if job_configs is not None and job_id < len(job_configs):
            # Use exact saved service_time to ensure identical scheduling decisions
            service_time = job_configs[job_id].get('service_time',
                                                    map_service_time(arousal, service_time_config))
        elif enable_emotion:
            # Calculate service time from arousal
            service_time = map_service_time(arousal, service_time_config)
        else:
            # Use default service time if emotion disabled
            service_time = service_time_config.base_service_time

        arrival_time = arrival_times[job_id]

        # Create job with emotion attributes
        job = Job(
            job_id=job_id,
            execution_duration=service_time,
            arrival_time=arrival_time,
            emotion_label=emotion_label,
            arousal=arousal,
            emotion_class=emotion_class
        )

        # Set conversation_index if loading from config
        if job_configs is not None and job_id < len(job_configs):
            job.conversation_index = job_configs[job_id].get('conversation_index')

        job_list.append(job)

    return job_list


def _generate_arrival_times(
        num_jobs: int,
        arrival_rate: float) -> np.ndarray:
    """
    Generate arrival times using Poisson process

    Args:
        num_jobs: Number of jobs
        arrival_rate: Arrival rate (λ) for Poisson process

    Returns:
        Array of arrival times
    """
    # Poisson process: arrival intervals follow exponential distribution
    # Inter-arrival time ~ Exponential(λ) where λ is the arrival_rate
    intervals = np.random.exponential(1.0 / arrival_rate, num_jobs)
    arrival_times = np.cumsum(intervals)

    return arrival_times


def get_emotion_aware_statistics(job_list: List[Job]) -> Dict:
    """
    Calculate statistics about emotion-aware job generation

    Args:
        job_list: List of Job objects with emotion attributes

    Returns:
        Dictionary with statistics
    """
    arousals = [job.arousal for job in job_list if job.arousal is not None]
    service_times = [job.execution_duration for job in job_list]
    arrival_times = [job.arrival_time for job in job_list]

    # Count by emotion class
    emotion_classes = [job.emotion_class for job in job_list if job.emotion_class is not None]
    class_counts = {
        'high': emotion_classes.count('high'),
        'medium': emotion_classes.count('medium'),
        'low': emotion_classes.count('low')
    }

    # Calculate arrival intervals
    arrival_intervals = np.diff(arrival_times) if len(arrival_times) > 1 else [0]

    stats = {
        'num_jobs': len(job_list),
        'arousal_mean': np.mean(arousals) if arousals else 0,
        'arousal_std': np.std(arousals) if arousals else 0,
        'service_time_mean': np.mean(service_times),
        'service_time_std': np.std(service_times),
        'service_time_min': np.min(service_times),
        'service_time_max': np.max(service_times),
        'arrival_interval_mean': np.mean(arrival_intervals),
        'arrival_interval_std': np.std(arrival_intervals),
        'emotion_class_counts': class_counts,
        'total_duration': arrival_times[-1] if arrival_times else 0,
    }

    return stats
