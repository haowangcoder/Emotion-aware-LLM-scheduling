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


# Example usage and testing
if __name__ == '__main__':
    print("=" * 70)
    print("Emotion-aware Task Generator Test")
    print("=" * 70)

    # Test 1: Basic emotion-aware job creation
    print("\n1. Basic Emotion-aware Job Creation (n=10, Poisson)")
    emotion_config = EmotionConfig()
    service_config = ServiceTimeConfig(base_service_time=2.0, alpha=0.5)

    jobs = create_emotion_aware_jobs(
        num_jobs=10,
        arrival_rate=2.0,
        emotion_config=emotion_config,
        service_time_config=service_config
    )

    print(f"\n   {'Job ID':<8} {'Emotion':<15} {'Arousal':<10} {'Class':<10} {'Service':<10} {'Arrival':<10}")
    print(f"   {'-'*70}")
    for job in jobs[:10]:
        print(f"   {job.job_id:<8} {job.emotion_label:<15} {job.arousal:<10.2f} "
              f"{job.emotion_class:<10} {job.execution_duration:<10.2f} {job.arrival_time:<10.2f}")

    # Test 2: Statistics
    print("\n2. Job Generation Statistics")
    stats = get_emotion_aware_statistics(jobs)
    print(f"   Total jobs: {stats['num_jobs']}")
    print(f"   Arousal mean: {stats['arousal_mean']:.3f}")
    print(f"   Service time mean: {stats['service_time_mean']:.3f} ± {stats['service_time_std']:.3f}")
    print(f"   Service time range: [{stats['service_time_min']:.3f}, {stats['service_time_max']:.3f}]")
    print(f"   Arrival interval mean: {stats['arrival_interval_mean']:.3f}")
    print(f"   Emotion class distribution:")
    for cls, count in stats['emotion_class_counts'].items():
        print(f"     {cls}: {count} ({count/stats['num_jobs']*100:.1f}%)")

    # Test 3: Compare with/without emotion awareness
    print("\n3. Comparison: With vs Without Emotion Awareness")
    jobs_with_emotion = create_emotion_aware_jobs(
        num_jobs=100,
        arrival_rate=2.0,
        enable_emotion=True
    )
    jobs_without_emotion = create_emotion_aware_jobs(
        num_jobs=100,
        arrival_rate=2.0,
        enable_emotion=False
    )

    stats_with = get_emotion_aware_statistics(jobs_with_emotion)
    stats_without = get_emotion_aware_statistics(jobs_without_emotion)

    print(f"\n   {'Metric':<30} {'With Emotion':<15} {'Without Emotion':<15}")
    print(f"   {'-'*60}")
    print(f"   {'Service time mean':<30} {stats_with['service_time_mean']:<15.3f} "
          f"{stats_without['service_time_mean']:<15.3f}")
    print(f"   {'Service time std':<30} {stats_with['service_time_std']:<15.3f} "
          f"{stats_without['service_time_std']:<15.3f}")
    print(f"   {'Service time range':<30} "
          f"{stats_with['service_time_max']-stats_with['service_time_min']:<15.3f} "
          f"{stats_without['service_time_max']-stats_without['service_time_min']:<15.3f}")

    print("\n" + "=" * 70)
