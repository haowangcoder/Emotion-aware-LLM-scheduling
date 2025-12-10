"""
Affect-Aware Task Generator for LLM Scheduling

This module creates jobs/tasks with emotion attributes for the affect-aware
scheduling simulator. It integrates:
1. Emotion sampling with NRC-VAD values and Russell quadrant mapping
2. Service time (uses default or BERT-predicted values)
3. Affect weight computation (Depression-First strategy)
4. Poisson arrival process for task generation

Tasks arrive according to a Poisson process with exponential inter-arrival times.
"""

import numpy as np
from typing import List, Dict, Optional

from core.job import Job
from core.emotion import (
    EmotionConfig,
    sample_emotion,
    sample_emotions_batch,
    sample_emotions_batch_stratified_quadrant,
)
from core.affect_weight import compute_urgency_and_weight


def generate_job_on_demand(
    job_id: int,
    arrival_time: float,
    emotion_config: EmotionConfig = None,
    default_service_time: float = 2.0,
    enable_emotion: bool = True,
    w_max: float = 2.0,
    p: float = 1.0,
    q: float = 1.0,
) -> Job:
    """
    Generate a single job on-demand for fixed-rate arrival simulation.

    Args:
        job_id: Unique job identifier
        arrival_time: When this job arrives in the system
        emotion_config: EmotionConfig object
        default_service_time: Default service time for jobs
        enable_emotion: Whether to enable emotion-aware features
        w_max: Maximum affect weight
        p: Exponent for negative valence
        q: Exponent for low arousal

    Returns:
        Single Job object with emotion and affect weight attributes
    """
    if emotion_config is None:
        emotion_config = EmotionConfig()

    # Sample emotion for this job
    if enable_emotion:
        emotion_label, arousal, valence = sample_emotion(emotion_config)
        russell_quadrant = emotion_config.classify_russell_quadrant(arousal, valence)

        # Compute affect weight
        urgency, affect_weight = compute_urgency_and_weight(
            arousal=arousal,
            valence=valence,
            w_max=w_max,
            p=p,
            q=q,
        )
    else:
        emotion_label = 'neutral'
        arousal = 0.0
        valence = 0.0
        russell_quadrant = 'calm'
        urgency = 0.0
        affect_weight = 1.0

    # Create job with default service time
    # (actual service time will be set during LLM execution or by length predictor)
    job = Job(
        job_id=job_id,
        execution_duration=default_service_time,
        arrival_time=arrival_time,
        emotion_label=emotion_label,
        arousal=arousal,
        valence=valence,
        russell_quadrant=russell_quadrant,
        affect_weight=affect_weight,
        urgency=urgency,
    )

    return job


def create_emotion_aware_jobs(
    num_jobs: int,
    arrival_rate: float = 1.0,
    emotion_config: EmotionConfig = None,
    default_service_time: float = 2.0,
    enable_emotion: bool = True,
    job_configs: List[Dict] = None,
    random_seed: int = None,
    w_max: float = 2.0,
    p: float = 1.0,
    q: float = 1.0,
) -> List[Job]:
    """
    Create a list of jobs with emotion attributes using Poisson arrival process.

    Args:
        num_jobs: Number of jobs to create
        arrival_rate: Arrival rate (lambda) for Poisson process
        emotion_config: EmotionConfig object
        default_service_time: Default service time for jobs
        enable_emotion: Whether to enable emotion-aware features
        job_configs: Optional list of job configurations to load (for reproducibility)
        random_seed: Optional random seed for reproducibility
        w_max: Maximum affect weight
        p: Exponent for negative valence
        q: Exponent for low arousal

    Returns:
        List of Job objects with emotion and affect weight attributes
    """
    if emotion_config is None:
        emotion_config = EmotionConfig()

    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
        import random
        random.seed(random_seed)

    # Load emotions from config or sample new ones
    if job_configs is not None:
        # Use pre-defined job configurations
        emotions_data = []
        for job_config in job_configs[:num_jobs]:
            emotion = job_config.get('emotion', 'neutral')
            arousal = job_config.get('arousal', 0.0)
            valence = job_config.get('valence', 0.0)
            emotions_data.append((emotion, arousal, valence))
    elif enable_emotion:
        # Sample emotions for all jobs
        emotions_data = sample_emotions_batch(num_jobs, emotion_config)
    else:
        # Neutral emotion if disabled
        emotions_data = [('neutral', 0.0, 0.0)] * num_jobs

    # Generate arrival times
    if job_configs is not None:
        arrival_times = [job_configs[i].get('arrival_time', 0.0) for i in range(num_jobs)]
    else:
        arrival_times = _generate_arrival_times(num_jobs, arrival_rate)

    # Create jobs with emotion and affect weight attributes
    job_list = []
    for job_id in range(num_jobs):
        emotion_label, arousal, valence = emotions_data[job_id]

        # Classify Russell quadrant
        russell_quadrant = emotion_config.classify_russell_quadrant(arousal, valence)

        # Get service time from config or use default
        if job_configs is not None and job_id < len(job_configs):
            service_time = job_configs[job_id].get('service_time', default_service_time)
        else:
            service_time = default_service_time

        # Compute affect weight
        if enable_emotion:
            urgency, affect_weight = compute_urgency_and_weight(
                arousal=arousal,
                valence=valence,
                w_max=w_max,
                p=p,
                q=q,
            )
        else:
            urgency = 0.0
            affect_weight = 1.0

        arrival_time = arrival_times[job_id]

        # Create job
        job = Job(
            job_id=job_id,
            execution_duration=service_time,
            arrival_time=arrival_time,
            emotion_label=emotion_label,
            arousal=arousal,
            valence=valence,
            russell_quadrant=russell_quadrant,
            affect_weight=affect_weight,
            urgency=urgency,
        )

        # Set conversation_index if loading from config
        if job_configs is not None and job_id < len(job_configs):
            job.conversation_index = job_configs[job_id].get('conversation_index')

        job_list.append(job)

    return job_list


def _generate_arrival_times(num_jobs: int, arrival_rate: float) -> np.ndarray:
    """
    Generate arrival times using Poisson process.

    Args:
        num_jobs: Number of jobs
        arrival_rate: Arrival rate (lambda) for Poisson process

    Returns:
        Array of arrival times
    """
    intervals = np.random.exponential(1.0 / arrival_rate, num_jobs)
    arrival_times = np.cumsum(intervals)
    return arrival_times


def get_emotion_aware_statistics(job_list: List[Job]) -> Dict:
    """
    Calculate statistics about emotion-aware job generation.

    Args:
        job_list: List of Job objects with emotion attributes

    Returns:
        Dictionary with statistics including Russell quadrant distribution
    """
    arousals = [job.arousal for job in job_list if job.arousal is not None]
    valences = [job.valence for job in job_list if job.valence is not None]
    service_times = [job.execution_duration for job in job_list]
    arrival_times = [job.arrival_time for job in job_list]
    affect_weights = [job.affect_weight for job in job_list if job.affect_weight is not None]

    # Count by Russell quadrant
    quadrant_counts = {
        'excited': 0,
        'calm': 0,
        'panic': 0,
        'depression': 0,
    }
    for job in job_list:
        quadrant = getattr(job, 'russell_quadrant', None)
        if quadrant in quadrant_counts:
            quadrant_counts[quadrant] += 1

    # Calculate arrival intervals
    arrival_intervals = np.diff(arrival_times) if len(arrival_times) > 1 else [0]

    stats = {
        'num_jobs': len(job_list),
        'arousal_mean': np.mean(arousals) if arousals else 0,
        'arousal_std': np.std(arousals) if arousals else 0,
        'valence_mean': np.mean(valences) if valences else 0,
        'valence_std': np.std(valences) if valences else 0,
        'service_time_mean': np.mean(service_times),
        'service_time_std': np.std(service_times),
        'service_time_min': np.min(service_times),
        'service_time_max': np.max(service_times),
        'affect_weight_mean': np.mean(affect_weights) if affect_weights else 1.0,
        'affect_weight_std': np.std(affect_weights) if affect_weights else 0.0,
        'arrival_interval_mean': np.mean(arrival_intervals),
        'arrival_interval_std': np.std(arrival_intervals),
        'quadrant_counts': quadrant_counts,
        'total_duration': arrival_times[-1] if arrival_times else 0,
    }

    return stats


def generate_job_trace(
    num_jobs: int,
    arrival_rate: float,
    emotion_config: EmotionConfig = None,
    default_service_time: float = 2.0,
    enable_emotion: bool = True,
    random_seed: int = None,
    use_stratified_sampling: bool = True,
    class_distribution: Optional[Dict[str, float]] = None,
    w_max: float = 2.0,
    p: float = 1.0,
    q: float = 1.0,
) -> List[Dict]:
    """
    Generate a complete job trace for reproducible experiments across schedulers.

    Uses stratified sampling by Russell quadrant (4 categories) for fair comparison.

    Args:
        num_jobs: Number of jobs to generate
        arrival_rate: Arrival rate (lambda) for Poisson process
        emotion_config: EmotionConfig object
        default_service_time: Default service time for jobs
        enable_emotion: Whether to enable emotion-aware features
        random_seed: Random seed for reproducibility
        use_stratified_sampling: Whether to use stratified sampling by Russell quadrant
        class_distribution: Target distribution for 4 quadrants (default: uniform 1/4 each)
        w_max: Maximum affect weight
        p: Exponent for negative valence
        q: Exponent for low arousal

    Returns:
        List of job configuration dictionaries
    """
    if emotion_config is None:
        emotion_config = EmotionConfig()

    # Set random seed
    if random_seed is not None:
        np.random.seed(random_seed)
        import random
        random.seed(random_seed)

    # Generate arrival times
    arrival_times = _generate_arrival_times(num_jobs, arrival_rate)

    # Generate emotions with Russell quadrant stratification
    if enable_emotion:
        if use_stratified_sampling:
            emotions_data = sample_emotions_batch_stratified_quadrant(
                num_jobs, emotion_config, class_distribution
            )
        else:
            # Simple random sampling
            emotions_data = []
            for emotion, arousal, valence in sample_emotions_batch(num_jobs, emotion_config):
                quadrant = emotion_config.classify_russell_quadrant(arousal, valence)
                emotions_data.append((emotion, arousal, valence, quadrant))
    else:
        emotions_data = [('neutral', 0.0, 0.0, 'calm')] * num_jobs

    # Create trace
    trace = []
    for job_id in range(num_jobs):
        if len(emotions_data[job_id]) == 4:
            emotion_label, arousal, valence, quadrant = emotions_data[job_id]
        else:
            emotion_label, arousal, valence = emotions_data[job_id]
            quadrant = emotion_config.classify_russell_quadrant(arousal, valence)

        # Compute affect weight
        if enable_emotion:
            urgency, affect_weight = compute_urgency_and_weight(
                arousal=arousal,
                valence=valence,
                w_max=w_max,
                p=p,
                q=q,
            )
        else:
            urgency = 0.0
            affect_weight = 1.0

        trace.append({
            'job_id': job_id,
            'arrival_time': float(arrival_times[job_id]),
            'emotion': emotion_label,
            'arousal': float(arousal),
            'valence': float(valence),
            'russell_quadrant': quadrant,
            'service_time': float(default_service_time),
            'affect_weight': float(affect_weight),
            'urgency': float(urgency),
        })

    # Print Russell quadrant distribution
    if enable_emotion:
        quadrant_counts = {'excited': 0, 'calm': 0, 'panic': 0, 'depression': 0}
        for entry in trace:
            quadrant = entry['russell_quadrant']
            if quadrant in quadrant_counts:
                quadrant_counts[quadrant] += 1

        print(f"  Russell quadrant distribution:")
        for quadrant in ['excited', 'calm', 'panic', 'depression']:
            count = quadrant_counts[quadrant]
            pct = count / num_jobs * 100
            print(f"    {quadrant}: {count} ({pct:.1f}%)")

    return trace


def create_jobs_from_trace(
    trace: List[Dict],
    emotion_config: EmotionConfig = None,
    max_jobs: int = None
) -> List[Job]:
    """
    Create Job objects from a pre-generated trace.

    Args:
        trace: List of job configuration dictionaries
        emotion_config: EmotionConfig object
        max_jobs: Maximum number of jobs to create (None = all)

    Returns:
        List of Job objects
    """
    if emotion_config is None:
        emotion_config = EmotionConfig()

    jobs = []
    trace_subset = trace[:max_jobs] if max_jobs else trace

    for job_config in trace_subset:
        arousal = job_config.get('arousal', 0.0)
        valence = job_config.get('valence', 0.0)

        # Get Russell quadrant from trace or compute
        russell_quadrant = job_config.get('russell_quadrant')
        if russell_quadrant is None:
            russell_quadrant = emotion_config.classify_russell_quadrant(arousal, valence)

        # Get affect weight from trace or use default
        affect_weight = job_config.get('affect_weight', 1.0)
        urgency = job_config.get('urgency', 0.0)

        job = Job(
            job_id=job_config['job_id'],
            execution_duration=job_config['service_time'],
            arrival_time=job_config['arrival_time'],
            emotion_label=job_config['emotion'],
            arousal=arousal,
            valence=valence,
            russell_quadrant=russell_quadrant,
            affect_weight=affect_weight,
            urgency=urgency,
        )
        jobs.append(job)

    return jobs
