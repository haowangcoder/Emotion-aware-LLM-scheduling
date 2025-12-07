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
        valence = emotion_config.get_valence(emotion_label)
        valence_class = emotion_config.classify_valence(valence)
        service_time = map_service_time(arousal, service_time_config)
    else:
        emotion_label = 'neutral'
        arousal = 0.0
        emotion_class = 'medium'
        valence = 0.0
        valence_class = 'neutral'
        service_time = service_time_config.base_service_time

    # Create job
    job = Job(
        job_id=job_id,
        execution_duration=service_time,
        arrival_time=arrival_time,
        emotion_label=emotion_label,
        arousal=arousal,
        emotion_class=emotion_class,
        valence=valence,
        valence_class=valence_class,
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
            # Use saved valence if available; otherwise compute from emotion label
            valence = job_config.get('valence')
            if valence is None:
                if enable_emotion and emotion in emotion_config.emotion_valence_map:
                    valence = emotion_config.get_valence(emotion)
                else:
                    valence = 0.0
            emotions_arousal.append((emotion, arousal, valence))
    elif enable_emotion:
        # Sample emotions for all jobs
        emotions_arousal = []
        for emotion, arousal in sample_emotions_batch(num_jobs, emotion_config):
            valence = emotion_config.get_valence(emotion)
            emotions_arousal.append((emotion, arousal, valence))
    else:
        # Neutral emotion if disabled
        emotions_arousal = [('neutral', 0.0, 0.0)] * num_jobs

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
        emotion_label, arousal, valence = emotions_arousal[job_id]
        emotion_class = emotion_config.classify_arousal(arousal)
        valence_class = None

        # Get service time: use saved value if loading from config, otherwise calculate
        if job_configs is not None and job_id < len(job_configs):
            # Use exact saved service_time to ensure identical scheduling decisions
            service_time = job_configs[job_id].get('service_time',
                                                    map_service_time(arousal, service_time_config))
            valence_class = job_configs[job_id].get('valence_class')
        elif enable_emotion:
            # Calculate service time from arousal
            service_time = map_service_time(arousal, service_time_config)
        else:
            # Use default service time if emotion disabled
            service_time = service_time_config.base_service_time

        if valence_class is None:
            valence_class = emotion_config.classify_valence(valence)

        arrival_time = arrival_times[job_id]

        # Create job with emotion attributes
        job = Job(
            job_id=job_id,
            execution_duration=service_time,
            arrival_time=arrival_time,
            emotion_label=emotion_label,
            arousal=arousal,
            emotion_class=emotion_class,
            valence=valence,
            valence_class=valence_class,
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


def _generate_bursty_arrivals(
        num_jobs: int,
        base_rate: float,
        burst_factor: float = 5.0,
        on_duration_mean: float = 10.0,
        off_duration_mean: float = 30.0) -> np.ndarray:
    """
    Generate bursty arrival times using ON/OFF model.

    In the ON state, arrivals occur at burst_factor * base_rate.
    In the OFF state, arrivals occur at base_rate / burst_factor.
    This creates periods of high activity (bursts) followed by quiet periods.

    Args:
        num_jobs: Number of jobs to generate
        base_rate: Base arrival rate (λ)
        burst_factor: Multiplier for arrival rate during ON state (default: 5.0)
        on_duration_mean: Mean duration of ON (burst) periods in seconds (default: 10.0)
        off_duration_mean: Mean duration of OFF (quiet) periods in seconds (default: 30.0)

    Returns:
        Array of arrival times with bursty pattern

    Example:
        With burst_factor=5, base_rate=1.0:
        - ON state: rate = 5.0 (high intensity burst)
        - OFF state: rate = 0.2 (quiet period)
    """
    arrival_times = []
    current_time = 0.0
    is_on_state = True  # Start with a burst
    state_end_time = np.random.exponential(on_duration_mean)

    while len(arrival_times) < num_jobs:
        # Determine current rate based on state
        if is_on_state:
            current_rate = base_rate * burst_factor
        else:
            current_rate = base_rate / burst_factor

        # Generate next arrival
        interval = np.random.exponential(1.0 / current_rate)
        current_time += interval

        # Check if we need to switch states
        while current_time > state_end_time:
            # Switch state
            is_on_state = not is_on_state
            if is_on_state:
                state_end_time += np.random.exponential(on_duration_mean)
            else:
                state_end_time += np.random.exponential(off_duration_mean)

        arrival_times.append(current_time)

    return np.array(arrival_times[:num_jobs])


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
    valence_classes = [job.valence_class for job in job_list if job.valence_class is not None]
    valence_counts = {
        'negative': valence_classes.count('negative'),
        'neutral': valence_classes.count('neutral'),
        'positive': valence_classes.count('positive'),
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
        'valence_class_counts': valence_counts,
        'total_duration': arrival_times[-1] if arrival_times else 0,
    }

    return stats

def generate_job_trace(
    num_jobs: int,
    arrival_rate: float,
    emotion_config: EmotionConfig = None,
    service_time_config: ServiceTimeConfig = None,
    enable_emotion: bool = True,
    random_seed: int = None,
    use_stratified_sampling: bool = True,  # for stratification
    class_distribution: Dict[str, float] = None,  # e.g., 9-class: {'high_positive': 1/9, ...}
    arrival_pattern: str = 'poisson',  # 'poisson' or 'bursty'
    burst_factor: float = 5.0,  # for bursty arrivals
    on_duration_mean: float = 10.0,  # mean ON period duration
    off_duration_mean: float = 30.0,  # mean OFF period duration
) -> List[Dict]:
    """
    Generate a complete job trace for reproducible experiments across schedulers.

    This creates a pre-determined sequence of job arrivals, emotions, and service times
    that can be reused by different schedulers for fair comparison.

    Args:
        num_jobs: Number of jobs to generate
        arrival_rate: Arrival rate (λ) for Poisson process
        emotion_config: EmotionConfig object
        service_time_config: ServiceTimeConfig object
        enable_emotion: Whether to enable emotion-aware features
        random_seed: Random seed for reproducibility
        use_stratified_sampling: Whether to use stratified sampling by 9-class category
        class_distribution: Target distribution for 9 classes (default: uniform 1/9 each)
        arrival_pattern: 'poisson' for standard Poisson, 'bursty' for ON/OFF bursty arrivals
        burst_factor: Rate multiplier during ON state (for bursty arrivals)
        on_duration_mean: Mean duration of burst periods in seconds
        off_duration_mean: Mean duration of quiet periods in seconds

    Returns:
        List of job configuration dictionaries
    """
    if emotion_config is None:
        emotion_config = EmotionConfig()
    if service_time_config is None:
        service_time_config = ServiceTimeConfig()

    # Set random seed
    if random_seed is not None:
        np.random.seed(random_seed)
        import random
        random.seed(random_seed)

    # Generate arrival times based on pattern
    if arrival_pattern == 'bursty':
        arrival_times = _generate_bursty_arrivals(
            num_jobs, arrival_rate, burst_factor, on_duration_mean, off_duration_mean
        )
        print(f"  Using bursty arrivals (factor={burst_factor}, ON={on_duration_mean}s, OFF={off_duration_mean}s)")
    else:
        arrival_times = _generate_arrival_times(num_jobs, arrival_rate)
        print(f"  Using Poisson arrivals (rate={arrival_rate})")

    # Generate emotions (9-class stratified or simple random)
    if enable_emotion:
        if use_stratified_sampling:
            from core.emotion import sample_emotions_batch_stratified_9class
            emotions_arousal_valence = sample_emotions_batch_stratified_9class(
                num_jobs, emotion_config, class_distribution
            )
        else:
            # Simple random sampling (returns 2-tuple, need to add valence)
            emotions_arousal_valence = []
            for emotion, arousal in sample_emotions_batch(num_jobs, emotion_config):
                valence = emotion_config.get_valence(emotion)
                emotions_arousal_valence.append((emotion, arousal, valence))
    else:
        emotions_arousal_valence = [('neutral', 0.0, 0.0)] * num_jobs

    # Create trace
    trace = []
    for job_id in range(num_jobs):
        emotion_label, arousal, valence = emotions_arousal_valence[job_id]
        valence_class = emotion_config.classify_valence(valence) if enable_emotion else 'neutral'
        service_time = map_service_time(arousal, service_time_config) if enable_emotion \
                      else service_time_config.base_service_time

        trace.append({
            'job_id': job_id,
            'arrival_time': float(arrival_times[job_id]),
            'emotion': emotion_label,
            'arousal': float(arousal),
            'valence': float(valence),
            'valence_class': valence_class,
            'service_time': float(service_time),
        })

    # Print 9-class distribution statistics
    if enable_emotion:
        category_counts = {cat: 0 for cat in emotion_config.get_categories()}
        for emotion_label, arousal, valence in emotions_arousal_valence:
            category = emotion_config.classify_emotion(arousal, valence)
            category_counts[category] += 1

        print(f"  9-class emotion distribution:")
        for category in emotion_config.get_categories():
            count = category_counts[category]
            pct = count / num_jobs * 100
            print(f"    {category}: {count} ({pct:.1f}%)")

    return trace



def shuffle_job_trace_arousal(trace_data, random_seed: int = None):
    """
    Shuffle arousal values across jobs while preserving service times.

    This breaks the emotion→length correlation while keeping the same
    service time distribution. Used to verify that SSJF-Emotion benefits
    come from the emotion-length relationship, not from data distribution.

    Args:
        trace_data: Either a list of job dicts, or a dict with 'jobs' key
                   (as saved by job_configs.json format)
        random_seed: Random seed for reproducibility

    Returns:
        Same format as input, with shuffled arousal/emotion assignments
    """
    import copy

    if random_seed is not None:
        np.random.seed(random_seed)

    # Handle both formats: list of jobs or {"metadata": ..., "jobs": [...]}
    if isinstance(trace_data, dict) and 'jobs' in trace_data:
        # Format: {"metadata": {...}, "jobs": [...]}
        shuffled_data = copy.deepcopy(trace_data)
        jobs = shuffled_data['jobs']
    else:
        # Format: list of job dicts
        shuffled_data = copy.deepcopy(trace_data)
        jobs = shuffled_data

    # Extract emotion-related fields
    arousal_values = [job['arousal'] for job in jobs]
    emotion_labels = [job['emotion'] for job in jobs]
    valence_values = [job['valence'] for job in jobs]
    valence_classes = [job.get('valence_class', 'neutral') for job in jobs]

    # Shuffle indices
    shuffle_indices = np.random.permutation(len(jobs))

    # Reassign shuffled emotion data (but keep service_time unchanged)
    for i, job in enumerate(jobs):
        shuffled_idx = shuffle_indices[i]
        job['arousal'] = arousal_values[shuffled_idx]
        job['emotion'] = emotion_labels[shuffled_idx]
        job['valence'] = valence_values[shuffled_idx]
        job['valence_class'] = valence_classes[shuffled_idx]
        # NOTE: service_time is NOT changed - this breaks the correlation

    return shuffled_data


def create_jobs_from_trace(
    trace: List[Dict],
    emotion_config: EmotionConfig = None,
    max_jobs: int = None
    ) -> List[Job]:
    """
    Create Job objects from a pre-generated trace.
    
    Args:
        trace: List of job configuration dictionaries
        emotion_config: EmotionConfig object (for arousal classification)
        max_jobs: Maximum number of jobs to create (None = all)
    
    Returns:
        List of Job objects
    """
    if emotion_config is None:
        emotion_config = EmotionConfig()
    
    jobs = []
    trace_subset = trace[:max_jobs] if max_jobs else trace
    
    for job_config in trace_subset:
        arousal = job_config['arousal']
        emotion_class = emotion_config.classify_arousal(arousal)
        valence = job_config.get('valence')
        if valence is None:
            if job_config['emotion'] in emotion_config.emotion_valence_map:
                valence = emotion_config.get_valence(job_config['emotion'])
            else:
                valence = 0.0
        valence_class = job_config.get('valence_class') or emotion_config.classify_valence(valence)
        
        job = Job(
            job_id=job_config['job_id'],
            execution_duration=job_config['service_time'],
            arrival_time=job_config['arrival_time'],
            emotion_label=job_config['emotion'],
            arousal=arousal,
            emotion_class=emotion_class,
            valence=valence,
            valence_class=valence_class,
        )
        jobs.append(job)
    
    return jobs
