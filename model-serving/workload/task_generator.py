"""
Emotion-aware Task Generator for LLM Scheduling

This module creates jobs/tasks with emotion attributes for the emotion-aware
scheduling simulator. It integrates:
1. Emotion sampling from emotion.py
2. Service time mapping from service_time_mapper.py
3. Arrival rate modification based on arousal
4. Support for multiple arrival distributions (Poisson, Gamma, Uniform)

The arrival rate is modified according to:
    λ(a) = λ_0 * (1 + γ * a)

where γ controls the sensitivity of arrival rate to arousal.
"""

import numpy as np
from typing import List, Dict, Tuple

from core.job import Job
from core.emotion import EmotionConfig, sample_emotion, sample_emotions_batch
from workload.service_time_mapper import ServiceTimeConfig, map_service_time



class EmotionAwareTaskConfig:
    """Configuration for emotion-aware task generation"""

    def __init__(self,
                 emotion_config: EmotionConfig = None,
                 service_time_config: ServiceTimeConfig = None,
                 base_arrival_rate: float = 1.0,
                 gamma: float = 0.5,
                 enable_emotion: bool = True):
        """
        Initialize emotion-aware task configuration

        Args:
            emotion_config: EmotionConfig for emotion sampling
            service_time_config: ServiceTimeConfig for service time mapping
            base_arrival_rate (λ_0): Base arrival rate when arousal is 0
            gamma (γ): Coefficient for arousal impact on arrival rate
            enable_emotion: Whether to enable emotion-aware features
        """
        self.emotion_config = emotion_config or EmotionConfig()
        self.service_time_config = service_time_config or ServiceTimeConfig()
        self.base_arrival_rate = base_arrival_rate
        self.gamma = gamma
        self.enable_emotion = enable_emotion

        # Validate gamma to ensure non-negative arrival rates
        if not 0 <= gamma < 1:
            raise ValueError(f"gamma must be in [0, 1), got {gamma}")

    def get_arrival_rate(self, arousal: float) -> float:
        """
        Calculate arrival rate for a given arousal level
        λ(a) = λ_0 * (1 + γ * a)

        Args:
            arousal: Arousal value

        Returns:
            Arrival rate (guaranteed >= 0)
        """
        arrival_rate = self.base_arrival_rate * (1 + self.gamma * arousal)
        return max(arrival_rate, 0.01)  # Ensure minimum positive rate


def create_emotion_aware_jobs(
        num_jobs: int,
        arrival_rate: float = 1.0,
        std: float = 1.0,
        coefficient_of_variance: float = 3.0,
        distribution: str = 'poisson',
        emotion_config: EmotionConfig = None,
        service_time_config: ServiceTimeConfig = None,
        gamma: float = 0.0,
        trace_scale: float = 1.0,
        enable_emotion: bool = True) -> List[Job]:
    """
    Create a list of jobs with emotion attributes

    Args:
        num_jobs: Number of jobs to create
        arrival_rate: Base arrival rate (λ_0)
        std: Standard deviation for uniform distribution
        coefficient_of_variance: CV for gamma distribution
        distribution: Arrival distribution type ('poisson', 'uniform', 'gamma')
        emotion_config: EmotionConfig object
        service_time_config: ServiceTimeConfig object
        gamma: Coefficient for arousal impact on arrival rate
        trace_scale: Scale factor for arrival time normalization (if used)
        enable_emotion: Whether to enable emotion-aware features

    Returns:
        List of Job objects with emotion attributes
    """
    if emotion_config is None:
        emotion_config = EmotionConfig()
    if service_time_config is None:
        service_time_config = ServiceTimeConfig()

    # Sample emotions for all jobs
    if enable_emotion:
        emotions_arousal = sample_emotions_batch(num_jobs, emotion_config)
    else:
        # Neutral emotion if disabled
        emotions_arousal = [('neutral', 0.0)] * num_jobs

    # Generate arrival times based on distribution
    arrival_times = _generate_arrival_times(
        num_jobs=num_jobs,
        arrival_rate=arrival_rate,
        std=std,
        coefficient_of_variance=coefficient_of_variance,
        distribution=distribution,
        emotions_arousal=emotions_arousal,
        gamma=gamma,
        trace_scale=trace_scale
    )

    # Create jobs with emotion attributes
    job_list = []
    for job_id in range(num_jobs):
        emotion_label, arousal = emotions_arousal[job_id]
        emotion_class = emotion_config.classify_arousal(arousal)

        # Map arousal to service time
        if enable_emotion:
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

        job_list.append(job)

    return job_list


def _generate_arrival_times(
        num_jobs: int,
        arrival_rate: float,
        std: float,
        coefficient_of_variance: float,
        distribution: str,
        emotions_arousal: List[Tuple[str, float]],
        gamma: float,
        trace_scale: float) -> np.ndarray:
    """
    Generate arrival times with emotion-aware arrival rate modification

    For emotion-aware generation, we have two approaches:
    1. Multi-stream: Separate Poisson processes for each emotion class
    2. Single-stream with modification: Adjust intervals based on individual arousal

    Currently implements approach 2 (single-stream with modification)

    Args:
        num_jobs: Number of jobs
        arrival_rate: Base arrival rate
        std: Standard deviation
        coefficient_of_variance: CV for gamma distribution
        distribution: Distribution type
        emotions_arousal: List of (emotion, arousal) tuples
        gamma: Arousal impact coefficient
        trace_scale: Scale factor for arrival time normalization (if used)

    Returns:
        Array of arrival times
    """
    # Generate base arrival times
    if distribution == 'poisson':
        request_list = np.random.poisson(arrival_rate, int(num_jobs / arrival_rate) + 10)
        while sum(request_list) < num_jobs:
            request_list = np.append(request_list,
                                      np.random.poisson(arrival_rate, int(num_jobs / arrival_rate) + 10))
        arrival_times = np.array(_generate_arrival_list(request_list, num_jobs), dtype=float)

    elif distribution == 'uniform':
        intervals = np.random.uniform(1 / (arrival_rate + std / 2),
                                       1 / (arrival_rate - std / 2),
                                       num_jobs)
        arrival_times = np.cumsum(intervals)

    elif distribution == 'gamma':
        shape = (1 / coefficient_of_variance) ** 2
        scale = coefficient_of_variance ** 2 / arrival_rate
        intervals = np.random.gamma(shape, scale, num_jobs)
        arrival_times = np.cumsum(intervals)

    else:
        raise ValueError(f"Unknown distribution: {distribution}. "
                         f"Choose from ['poisson', 'uniform', 'gamma']")

    # Apply emotion-based arrival rate modification if gamma > 0
    if gamma > 0 and distribution != 'poisson':
        # For non-Poisson distributions, modify intervals based on arousal
        # Higher arousal -> shorter intervals -> higher arrival rate
        intervals = np.diff(np.concatenate([[0], arrival_times]))
        modified_intervals = []

        for i, (emotion, arousal) in enumerate(emotions_arousal):
            # Rate multiplier based on arousal
            rate_multiplier = 1 + gamma * arousal
            # Interval is inversely proportional to rate
            modified_interval = intervals[i] / rate_multiplier if rate_multiplier > 0 else intervals[i]
            modified_intervals.append(modified_interval)

        arrival_times = np.cumsum(modified_intervals)

    return arrival_times


def _generate_arrival_list(request_list, total_num_requests):
    """Generate a list of arrival times from per-timestep request counts."""
    arrival_list = []
    current_num_requests = 0
    for t, num_requests in enumerate(request_list):
        for _ in range(num_requests):
            arrival_list.append(t)
            current_num_requests += 1
            if current_num_requests == total_num_requests:
                return arrival_list
    return arrival_list


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
        distribution='poisson',
        emotion_config=emotion_config,
        service_time_config=service_config,
        gamma=0.3
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
        distribution='gamma',
        gamma=0.5,
        enable_emotion=True
    )
    jobs_without_emotion = create_emotion_aware_jobs(
        num_jobs=100,
        arrival_rate=2.0,
        distribution='gamma',
        gamma=0.5,
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
