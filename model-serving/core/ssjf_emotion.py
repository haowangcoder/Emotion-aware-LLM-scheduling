"""
Emotion-aware SSJF (Shortest-Service-Job-First) Scheduler

This scheduler implements Shortest-Service-Job-First scheduling for emotion-aware
jobs, where service time is computed from emotion arousal values using the formula:
    S_i = L_0 * (1 + α * a_i)

The scheduler prioritizes jobs with shorter expected service times, which are
directly influenced by their emotional arousal levels. This can improve average
response time compared to FCFS, but may affect fairness across emotion categories.
"""

from typing import List, Optional, Dict
from collections import defaultdict

from core.scheduler_base import SchedulerBase
from core.job import Job


class SSJFEmotionScheduler(SchedulerBase):
    """
    Speculative Shortest-Job-First scheduler for emotion-aware jobs

    Schedules jobs based on their service time (execution_duration), which has
    been computed from arousal values. Includes starvation prevention to ensure
    long jobs eventually get executed.
    """

    def __init__(self,
                 starvation_threshold: float = float('inf'),
                 starvation_coefficient: float = 2.0):
        """
        Initialize SSJF Emotion scheduler

        Args:
            starvation_threshold: Absolute time threshold for starvation prevention
            starvation_coefficient: Relative threshold as multiple of execution duration
        """
        super().__init__(name="SSJF-Emotion")
        self.starvation_threshold = starvation_threshold
        self.starvation_coefficient = starvation_coefficient

        # Track statistics by emotion class
        self.emotion_class_stats = defaultdict(lambda: {
            'scheduled': 0,
            'total_waiting_time': 0
        })

    def schedule(self, waiting_queue: List[Job], current_time: float = 0) -> Optional[Job]:
        """
        Select the job with the shortest service time (execution_duration)

        Implements two levels of starvation prevention:
        1. Absolute threshold: if waiting_time > starvation_threshold
        2. Relative threshold: if waiting_time > starvation_coefficient * execution_duration

        Args:
            waiting_queue: List of jobs waiting to be executed
            current_time: Current time

        Returns:
            Job with shortest service time (or starving job), or None if queue is empty
        """
        if not waiting_queue:
            return None

        # Check for starving jobs (absolute threshold)
        if self.starvation_threshold < float('inf'):
            for job in waiting_queue:
                waiting_time = current_time - job.arrival_time
                if waiting_time >= self.starvation_threshold:
                    return job

        # Check for starving jobs (relative threshold)
        for job in waiting_queue:
            waiting_time = current_time - job.arrival_time
            threshold = self.starvation_coefficient * job.execution_duration
            if waiting_time >= threshold:
                return job

        # Select job with shortest execution duration (service time)
        shortest_job = min(waiting_queue, key=lambda j: j.execution_duration)

        return shortest_job

    def on_job_scheduled(self, job: Job, current_time: float):
        """
        Track statistics when a job is scheduled

        Args:
            job: The scheduled job
            current_time: Current time
        """
        super().on_job_scheduled(job, current_time)

        # Track per-emotion-class statistics
        if job.emotion_class is not None:
            stats = self.emotion_class_stats[job.emotion_class]
            stats['scheduled'] += 1
            stats['total_waiting_time'] += (current_time - job.arrival_time)

    def get_statistics(self) -> dict:
        """
        Get comprehensive scheduler statistics including emotion class breakdown

        Returns:
            Dictionary with scheduler statistics
        """
        base_stats = super().get_statistics()

        # Add per-emotion-class statistics
        emotion_stats = {}
        for emotion_class, stats in self.emotion_class_stats.items():
            avg_waiting = (stats['total_waiting_time'] / stats['scheduled']
                           if stats['scheduled'] > 0 else 0)
            emotion_stats[emotion_class] = {
                'scheduled': stats['scheduled'],
                'avg_waiting_time': avg_waiting
            }

        base_stats['emotion_class_stats'] = emotion_stats
        return base_stats

    def get_emotion_class_fairness(self) -> Dict[str, float]:
        """
        Calculate average waiting time for each emotion class

        Returns:
            Dictionary mapping emotion_class -> average waiting time
        """
        fairness_metrics = {}

        for emotion_class, stats in self.emotion_class_stats.items():
            avg_waiting = (stats['total_waiting_time'] / stats['scheduled']
                           if stats['scheduled'] > 0 else 0)
            fairness_metrics[emotion_class] = avg_waiting

        return fairness_metrics


class SSJFEmotionPriorityScheduler(SchedulerBase):
    """
    Advanced SSJF scheduler with emotion-based priority adjustment

    In addition to service time, this scheduler can adjust priority based on
    emotion class to improve fairness or prioritize certain emotional states.
    """

    def __init__(self,
                 priority_weights: Dict[str, float] = None,
                 starvation_threshold: float = float('inf')):
        """
        Initialize SSJF with emotion priority

        Args:
            priority_weights: Weights for each emotion class (default: equal weights)
            starvation_threshold: Absolute time threshold for starvation prevention
        """
        super().__init__(name="SSJF-Emotion-Priority")
        self.priority_weights = priority_weights or {
            'high': 1.0,
            'medium': 1.0,
            'low': 1.0
        }
        self.starvation_threshold = starvation_threshold

    def schedule(self, waiting_queue: List[Job], current_time: float = 0) -> Optional[Job]:
        """
        Select job based on weighted service time

        Effective service time = service_time / priority_weight
        Lower effective service time -> higher priority

        Args:
            waiting_queue: List of jobs waiting to be executed
            current_time: Current time

        Returns:
            Job with lowest weighted service time, or None if queue is empty
        """
        if not waiting_queue:
            return None

        # Check for starving jobs
        if self.starvation_threshold < float('inf'):
            for job in waiting_queue:
                waiting_time = current_time - job.arrival_time
                if waiting_time >= self.starvation_threshold:
                    return job

        # Calculate weighted service time for each job
        def get_priority_score(job: Job) -> float:
            weight = self.priority_weights.get(job.emotion_class, 1.0)
            # Lower score = higher priority
            return job.execution_duration / weight if weight > 0 else float('inf')

        # Select job with lowest priority score
        selected_job = min(waiting_queue, key=get_priority_score)

        return selected_job
