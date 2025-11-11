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


# Example usage and testing
if __name__ == '__main__':
    from job import Job
    import random

    print("=" * 70)
    print("SSJF Emotion Scheduler Test")
    print("=" * 70)

    # Create emotion-aware test jobs
    random.seed(42)
    emotions = ['excited', 'sad', 'angry', 'calm', 'neutral']
    arousals = [0.9, -0.6, 0.8, -0.3, 0.0]
    emotion_classes = ['high', 'low', 'high', 'low', 'medium']

    # Service times computed from arousal
    # S_i = 2.0 * (1 + 0.5 * a_i)
    service_times = [2.0 * (1 + 0.5 * a) for a in arousals]

    jobs = []
    for i in range(5):
        job = Job(
            job_id=i,
            execution_duration=service_times[i],
            arrival_time=i * 1.0,
            emotion_label=emotions[i],
            arousal=arousals[i],
            emotion_class=emotion_classes[i]
        )
        jobs.append(job)

    # Test 1: Basic SSJF Emotion scheduling
    print("\n1. Basic SSJF Emotion Scheduling")
    scheduler = SSJFEmotionScheduler()
    queue = jobs.copy()

    print(f"\n   {'Job':<5} {'Emotion':<10} {'Arousal':<10} {'Class':<10} {'Service Time':<15}")
    print(f"   {'-'*55}")
    for job in queue:
        print(f"   {job.job_id:<5} {job.emotion_label:<10} {job.arousal:<10.2f} "
              f"{job.emotion_class:<10} {job.execution_duration:<15.3f}")

    print(f"\n   Scheduling order (by service time): ", end="")
    scheduled_order = []
    while queue:
        job = scheduler.schedule(queue, current_time=0)
        print(f"{job.job_id} ", end="")
        scheduled_order.append(job.job_id)
        scheduler.on_job_scheduled(job, current_time=len(scheduled_order))
        queue.remove(job)
    print()

    # Test 2: Starvation prevention
    print("\n2. Starvation Prevention Test")
    scheduler_safe = SSJFEmotionScheduler(
        starvation_threshold=10.0,
        starvation_coefficient=3.0
    )
    queue = jobs.copy()

    # Simulate: jobs arrived at different times, current time is 15
    current_time = 15.0
    for job in queue:
        job.arrival_time = 0  # All arrived at time 0

    print(f"   Current time: {current_time}")
    print(f"   Starvation threshold: {scheduler_safe.starvation_threshold}")
    print(f"   All jobs waiting for: {current_time} seconds")

    selected = scheduler_safe.schedule(queue, current_time=current_time)
    print(f"   Selected job (should trigger starvation): Job {selected.job_id}")
    print(f"   Job {selected.job_id} service time: {selected.execution_duration:.3f}")

    # Test 3: Emotion class statistics
    print("\n3. Emotion Class Statistics")
    scheduler_stats = SSJFEmotionScheduler()
    queue = jobs.copy()

    current_time = 0
    while queue:
        job = scheduler_stats.schedule(queue, current_time=current_time)
        scheduler_stats.on_job_scheduled(job, current_time=current_time)
        current_time += job.execution_duration
        queue.remove(job)

    stats = scheduler_stats.get_statistics()
    print(f"\n   Overall Statistics:")
    print(f"     Scheduled count: {stats['scheduled_count']}")
    print(f"     Average waiting time: {stats['avg_waiting_time']:.3f}")

    print(f"\n   Per-Emotion-Class Statistics:")
    for emotion_class, class_stats in stats['emotion_class_stats'].items():
        print(f"     {emotion_class}:")
        print(f"       Scheduled: {class_stats['scheduled']}")
        print(f"       Avg waiting time: {class_stats['avg_waiting_time']:.3f}")

    # Test 4: Priority-adjusted scheduling
    print("\n4. Priority-Adjusted SSJF Scheduling")
    # Give higher weight (priority) to low arousal emotions
    priority_scheduler = SSJFEmotionPriorityScheduler(
        priority_weights={'high': 0.8, 'medium': 1.0, 'low': 1.5}
    )
    queue = jobs.copy()

    print(f"   Priority weights: {priority_scheduler.priority_weights}")
    print(f"   (Higher weight = higher priority = scheduled sooner)")
    print(f"\n   Scheduling order: ", end="")

    while queue:
        job = priority_scheduler.schedule(queue, current_time=0)
        print(f"{job.job_id}({job.emotion_class}) ", end="")
        queue.remove(job)
    print()

    # Test 5: Compare SSJF vs FCFS order
    print("\n5. Comparison: SSJF-Emotion vs FCFS")
    from scheduler_base import FCFSScheduler

    fcfs = FCFSScheduler()
    ssjf = SSJFEmotionScheduler()

    queue_fcfs = jobs.copy()
    queue_ssjf = jobs.copy()

    print(f"\n   FCFS order: ", end="")
    while queue_fcfs:
        job = fcfs.schedule(queue_fcfs)
        print(f"{job.job_id} ", end="")
        queue_fcfs.remove(job)

    print(f"\n   SSJF order: ", end="")
    while queue_ssjf:
        job = ssjf.schedule(queue_ssjf)
        print(f"{job.job_id} ", end="")
        queue_ssjf.remove(job)
    print()

    print("\n" + "=" * 70)
