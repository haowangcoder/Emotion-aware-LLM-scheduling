"""
Affect-Weighted Shortest-Service-Job-First (AW-SSJF) Scheduler

This scheduler implements the WSPT (Weighted Shortest Processing Time) rule
for emotion-aware job scheduling with a "Depression-First" strategy.

The core scheduling decision is based on the score:
    Score_i = S_i / w_i

where:
    - S_i is the predicted service time (from BERT predictor)
    - w_i is the affect weight (higher for depressed users)

Jobs with lower scores are scheduled first, meaning:
    - Short jobs get priority (SJF efficiency)
    - Depressed users get priority (affect-awareness)

The Depression-First strategy prioritizes users with:
    - Low valence (unpleasant emotions)
    - Low arousal (low activation/energy)

This combination targets users in a depressed/sad state who may need
faster responses to avoid further emotional deterioration.

References:
    - Smith, W. E. (1956). Various optimizers for single-stage production.
    - Russell, J. A. (1980). A circumplex model of affect.
"""

from typing import List, Optional, Dict
from collections import defaultdict

from core.scheduler_base import SchedulerBase
from core.job import Job
from core.affect_weight import affect_weight, compute_urgency


class AWSSJFScheduler(SchedulerBase):
    """
    Affect-Weighted Shortest-Service-Job-First (AW-SSJF) Scheduler.

    Implements WSPT-style scheduling with affect-based weights:
        Score_i = S_i / w_i

    Lower score = higher priority (scheduled first).

    The affect weight uses a "Depression-First" strategy:
    - Only negative valence AND low arousal users get priority boosts
    - Supports confidence discount to prevent "gaming" the system

    Args:
        w_max: Maximum affect weight (controls max "queue jumping" power).
               Default: 2.0. Recommended range: [1.2, 3.0].
        p: Exponent for negative valence. Default: 1.0 (linear).
        q: Exponent for low arousal. Default: 1.0 (linear).
        use_confidence: Whether to apply confidence discount. Default: True.
        starvation_threshold: Absolute time threshold for starvation prevention.
        starvation_coefficient: Relative threshold as multiple of service time.

    Example:
        >>> scheduler = AWSSJFScheduler(w_max=2.0)
        >>> next_job = scheduler.schedule(waiting_queue, current_time=10.0)
    """

    def __init__(
        self,
        w_max: float = 2.0,
        p: float = 1.0,
        q: float = 1.0,
        use_confidence: bool = True,
        starvation_threshold: float = float('inf'),
        starvation_coefficient: float = 3.0,
    ):
        super().__init__(name="AW-SSJF")
        self.w_max = w_max
        self.p = p
        self.q = q
        self.use_confidence = use_confidence
        self.starvation_threshold = starvation_threshold
        self.starvation_coefficient = starvation_coefficient

        # Statistics by Russell quadrant
        self.quadrant_stats = defaultdict(lambda: {
            'scheduled': 0,
            'total_waiting_time': 0,
            'total_weight': 0.0,
        })

    def _get_service_time(self, job: Job) -> float:
        """Get predicted service time from job."""
        # Priority: predicted_service_time > execution_duration > default
        if hasattr(job, 'predicted_service_time') and job.predicted_service_time is not None:
            return job.predicted_service_time
        if hasattr(job, 'execution_duration') and job.execution_duration is not None:
            return job.execution_duration
        return 2.0  # Default fallback

    def _compute_weight(self, job: Job) -> float:
        """Compute affect weight for a job."""
        arousal = getattr(job, 'arousal', 0.0) or 0.0
        valence = getattr(job, 'valence', 0.0) or 0.0
        confidence = getattr(job, 'emotion_confidence', 1.0) if self.use_confidence else 1.0

        return affect_weight(
            arousal=arousal,
            valence=valence,
            confidence=confidence,
            w_max=self.w_max,
            p=self.p,
            q=self.q
        )

    def _compute_score(self, job: Job) -> float:
        """
        Compute WSPT score: Score = S / w.

        Lower score = higher priority.
        """
        service_time = self._get_service_time(job)
        weight = self._compute_weight(job)

        # Avoid division by zero
        if weight <= 0:
            weight = 1.0

        return service_time / weight

    def schedule(self, waiting_queue: List[Job], current_time: float = 0) -> Optional[Job]:
        """
        Select the job with the lowest WSPT score.

        Implements starvation prevention:
        1. Absolute threshold: schedule if waiting_time > starvation_threshold
        2. Relative threshold: schedule if waiting_time > coefficient * service_time

        Args:
            waiting_queue: List of jobs waiting to be executed
            current_time: Current simulation time

        Returns:
            Job with lowest score, or None if queue is empty
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
            service_time = self._get_service_time(job)
            threshold = self.starvation_coefficient * service_time
            if waiting_time >= threshold:
                return job

        # Select job with lowest score
        # Ties broken by arrival time (FCFS among equal scores)
        selected_job = min(
            waiting_queue,
            key=lambda j: (self._compute_score(j), j.arrival_time)
        )

        return selected_job

    def on_job_scheduled(self, job: Job, current_time: float):
        """Track statistics when job is scheduled."""
        super().on_job_scheduled(job, current_time)

        waiting_time = current_time - job.arrival_time
        weight = self._compute_weight(job)

        # Track by Russell quadrant
        quadrant = getattr(job, 'russell_quadrant', None)
        if quadrant:
            stats = self.quadrant_stats[quadrant]
            stats['scheduled'] += 1
            stats['total_waiting_time'] += waiting_time
            stats['total_weight'] += weight

    def get_statistics(self) -> Dict:
        """Get comprehensive scheduler statistics."""
        base_stats = super().get_statistics()

        # Add per-quadrant statistics
        quadrant_stats = {}
        for quadrant, stats in self.quadrant_stats.items():
            if stats['scheduled'] > 0:
                avg_waiting = stats['total_waiting_time'] / stats['scheduled']
                avg_weight = stats['total_weight'] / stats['scheduled']
            else:
                avg_waiting = 0
                avg_weight = 1.0

            quadrant_stats[quadrant] = {
                'scheduled': stats['scheduled'],
                'avg_waiting_time': avg_waiting,
                'avg_weight': avg_weight,
            }

        base_stats['quadrant_stats'] = quadrant_stats
        base_stats['scheduler_params'] = {
            'w_max': self.w_max,
            'p': self.p,
            'q': self.q,
            'use_confidence': self.use_confidence,
        }

        return base_stats

    def get_quadrant_fairness(self) -> Dict[str, float]:
        """
        Calculate average waiting time for each Russell quadrant.

        Returns:
            Dictionary mapping quadrant -> average waiting time
        """
        fairness = {}
        for quadrant, stats in self.quadrant_stats.items():
            if stats['scheduled'] > 0:
                fairness[quadrant] = stats['total_waiting_time'] / stats['scheduled']
            else:
                fairness[quadrant] = 0.0
        return fairness

    def get_config(self) -> Dict:
        """Get scheduler configuration parameters."""
        return {
            'name': self.name,
            'w_max': self.w_max,
            'p': self.p,
            'q': self.q,
            'use_confidence': self.use_confidence,
            'starvation_threshold': self.starvation_threshold,
            'starvation_coefficient': self.starvation_coefficient,
        }
