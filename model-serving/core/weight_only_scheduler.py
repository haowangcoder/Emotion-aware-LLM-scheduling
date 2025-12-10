"""
Weight-Only Scheduler (Baseline for AW-SSJF Comparison)

This scheduler implements pure affect-based scheduling, completely ignoring
service time. Jobs are scheduled based solely on their affect weight,
with higher weights getting priority.

This scheduler serves as a baseline to demonstrate:
1. The importance of service time in scheduling efficiency
2. The trade-off between emotional fairness and system throughput

Expected Results:
    - Better emotional fairness (lower latency for depressed users)
    - Worse overall average latency (ignoring SJF efficiency)

Comparison with AW-SSJF:
    - AW-SSJF: Score = S / w (balances efficiency and fairness)
    - Weight-Only: Score = -w (pure fairness, ignores efficiency)

References:
    - Russell, J. A. (1980). A circumplex model of affect.
"""

from typing import List, Optional, Dict
from collections import defaultdict

from core.scheduler_base import SchedulerBase
from core.job import Job
from core.affect_weight import affect_weight


class WeightOnlyScheduler(SchedulerBase):
    """
    Weight-Only Scheduler (Pure Affect-Based Priority).

    Schedules jobs purely based on affect weight, ignoring service time.
    Higher weight = higher priority (scheduled first).

    This scheduler demonstrates the extreme case of emotional fairness
    without considering system efficiency.

    Args:
        w_max: Maximum affect weight. Default: 2.0.
        p: Exponent for negative valence. Default: 1.0.
        q: Exponent for low arousal. Default: 1.0.
        use_confidence: Whether to apply confidence discount. Default: True.
        starvation_threshold: Absolute time threshold for starvation prevention.

    Example:
        >>> scheduler = WeightOnlyScheduler(w_max=2.0)
        >>> next_job = scheduler.schedule(waiting_queue, current_time=10.0)
    """

    def __init__(
        self,
        w_max: float = 2.0,
        p: float = 1.0,
        q: float = 1.0,
        use_confidence: bool = True,
        starvation_threshold: float = float('inf'),
    ):
        super().__init__(name="Weight-Only")
        self.w_max = w_max
        self.p = p
        self.q = q
        self.use_confidence = use_confidence
        self.starvation_threshold = starvation_threshold

        # Statistics by Russell quadrant
        self.quadrant_stats = defaultdict(lambda: {
            'scheduled': 0,
            'total_waiting_time': 0,
            'total_weight': 0.0,
        })

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

    def schedule(self, waiting_queue: List[Job], current_time: float = 0) -> Optional[Job]:
        """
        Select the job with the highest affect weight.

        Higher weight = higher priority (depressed users first).
        Ties are broken by arrival time (FCFS among equal weights).

        Args:
            waiting_queue: List of jobs waiting to be executed
            current_time: Current simulation time

        Returns:
            Job with highest weight, or None if queue is empty
        """
        if not waiting_queue:
            return None

        # Starvation prevention
        if self.starvation_threshold < float('inf'):
            for job in waiting_queue:
                waiting_time = current_time - job.arrival_time
                if waiting_time >= self.starvation_threshold:
                    return job

        # Select job with highest weight
        # Ties broken by earliest arrival time
        selected_job = max(
            waiting_queue,
            key=lambda j: (self._compute_weight(j), -j.arrival_time)
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
        }
