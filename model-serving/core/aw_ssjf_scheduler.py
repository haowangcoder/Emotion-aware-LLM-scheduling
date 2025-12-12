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

v2.0 Improvements:
    - Supports multiple weight modes: HARD, SOFT, DUAL_CHANNEL
    - Soft gating (sigmoid) for noise robustness
    - Dual-channel: Depression-First + Panic-Second

References:
    - Smith, W. E. (1956). Various optimizers for single-stage production.
    - Russell, J. A. (1980). A circumplex model of affect.
"""

from typing import List, Optional, Dict, Union
from collections import defaultdict

from core.scheduler_base import SchedulerBase
from core.job import Job
from core.affect_weight import affect_weight, compute_urgency
from core.affect_weight_v2 import (
    WeightConfig, WeightMode, affect_weight_v2, compute_urgency_v2,
    get_preset_config, get_detailed_weight_info
)


class AWSSJFScheduler(SchedulerBase):
    """
    Affect-Weighted Shortest-Service-Job-First (AW-SSJF) Scheduler.

    Implements WSPT-style scheduling with affect-based weights:
        Score_i = S_i / w_i

    Lower score = higher priority (scheduled first).

    The affect weight uses a "Depression-First" strategy:
    - Only negative valence AND low arousal users get priority boosts
    - Supports confidence discount to prevent "gaming" the system

    v2.0: Now supports multiple weight modes via WeightConfig:
    - HARD: Original hard gating (v1.0 compatible)
    - SOFT: Sigmoid soft gating (recommended, noise-robust)
    - DUAL_CHANNEL: Depression-First + Panic-Second

    Args:
        w_max: Maximum affect weight (controls max "queue jumping" power).
               Default: 2.0. Recommended range: [1.2, 3.0].
        p: Exponent for negative valence. Default: 1.0 (linear).
        q: Exponent for low arousal. Default: 1.0 (linear).
        use_confidence: Whether to apply confidence discount. Default: True.
        starvation_threshold: Absolute time threshold for starvation prevention.
        starvation_coefficient: Relative threshold as multiple of service time.
        weight_config: v2.0 WeightConfig object (overrides w_max, p, q if provided)
        weight_preset: Name of preset config ('depression_first_soft', etc.)

    Example:
        >>> # Legacy mode (v1.0 compatible)
        >>> scheduler = AWSSJFScheduler(w_max=2.0)
        >>>
        >>> # v2.0 with preset
        >>> scheduler = AWSSJFScheduler(weight_preset='depression_first_soft')
        >>>
        >>> # v2.0 with custom config
        >>> config = WeightConfig(mode=WeightMode.DUAL_CHANNEL, gamma_panic=0.3)
        >>> scheduler = AWSSJFScheduler(weight_config=config)
    """

    def __init__(
        self,
        w_max: float = 2.0,
        p: float = 1.0,
        q: float = 1.0,
        use_confidence: bool = True,
        starvation_threshold: float = float('inf'),
        starvation_coefficient: float = 3.0,
        weight_config: Optional[WeightConfig] = None,
        weight_preset: Optional[str] = None,
    ):
        super().__init__(name="AW-SSJF")

        # v2.0 weight configuration (takes precedence over legacy params)
        if weight_preset is not None:
            self.weight_config = get_preset_config(weight_preset)
            self.use_v2_weights = True
        elif weight_config is not None:
            self.weight_config = weight_config
            self.use_v2_weights = True
        else:
            # Legacy mode: use original v1.0 weight calculation
            self.weight_config = None
            self.use_v2_weights = False

        # Legacy parameters (used when use_v2_weights=False)
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

        # v2.0: Track detailed weight info for analysis
        self.weight_details_log = []

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

        if self.use_v2_weights and self.weight_config is not None:
            # v2.0 weight calculation
            return affect_weight_v2(
                arousal=arousal,
                valence=valence,
                confidence=confidence,
                config=self.weight_config
            )
        else:
            # Legacy v1.0 weight calculation
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

        # Include weight configuration info
        if self.use_v2_weights and self.weight_config is not None:
            base_stats['scheduler_params'] = {
                'version': 'v2.0',
                'weight_mode': self.weight_config.mode.value,
                'w_max': self.weight_config.w_max,
                'p': self.weight_config.p,
                'q': self.weight_config.q,
                'k_v': self.weight_config.k_v,
                'k_a': self.weight_config.k_a,
                'use_confidence': self.use_confidence,
            }
            if self.weight_config.mode == WeightMode.DUAL_CHANNEL:
                base_stats['scheduler_params'].update({
                    'r': self.weight_config.r,
                    'gamma_dep': self.weight_config.gamma_dep,
                    'gamma_panic': self.weight_config.gamma_panic,
                })
        else:
            base_stats['scheduler_params'] = {
                'version': 'v1.0',
                'weight_mode': 'hard',
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
        config = {
            'name': self.name,
            'starvation_threshold': self.starvation_threshold,
            'starvation_coefficient': self.starvation_coefficient,
            'use_confidence': self.use_confidence,
        }

        if self.use_v2_weights and self.weight_config is not None:
            config.update({
                'weight_version': 'v2.0',
                'weight_mode': self.weight_config.mode.value,
                'w_max': self.weight_config.w_max,
                'p': self.weight_config.p,
                'q': self.weight_config.q,
                'k_v': self.weight_config.k_v,
                'k_a': self.weight_config.k_a,
                'tau_v': self.weight_config.tau_v,
                'tau_a': self.weight_config.tau_a,
            })
            if self.weight_config.mode == WeightMode.DUAL_CHANNEL:
                config.update({
                    'r': self.weight_config.r,
                    'tau_h': self.weight_config.tau_h,
                    'gamma_dep': self.weight_config.gamma_dep,
                    'gamma_panic': self.weight_config.gamma_panic,
                })
        else:
            config.update({
                'weight_version': 'v1.0',
                'weight_mode': 'hard',
                'w_max': self.w_max,
                'p': self.p,
                'q': self.q,
            })

        return config
