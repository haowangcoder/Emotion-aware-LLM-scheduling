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

v2.0 Improvements:
    - Supports multiple weight modes: HARD, SOFT, DUAL_CHANNEL
    - Soft gating (sigmoid) for noise robustness
    - Dual-channel: Depression-First + Panic-Second

References:
    - Russell, J. A. (1980). A circumplex model of affect.
"""

from typing import List, Optional, Dict
from collections import defaultdict

from core.scheduler_base import SchedulerBase
from core.job import Job
from core.affect_weight import affect_weight
from core.affect_weight_v2 import (
    WeightConfig, WeightMode, affect_weight_v2, compute_urgency_v2,
    get_preset_config
)


class WeightOnlyScheduler(SchedulerBase):
    """
    Weight-Only Scheduler (Pure Affect-Based Priority).

    Schedules jobs purely based on affect weight, ignoring service time.
    Higher weight = higher priority (scheduled first).

    This scheduler demonstrates the extreme case of emotional fairness
    without considering system efficiency.

    v2.0: Now supports multiple weight modes via WeightConfig:
    - HARD: Original hard gating (v1.0 compatible)
    - SOFT: Sigmoid soft gating (recommended, noise-robust)
    - DUAL_CHANNEL: Depression-First + Panic-Second

    Args:
        w_max: Maximum affect weight. Default: 2.0.
        p: Exponent for negative valence. Default: 1.0.
        q: Exponent for low arousal. Default: 1.0.
        use_confidence: Whether to apply confidence discount. Default: True.
        starvation_threshold: Absolute time threshold for starvation prevention.
        weight_config: v2.0 WeightConfig object (overrides w_max, p, q if provided)
        weight_preset: Name of preset config ('depression_first_soft', etc.)

    Example:
        >>> # Legacy mode (v1.0 compatible)
        >>> scheduler = WeightOnlyScheduler(w_max=2.0)
        >>>
        >>> # v2.0 with preset
        >>> scheduler = WeightOnlyScheduler(weight_preset='dual_channel_balanced')
        >>>
        >>> # v2.0 with custom config
        >>> config = WeightConfig(mode=WeightMode.DUAL_CHANNEL, gamma_panic=0.3)
        >>> scheduler = WeightOnlyScheduler(weight_config=config)
    """

    def __init__(
        self,
        w_max: float = 2.0,
        p: float = 1.0,
        q: float = 1.0,
        use_confidence: bool = True,
        starvation_threshold: float = float('inf'),
        weight_config: Optional[WeightConfig] = None,
        weight_preset: Optional[str] = None,
    ):
        super().__init__(name="Weight-Only")

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

        # Statistics by Russell quadrant
        self.quadrant_stats = defaultdict(lambda: {
            'scheduled': 0,
            'total_waiting_time': 0,
            'total_weight': 0.0,
        })

    def _compute_weight(self, job: Job) -> float:
        """
        Compute affect weight for a job.

        Also updates job.affect_weight and job.urgency to ensure logging
        reflects the actual values used for scheduling.
        """
        arousal = getattr(job, 'arousal', 0.0) or 0.0
        valence = getattr(job, 'valence', 0.0) or 0.0
        confidence = getattr(job, 'emotion_confidence', 1.0) if self.use_confidence else 1.0

        if self.use_v2_weights and self.weight_config is not None:
            # v2.0 weight calculation
            weight = affect_weight_v2(
                arousal=arousal,
                valence=valence,
                confidence=confidence,
                config=self.weight_config
            )
            # Update job object with v2 computed values for accurate logging
            urgency = compute_urgency_v2(arousal, valence, self.weight_config)
            job.affect_weight = weight
            job.urgency = urgency
            return weight
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
