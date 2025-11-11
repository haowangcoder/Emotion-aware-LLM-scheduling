"""
Scheduler Configuration.

Contains scheduling algorithm selection, system load parameters,
and starvation prevention settings for emotion-aware schedulers.
"""
import os


# ============================================================================
# SCHEDULER CONFIGURATION
# ============================================================================

# Scheduling Algorithm Selection
SCHEDULER_ALGORITHM = 'FCFS'  # 'FCFS', 'SJF', 'SJFP', 'SSJF-Emotion'
SCHEDULER_ALGORITHM = os.environ.get('SCHEDULER_ALGORITHM_ENV', SCHEDULER_ALGORITHM)

# System Load Parameter
# ρ = (λ * E[S]) / N where N=1 (single GPU)
# Target load levels for experiments: [0.4, 0.6, 0.8, 0.9]
SYSTEM_LOAD = 0.6  # Target system load (ρ)
SYSTEM_LOAD = float(os.environ.get('SYSTEM_LOAD_ENV', SYSTEM_LOAD))

# Starvation Prevention for SSJF-Emotion
EMOTION_SSJF_STARVATION_THRESHOLD = 100.0  # Absolute time threshold
EMOTION_SSJF_STARVATION_COEFFICIENT = 3.0  # Relative threshold (multiple of execution duration)

# Environment variable overrides
EMOTION_SSJF_STARVATION_THRESHOLD = float(os.environ.get('EMOTION_SSJF_STARVATION_THRESHOLD_ENV',
                                                          EMOTION_SSJF_STARVATION_THRESHOLD))
EMOTION_SSJF_STARVATION_COEFFICIENT = float(os.environ.get('EMOTION_SSJF_STARVATION_COEFFICIENT_ENV',
                                                            EMOTION_SSJF_STARVATION_COEFFICIENT))


# ============================================================================
# SCHEDULER PARAMETER DOCUMENTATION
# ============================================================================
"""
Scheduler Parameters:

1. SCHEDULER_ALGORITHM:
   - Options: 'FCFS', 'SJF', 'SJFP', 'SSJF-Emotion'
   - Default: 'FCFS'
   - Description: Scheduling strategy to use
   - FCFS: Baseline first-come-first-served
   - SSJF-Emotion: Shortest-service-job-first with emotion-aware service times

2. SYSTEM_LOAD (ρ):
   - Range: 0 < ρ < 1
   - Default: 0.6
   - Description: Target system utilization
   - Formula: ρ = (λ * E[S]) / N
   - Recommended test values: [0.4, 0.6, 0.8, 0.9]

3. EMOTION_SSJF_STARVATION_THRESHOLD:
   - Range: > 0
   - Default: 100.0
   - Description: Absolute time threshold for starvation prevention

4. EMOTION_SSJF_STARVATION_COEFFICIENT:
   - Range: > 1
   - Default: 3.0
   - Description: Relative threshold (multiple of execution duration)
"""
