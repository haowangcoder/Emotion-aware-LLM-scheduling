"""
Affect Weight Module v2.0 - Improved Weight Computation

This module extends the original Depression-First weight calculation with:
1. Soft gating (sigmoid) - Eliminates discontinuity at 0, more robust to noise
2. Dual-channel weights - Depression-First + Panic-Second for layered response
3. Backward compatibility - Original HARD mode still available

Key Improvements over v1.0:
- Sigmoid soft gating replaces hard max(0, -x) for smoother transitions
- Panic channel gives limited priority to high-arousal negative emotions
- Configurable weight modes for easy ablation studies

Mathematical Formulation:
------------------------
SOFT mode (sigmoid gating):
    n = σ(k_v * (-v - τ_v))    # Negative valence intensity
    ℓ = σ(k_a * (-a - τ_a))    # Low arousal intensity
    u = n^p * ℓ^q              # Depression urgency
    w = 1 + (w_max - 1) * c * u

DUAL_CHANNEL mode:
    u_dep = n^p * ℓ^q                    # Depression urgency
    u_panic = n^p * h^r, h = σ(k_a*(a-τ_h))  # Panic urgency
    u = γ_dep * u_dep + γ_panic * u_panic
    w = 1 + (w_max - 1) * c * u

References:
    - Russell, J. A. (1980). A circumplex model of affect.
    - Smith, W. E. (1956). Various optimizers for single-stage production.
"""

import math
from enum import Enum
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


class WeightMode(Enum):
    """Weight computation modes."""
    HARD = "hard"           # Original hard gating: max(0, -x)
    SOFT = "soft"           # Soft gating: sigmoid
    DUAL_CHANNEL = "dual"   # Depression-First + Panic-Second


@dataclass
class WeightConfig:
    """Configuration for affect weight computation.

    Attributes:
        mode: Weight computation mode (HARD, SOFT, DUAL_CHANNEL)
        w_max: Maximum weight (controls max "queue jumping" power)
        p: Exponent for negative valence
        q: Exponent for low arousal
        r: Exponent for high arousal (DUAL_CHANNEL only)

        # Soft gating parameters (SOFT and DUAL_CHANNEL modes)
        k_v: Steepness for valence sigmoid (higher = sharper transition)
        k_a: Steepness for arousal sigmoid
        tau_v: Threshold for negative valence (default 0)
        tau_a: Threshold for low arousal (default 0)
        tau_h: Threshold for high arousal (default 0, DUAL_CHANNEL only)

        # Dual-channel parameters
        gamma_dep: Weight for depression channel (default 1.0)
        gamma_panic: Weight for panic channel (default 0.3)
    """
    mode: WeightMode = WeightMode.SOFT
    w_max: float = 2.0
    p: float = 1.0
    q: float = 1.0
    r: float = 1.0

    # Soft gating parameters
    k_v: float = 5.0      # Steepness for valence (5.0 gives smooth but decisive transition)
    k_a: float = 5.0      # Steepness for arousal
    tau_v: float = 0.0    # Threshold for negative valence
    tau_a: float = 0.0    # Threshold for low arousal
    tau_h: float = 0.0    # Threshold for high arousal

    # Dual-channel parameters
    gamma_dep: float = 1.0    # Depression channel weight
    gamma_panic: float = 0.3  # Panic channel weight (lower = less priority)


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def compute_urgency_hard(
    arousal: float,
    valence: float,
    p: float = 1.0,
    q: float = 1.0
) -> float:
    """
    Original hard gating urgency computation.

    u = max(0, -v)^p * max(0, -a)^q

    This has discontinuity at v=0 and a=0.
    """
    n = max(0.0, -valence)
    ell = max(0.0, -arousal)
    return (n ** p) * (ell ** q)


def compute_urgency_soft(
    arousal: float,
    valence: float,
    p: float = 1.0,
    q: float = 1.0,
    k_v: float = 5.0,
    k_a: float = 5.0,
    tau_v: float = 0.0,
    tau_a: float = 0.0
) -> float:
    """
    Soft gating urgency computation using sigmoid.

    n = σ(k_v * (-v - τ_v))
    ℓ = σ(k_a * (-a - τ_a))
    u = n^p * ℓ^q

    Advantages:
    - Continuous and differentiable everywhere
    - Smooth transition around thresholds
    - More robust to noise in emotion predictions

    Args:
        arousal: Arousal value in [-1, 1]
        valence: Valence value in [-1, 1]
        p: Exponent for negative valence
        q: Exponent for low arousal
        k_v: Steepness for valence sigmoid
        k_a: Steepness for arousal sigmoid
        tau_v: Threshold for "negative enough" valence
        tau_a: Threshold for "low enough" arousal

    Returns:
        Urgency score u in [0, 1]
    """
    # Soft gating with sigmoid
    n = sigmoid(k_v * (-valence - tau_v))
    ell = sigmoid(k_a * (-arousal - tau_a))

    return (n ** p) * (ell ** q)


def compute_urgency_dual_channel(
    arousal: float,
    valence: float,
    p: float = 1.0,
    q: float = 1.0,
    r: float = 1.0,
    k_v: float = 5.0,
    k_a: float = 5.0,
    tau_v: float = 0.0,
    tau_a: float = 0.0,
    tau_h: float = 0.0,
    gamma_dep: float = 1.0,
    gamma_panic: float = 0.3
) -> Tuple[float, float, float]:
    """
    Dual-channel urgency: Depression-First + Panic-Second.

    Depression channel (low valence AND low arousal):
        u_dep = n^p * ℓ^q

    Panic channel (low valence AND high arousal):
        u_panic = n^p * h^r, where h = σ(k_a*(a - τ_h))

    Combined urgency:
        u = γ_dep * u_dep + γ_panic * u_panic

    Args:
        arousal, valence: Emotion coordinates
        p, q, r: Exponents
        k_v, k_a: Sigmoid steepness
        tau_v, tau_a, tau_h: Thresholds
        gamma_dep: Depression channel weight (default 1.0)
        gamma_panic: Panic channel weight (default 0.3)

    Returns:
        Tuple of (total_urgency, depression_urgency, panic_urgency)
    """
    # Negative valence intensity (shared by both channels)
    n = sigmoid(k_v * (-valence - tau_v))

    # Low arousal intensity (depression channel)
    ell = sigmoid(k_a * (-arousal - tau_a))

    # High arousal intensity (panic channel)
    h = sigmoid(k_a * (arousal - tau_h))

    # Channel urgencies
    u_dep = (n ** p) * (ell ** q)
    u_panic = (n ** p) * (h ** r)

    # Combined urgency (weighted sum)
    u_total = gamma_dep * u_dep + gamma_panic * u_panic

    # Clamp to [0, 1] in case gamma_dep + gamma_panic > 1
    u_total = min(1.0, u_total)

    return u_total, u_dep, u_panic


def compute_urgency_v2(
    arousal: float,
    valence: float,
    config: Optional[WeightConfig] = None
) -> float:
    """
    Unified urgency computation with configurable mode.

    Args:
        arousal: Arousal value in [-1, 1]
        valence: Valence value in [-1, 1]
        config: Weight configuration (default: SOFT mode)

    Returns:
        Urgency score u in [0, 1]
    """
    if config is None:
        config = WeightConfig()

    if config.mode == WeightMode.HARD:
        return compute_urgency_hard(arousal, valence, config.p, config.q)

    elif config.mode == WeightMode.SOFT:
        return compute_urgency_soft(
            arousal, valence,
            p=config.p, q=config.q,
            k_v=config.k_v, k_a=config.k_a,
            tau_v=config.tau_v, tau_a=config.tau_a
        )

    elif config.mode == WeightMode.DUAL_CHANNEL:
        u_total, _, _ = compute_urgency_dual_channel(
            arousal, valence,
            p=config.p, q=config.q, r=config.r,
            k_v=config.k_v, k_a=config.k_a,
            tau_v=config.tau_v, tau_a=config.tau_a, tau_h=config.tau_h,
            gamma_dep=config.gamma_dep, gamma_panic=config.gamma_panic
        )
        return u_total

    else:
        raise ValueError(f"Unknown weight mode: {config.mode}")


def affect_weight_v2(
    arousal: float,
    valence: float,
    confidence: float = 1.0,
    config: Optional[WeightConfig] = None
) -> float:
    """
    Compute affect-based scheduling weight (v2.0).

    w = 1 + (w_max - 1) * confidence * urgency

    Args:
        arousal: Arousal value in [-1, 1]
        valence: Valence value in [-1, 1]
        confidence: Emotion recognition confidence in [0, 1]
        config: Weight configuration

    Returns:
        Scheduling weight w in [1, w_max]
    """
    if config is None:
        config = WeightConfig()

    u = compute_urgency_v2(arousal, valence, config)
    w = 1.0 + (config.w_max - 1.0) * confidence * u

    return w


def compute_wspt_score_v2(
    predicted_service_time: float,
    arousal: float,
    valence: float,
    confidence: float = 1.0,
    config: Optional[WeightConfig] = None
) -> float:
    """
    Compute WSPT score with v2 weight computation.

    Score = S / w (lower = higher priority)
    """
    w = affect_weight_v2(arousal, valence, confidence, config)

    if w <= 0:
        w = 1.0

    return predicted_service_time / w


def get_detailed_weight_info(
    arousal: float,
    valence: float,
    confidence: float = 1.0,
    config: Optional[WeightConfig] = None
) -> Dict:
    """
    Get detailed weight computation breakdown for debugging/logging.

    Returns:
        Dictionary with all intermediate values
    """
    if config is None:
        config = WeightConfig()

    result = {
        'arousal': arousal,
        'valence': valence,
        'confidence': confidence,
        'mode': config.mode.value,
        'w_max': config.w_max,
    }

    if config.mode == WeightMode.HARD:
        n = max(0.0, -valence)
        ell = max(0.0, -arousal)
        u = (n ** config.p) * (ell ** config.q)
        result.update({
            'n': n,
            'ell': ell,
            'urgency': u,
            'weight': 1.0 + (config.w_max - 1.0) * confidence * u
        })

    elif config.mode == WeightMode.SOFT:
        n = sigmoid(config.k_v * (-valence - config.tau_v))
        ell = sigmoid(config.k_a * (-arousal - config.tau_a))
        u = (n ** config.p) * (ell ** config.q)
        result.update({
            'n': n,
            'ell': ell,
            'k_v': config.k_v,
            'k_a': config.k_a,
            'tau_v': config.tau_v,
            'tau_a': config.tau_a,
            'urgency': u,
            'weight': 1.0 + (config.w_max - 1.0) * confidence * u
        })

    elif config.mode == WeightMode.DUAL_CHANNEL:
        u_total, u_dep, u_panic = compute_urgency_dual_channel(
            arousal, valence,
            p=config.p, q=config.q, r=config.r,
            k_v=config.k_v, k_a=config.k_a,
            tau_v=config.tau_v, tau_a=config.tau_a, tau_h=config.tau_h,
            gamma_dep=config.gamma_dep, gamma_panic=config.gamma_panic
        )
        result.update({
            'urgency_depression': u_dep,
            'urgency_panic': u_panic,
            'gamma_dep': config.gamma_dep,
            'gamma_panic': config.gamma_panic,
            'urgency_total': u_total,
            'weight': 1.0 + (config.w_max - 1.0) * confidence * u_total
        })

    return result


# Convenience functions for backward compatibility

def compute_urgency(
    arousal: float,
    valence: float,
    p: float = 1.0,
    q: float = 1.0
) -> float:
    """Backward compatible: original hard gating."""
    return compute_urgency_hard(arousal, valence, p, q)


def affect_weight(
    arousal: float,
    valence: float,
    confidence: float = 1.0,
    w_max: float = 2.0,
    p: float = 1.0,
    q: float = 1.0
) -> float:
    """Backward compatible: original hard gating weight."""
    u = compute_urgency_hard(arousal, valence, p, q)
    return 1.0 + (w_max - 1.0) * confidence * u


# =============================================================================
# Preset Configurations
# =============================================================================
#
# Four preset configurations designed for different use cases:
#
# 1. depression_first_hard (v1.0 Compatible Baseline)
#    - Uses hard gating max(0, -x), identical to original v1.0 implementation
#    - Only Depression quadrant (valence<0 AND arousal<0) gets priority
#    - Use case: Ablation studies comparing against original implementation
#    - Drawback: Discontinuous at 0, sensitive to noise
#
# 2. depression_first_soft (Recommended Default)
#    - Uses sigmoid soft gating, continuous and differentiable everywhere
#    - Still Depression-First strategy, but with smoother transitions
#    - k=5.0 provides steep enough transition while maintaining continuity
#    - Use case: Production deployment, experiments requiring stability
#    - Advantage: More robust to noise in emotion predictions
#
# 3. dual_channel_balanced (Extended Version)
#    - Considers both Depression and Panic quadrants
#    - Depression (gamma=1.0): Full priority boost
#    - Panic (gamma=0.3): Limited priority boost (~30%)
#    - Use case: Emotional support systems that need to respond to anxiety/panic
#    - Rationale: Panic users may also need faster response, but not equal to Depression
#
# 4. dual_channel_depression_heavy (Depression Heavy Priority)
#    - Strongly emphasizes Depression, minimal Panic priority
#    - Depression (gamma=1.0): Full priority boost
#    - Panic (gamma=0.15): Minimal priority boost (~15%)
#    - w_max=2.5: Larger maximum weight for stronger "queue jumping"
#    - Use case: Strict Depression-First research, maximizing Depression speedup
#

PRESET_CONFIGS = {
    # --------------------------------------------------------------------------
    # depression_first_hard: Original hard gating version (v1.0 compatible)
    # --------------------------------------------------------------------------
    # Characteristics:
    #   - Uses max(0, -x) hard gating
    #   - Panic quadrant weight = 1.0 (no boost)
    #   - Discontinuous at valence=0 or arousal=0
    # Use cases:
    #   - Comparison with original paper/implementation
    #   - Ablation experiment baseline
    # Example effects:
    #   - (a=-0.8, v=-0.8) -> u=0.64, w=1.64
    #   - (a=0.8, v=-0.8)  -> u=0.0,  w=1.0 (Panic gets no boost)
    #   - (a=-0.1, v=-0.1) -> u=0.01, w=1.01 (edge case)
    # --------------------------------------------------------------------------
    'depression_first_hard': WeightConfig(
        mode=WeightMode.HARD,
        w_max=2.0,
        p=1.0,
        q=1.0
    ),

    # --------------------------------------------------------------------------
    # depression_first_soft: Soft gating version (recommended default)
    # --------------------------------------------------------------------------
    # Characteristics:
    #   - Uses sigmoid soft gating, continuous and differentiable everywhere
    #   - Still Depression-First strategy
    #   - k=5.0 provides steep transition (most change occurs within +/-0.4)
    # Use cases:
    #   - Production environment deployment
    #   - Experiments requiring noise robustness
    # Example effects:
    #   - (a=-0.8, v=-0.8) -> u~0.94, w~1.94 (higher due to sigmoid saturation)
    #   - (a=0.8, v=-0.8)  -> u~0.02, w~1.02 (Panic gets slight boost)
    #   - (a=-0.1, v=-0.1) -> u~0.41, w~1.41 (smoother edge case)
    # Parameter explanation:
    #   - k_v=5.0: sigmoid steepness for valence
    #   - k_a=5.0: sigmoid steepness for arousal
    #   - Larger k -> closer to hard gating but maintains continuity
    # --------------------------------------------------------------------------
    'depression_first_soft': WeightConfig(
        mode=WeightMode.SOFT,
        w_max=2.0,
        p=1.0,
        q=1.0,
        k_v=5.0,
        k_a=5.0
    ),

    # --------------------------------------------------------------------------
    # dual_channel_balanced: Dual-channel balanced version
    # --------------------------------------------------------------------------
    # Characteristics:
    #   - Depression channel: gamma_dep=1.0 (full priority)
    #   - Panic channel: gamma_panic=0.3 (30% priority boost)
    #   - Layered response to negative emotions
    # Design rationale:
    #   - Depression users need highest priority (potential crisis tendency)
    #   - Panic users may also need faster response (anxiety/panic state)
    #   - But Panic should not get equal priority to Depression
    # Use cases:
    #   - Emotional support chat systems
    #   - Mental health applications
    #   - Scenarios requiring broader emotional response
    # Example effects:
    #   - (a=-0.8, v=-0.8) -> u_dep~0.94, u_panic~0.02, u~0.95, w~1.95
    #   - (a=0.8, v=-0.8)  -> u_dep~0.02, u_panic~0.94, u~0.30, w~1.30
    #   - Panic users now get some priority boost (but much less than Depression)
    # --------------------------------------------------------------------------
    'dual_channel_balanced': WeightConfig(
        mode=WeightMode.DUAL_CHANNEL,
        w_max=2.0,
        p=1.0,
        q=1.0,
        r=1.0,
        k_v=5.0,
        k_a=5.0,
        gamma_dep=1.0,
        gamma_panic=0.3
    ),

    # --------------------------------------------------------------------------
    # dual_channel_depression_heavy: Depression heavy priority version
    # --------------------------------------------------------------------------
    # Characteristics:
    #   - Depression channel: gamma_dep=1.0 (full priority)
    #   - Panic channel: gamma_panic=0.15 (only 15% priority boost)
    #   - w_max=2.5: Larger maximum weight
    # Design rationale:
    #   - Strict Depression-First strategy
    #   - Panic gets token boost rather than being completely ignored
    #   - Larger w_max gives Depression stronger "queue jumping" ability
    # Use cases:
    #   - Research strictly emphasizing Depression priority
    #   - Scenarios requiring maximum Depression speedup
    # Example effects:
    #   - (a=-0.8, v=-0.8) -> w~2.41 (stronger queue jumping)
    #   - (a=0.8, v=-0.8)  -> w~1.21 (Panic gets minimal boost)
    # --------------------------------------------------------------------------
    'dual_channel_depression_heavy': WeightConfig(
        mode=WeightMode.DUAL_CHANNEL,
        w_max=2.5,
        p=1.0,
        q=1.0,
        r=1.0,
        k_v=5.0,
        k_a=5.0,
        gamma_dep=1.0,
        gamma_panic=0.15
    ),
}


def get_preset_config(name: str) -> WeightConfig:
    """
    Get a preset configuration by name.

    Available presets:
        - 'depression_first_hard': Original hard gating (v1.0 compatible)
        - 'depression_first_soft': Soft gating (recommended default)
        - 'dual_channel_balanced': Depression + Panic balanced response
        - 'dual_channel_depression_heavy': Depression heavy priority

    Args:
        name: Preset configuration name

    Returns:
        WeightConfig instance

    Raises:
        ValueError: If preset name is not found
    """
    if name not in PRESET_CONFIGS:
        available = list(PRESET_CONFIGS.keys())
        raise ValueError(f"Unknown preset: {name}. Available: {available}")
    return PRESET_CONFIGS[name]


def list_presets() -> Dict[str, str]:
    """
    List all available preset configurations with descriptions.

    Returns:
        Dictionary mapping preset name to description
    """
    return {
        'depression_first_hard': 'Original hard gating (v1.0 compatible), '
                                  'only Depression quadrant gets priority',
        'depression_first_soft': 'Soft sigmoid gating (recommended), '
                                  'continuous and noise-robust',
        'dual_channel_balanced': 'Depression + Panic dual channel, '
                                  'gamma_dep=1.0, gamma_panic=0.3',
        'dual_channel_depression_heavy': 'Depression heavy priority, '
                                          'w_max=2.5, gamma_panic=0.15',
    }
