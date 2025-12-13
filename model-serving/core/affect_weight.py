"""
Affect Weight Module for Depression-First Scheduling.

This module implements the affect weight calculation for the AW-SSJF
(Affect-Weighted Shortest-Service-Job-First) scheduler.

The core principle is "Depression-First": prioritize users in a depressed
emotional state (low valence + low arousal) by giving them higher scheduling
weights, resulting in lower effective service time scores.

Theoretical Background:
    Based on Russell's Circumplex Model of Affect (1980), emotions are mapped
    to a 2D space: valence (pleasant-unpleasant) and arousal (activation level).

    - Depression/Sadness: Low valence + Low arousal (southwest quadrant)
    - Fear/Panic: Low valence + High arousal (northwest quadrant)
    - Excitement: High valence + High arousal (northeast quadrant)
    - Calm/Relaxed: High valence + Low arousal (southeast quadrant)

Weight Computation:
    1. Urgency u = n^p * ell^q where:
       - n = max(0, -valence)  # Unpleasant intensity [0, 1]
       - ell = max(0, -arousal)  # Low arousal intensity [0, 1]
       - p, q >= 1 control curve shape (1 = linear, 2 = threshold effect)

    2. Weight w = 1 + (w_max - 1) * c * u where:
       - w_max controls maximum "queue jumping" power
       - c is emotion recognition confidence (optional discount)

References:
    - Russell, J. A. (1980). A circumplex model of affect.
    - Smith, W. E. (1956). Various optimizers for single-stage production.
    - Mohammad, S. M. (2018). NRC-VAD-Lexicon. ACL 2018.
"""

from typing import Tuple


def compute_urgency(
    arousal: float,
    valence: float,
    p: float = 1.0,
    q: float = 1.0
) -> float:
    """
    Compute user urgency score based on Depression-First principle.

    Only users with negative valence AND low arousal (depression state)
    receive high urgency scores. This is achieved through multiplication:
    both conditions must be met for a high score.

    Args:
        arousal: Arousal value in [-1, 1] from NRC-VAD-Lexicon (no normalization)
        valence: Valence value in [-1, 1] from NRC-VAD-Lexicon (no normalization)
        p: Exponent for negative valence (>= 1). Higher = more threshold effect.
        q: Exponent for low arousal (>= 1). Higher = more threshold effect.

    Returns:
        Urgency score u in [0, 1]
        - u = 0: Positive emotion or high arousal (no urgency boost)
        - u = 1: Maximum depression state (valence = -1, arousal = -1)

    Examples:
        >>> compute_urgency(arousal=-0.8, valence=-0.8)  # Depressed
        0.64
        >>> compute_urgency(arousal=0.8, valence=-0.8)   # Panicked (high arousal)
        0.0
        >>> compute_urgency(arousal=-0.8, valence=0.8)   # Calm (positive valence)
        0.0
        >>> compute_urgency(arousal=0.8, valence=0.8)    # Excited
        0.0
    """
    # Extract negative parts only (positive emotions get 0)
    n = max(0.0, -valence)    # Unpleasant intensity in [0, 1]
    ell = max(0.0, -arousal)  # Low arousal intensity in [0, 1]

    # Depression urgency = n^p * ell^q
    # Both must be present for high urgency (multiplication)
    u = (n ** p) * (ell ** q)

    return u


def affect_weight(
    arousal: float,
    valence: float,
    confidence: float = 1.0,
    w_max: float = 2.0,
    p: float = 1.0,
    q: float = 1.0
) -> float:
    """
    Compute affect-based scheduling weight (Depression-First + Confidence Discount).

    Higher weight = higher priority in WSPT-style scheduling.
    The weight is used to compute the scheduling score: Score = S / w
    where S is the predicted service time. Lower score = scheduled first.

    Args:
        arousal: Arousal value in [-1, 1]
        valence: Valence value in [-1, 1]
        confidence: Emotion recognition confidence in [0, 1] (default: 1.0)
                   Lower confidence reduces the urgency boost.
        w_max: Maximum weight (controls max "queue jumping" power).
               Recommended range: [1.2, 3.0]. Default: 2.0.
        p: Exponent for negative valence (>= 1)
        q: Exponent for low arousal (>= 1)

    Returns:
        Scheduling weight w >= 1.0
        - w = 1.0: No priority boost (neutral/positive emotions)
        - w = w_max: Maximum priority boost (depression with full confidence)

    Examples:
        >>> affect_weight(arousal=-0.8, valence=-0.8, w_max=2.0)  # Depressed
        1.64  # Significant boost
        >>> affect_weight(arousal=0.8, valence=-0.8, w_max=2.0)   # Panicked
        1.0   # No boost (high arousal)
        >>> affect_weight(arousal=-0.8, valence=0.8, w_max=2.0)   # Calm
        1.0   # No boost (positive valence)
    """
    # Step 1: Compute depression urgency
    u = compute_urgency(arousal, valence, p, q)

    # Step 2: Apply confidence discount and compute weight
    # w = 1 + (w_max - 1) * c * u
    w = 1.0 + (w_max - 1.0) * confidence * u

    return w


def compute_wspt_score(
    predicted_service_time: float,
    arousal: float,
    valence: float,
    confidence: float = 1.0,
    w_max: float = 2.0,
    p: float = 1.0,
    q: float = 1.0
) -> float:
    """
    Compute WSPT (Weighted Shortest Processing Time) score.

    Score = S / w where S is service time and w is affect weight.
    Lower score = higher priority (scheduled first).

    Args:
        predicted_service_time: Predicted service time S (from BERT predictor)
        arousal: Arousal value in [-1, 1]
        valence: Valence value in [-1, 1]
        confidence: Emotion recognition confidence in [0, 1]
        w_max: Maximum weight
        p: Exponent for negative valence
        q: Exponent for low arousal

    Returns:
        WSPT score (lower = higher priority)

    Note:
        This is the core scheduling decision metric for AW-SSJF.
        Jobs are sorted by score in ascending order.
    """
    w = affect_weight(arousal, valence, confidence, w_max, p, q)

    # Avoid division by zero
    if w <= 0:
        w = 1.0

    return predicted_service_time / w


def compute_urgency_and_weight(
    arousal: float,
    valence: float,
    confidence: float = 1.0,
    w_max: float = 2.0,
    p: float = 1.0,
    q: float = 1.0
) -> Tuple[float, float]:
    """
    Compute both urgency and weight in a single call.

    Convenience function that returns both values for logging/analysis.

    Args:
        arousal: Arousal value in [-1, 1]
        valence: Valence value in [-1, 1]
        confidence: Emotion recognition confidence in [0, 1]
        w_max: Maximum weight
        p: Exponent for negative valence
        q: Exponent for low arousal

    Returns:
        Tuple of (urgency, weight)
    """
    u = compute_urgency(arousal, valence, p, q)
    w = 1.0 + (w_max - 1.0) * confidence * u
    return u, w
