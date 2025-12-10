"""
Emotion Module for Emotion-aware LLM Scheduling

This module provides emotion sampling functionality based on the EmpatheticDialogues
dataset, which contains 32 emotion categories. Each emotion is mapped to:
  - Valence: pleasant-unpleasant dimension [-1, 1]
  - Arousal: activation level [-1, 1]
  - Dominance: control dimension [-1, 1]

The values are extracted from NRC-VAD-Lexicon v2.1 (Mohammad, 2018).

Russell's Circumplex Model Quadrants:
  - Excited: V >= 0, A >= 0 (high valence, high arousal)
  - Calm: V >= 0, A < 0 (high valence, low arousal)
  - Panic: V < 0, A >= 0 (low valence, high arousal)
  - Depression: V < 0, A < 0 (low valence, low arousal)

References:
  - Russell, J. A. (1980). A circumplex model of affect.
  - Mohammad, S. M. (2018). NRC-VAD-Lexicon. ACL 2018.
  - EmpatheticDialogues: https://arxiv.org/abs/1811.00207
"""

import random
import numpy as np
from typing import Tuple, Dict, List


# ============================================================================
# NRC-VAD-Lexicon Values (extracted from tools/emotion_vad_values.csv)
# Values are in range [-1, 1] for Valence and Arousal
# ============================================================================

EMOTION_VALENCE_MAP = {
    'excited': 0.816,
    'joyful': 0.99,
    'proud': 0.812,
    'hopeful': 0.894,
    'trusting': 0.714,
    'faithful': 0.776,
    'grateful': 0.916,
    'confident': 0.53,
    'content': 0.528,
    'sentimental': 0.166,
    'nostalgic': -0.084,
    'prepared': 0.42,
    'impressed': 0.612,
    'caring': 0.27,
    'surprised': 0.568,
    'anticipating': 0.396,
    'terrified': -0.82,
    'afraid': -0.977,
    'anxious': -0.438,
    'angry': -0.756,
    'furious': -0.876,
    'annoyed': -0.792,
    'disgusted': -0.898,
    'jealous': -0.654,
    'embarrassed': -0.632,
    'apprehensive': -0.188,
    'guilty': -0.73,
    'ashamed': -0.688,
    'sad': -0.55,
    'lonely': -0.5,
    'disappointed': -0.858,
    'devastated': -0.76,
}

EMOTION_AROUSAL_MAP = {
    'excited': 0.862,
    'joyful': 0.24,
    'proud': 0.4,
    'hopeful': -0.286,
    'trusting': 0.016,
    'faithful': -0.334,
    'grateful': -0.294,
    'confident': -0.352,
    'content': -0.408,
    'sentimental': -0.244,
    'nostalgic': -0.298,
    'prepared': -0.04,
    'impressed': 0.54,
    'caring': -0.062,
    'surprised': 0.71,
    'anticipating': 0.078,
    'terrified': 0.804,
    'afraid': 0.352,
    'anxious': 0.75,
    'angry': 0.66,
    'furious': 0.906,
    'annoyed': 0.566,
    'disgusted': 0.546,
    'jealous': 0.71,
    'embarrassed': 0.12,
    'apprehensive': 0.184,
    'guilty': 0.54,
    'ashamed': 0.176,
    'sad': -0.334,
    'lonely': -0.548,
    'disappointed': -0.056,
    'devastated': 0.088,
}

EMOTION_DOMINANCE_MAP = {
    'excited': 0.418,
    'joyful': 0.296,
    'proud': 0.746,
    'hopeful': 0.254,
    'trusting': 0.5,
    'faithful': 0.254,
    'grateful': 0.12,
    'confident': 0.446,
    'content': 0.118,
    'sentimental': -0.376,
    'nostalgic': -0.632,
    'prepared': 0.466,
    'impressed': -0.142,
    'caring': 0.0,
    'surprised': 0.078,
    'anticipating': 0.422,
    'terrified': -0.226,
    'afraid': -0.529,
    'anxious': -0.132,
    'angry': 0.208,
    'furious': 0.196,
    'annoyed': -0.31,
    'disgusted': -0.452,
    'jealous': -0.31,
    'embarrassed': -0.47,
    'apprehensive': -0.138,
    'guilty': -0.296,
    'ashamed': -0.544,
    'sad': -0.702,
    'lonely': -0.524,
    'disappointed': -0.518,
    'devastated': -0.472,
}

# ============================================================================
# Russell Quadrant Mapping
# Based on valence >= 0 and arousal >= 0 thresholds
# ============================================================================

RUSSELL_QUADRANT_MAP = {
    # V >= 0, A >= 0: High valence, high arousal (excited, happy)
    'excited': ['excited', 'impressed', 'surprised', 'joyful', 'proud', 'trusting', 'anticipating'],
    # V >= 0, A < 0: High valence, low arousal (calm, relaxed)
    'calm': ['hopeful', 'faithful', 'grateful', 'confident', 'content', 'sentimental', 'prepared', 'caring'],
    # V < 0, A >= 0: Low valence, high arousal (panic, fear)
    'panic': ['terrified', 'afraid', 'anxious', 'angry', 'furious', 'annoyed', 'disgusted',
              'jealous', 'embarrassed', 'apprehensive', 'guilty', 'ashamed', 'devastated'],
    # V < 0, A < 0: Low valence, low arousal (depression, sadness)
    'depression': ['nostalgic', 'sad', 'lonely', 'disappointed'],
}

# Build reverse mapping: emotion -> quadrant
_EMOTION_TO_QUADRANT = {}
for quadrant, emotions in RUSSELL_QUADRANT_MAP.items():
    for emotion in emotions:
        _EMOTION_TO_QUADRANT[emotion] = quadrant

# All quadrant names
RUSSELL_QUADRANTS = ['excited', 'calm', 'panic', 'depression']

# Default emotion probability distribution (uniform)
DEFAULT_EMOTION_PROBS = {emotion: 1.0/len(EMOTION_AROUSAL_MAP)
                         for emotion in EMOTION_AROUSAL_MAP.keys()}


# ============================================================================
# EmotionConfig Class
# ============================================================================

class EmotionConfig:
    """Configuration for emotion sampling and VAD mapping"""

    def __init__(self,
                 emotion_arousal_map: Dict[str, float] = None,
                 emotion_valence_map: Dict[str, float] = None,
                 emotion_dominance_map: Dict[str, float] = None,
                 emotion_probs: Dict[str, float] = None,
                 arousal_noise_std: float = 0.0,
                 valence_noise_std: float = 0.0):
        """
        Initialize emotion configuration

        Args:
            emotion_arousal_map: Dictionary mapping emotion labels to arousal values
            emotion_valence_map: Dictionary mapping emotion labels to valence values
            emotion_dominance_map: Dictionary mapping emotion labels to dominance values
            emotion_probs: Dictionary mapping emotion labels to sampling probabilities
            arousal_noise_std: Standard deviation of Gaussian noise for arousal
            valence_noise_std: Standard deviation of Gaussian noise for valence
        """
        self.emotion_arousal_map = emotion_arousal_map or EMOTION_AROUSAL_MAP
        self.emotion_valence_map = emotion_valence_map or EMOTION_VALENCE_MAP
        self.emotion_dominance_map = emotion_dominance_map or EMOTION_DOMINANCE_MAP
        self.emotion_probs = emotion_probs or DEFAULT_EMOTION_PROBS
        self.arousal_noise_std = arousal_noise_std
        self.valence_noise_std = valence_noise_std

        # Validate probabilities sum to 1.0
        prob_sum = sum(self.emotion_probs.values())
        if abs(prob_sum - 1.0) > 1e-6:
            self.emotion_probs = {k: v/prob_sum for k, v in self.emotion_probs.items()}

        # Validate all emotions have required values
        for emotion in self.emotion_probs.keys():
            if emotion not in self.emotion_arousal_map:
                raise ValueError(f"Emotion '{emotion}' not found in arousal map")
            if emotion not in self.emotion_valence_map:
                raise ValueError(f"Emotion '{emotion}' not found in valence map")

    def get_emotions(self) -> List[str]:
        """Return list of all emotion labels"""
        return list(self.emotion_arousal_map.keys())

    def get_arousal(self, emotion: str, add_noise: bool = True) -> float:
        """
        Get arousal value for a given emotion

        Args:
            emotion: Emotion label
            add_noise: Whether to add Gaussian noise

        Returns:
            Arousal value in [-1, 1]
        """
        if emotion not in self.emotion_arousal_map:
            raise ValueError(f"Unknown emotion: {emotion}")

        arousal = self.emotion_arousal_map[emotion]

        if add_noise and self.arousal_noise_std > 0:
            noise = np.random.normal(0, self.arousal_noise_std)
            arousal = np.clip(arousal + noise, -1.0, 1.0)

        return arousal

    def get_valence(self, emotion: str, add_noise: bool = True) -> float:
        """
        Get valence value for a given emotion

        Args:
            emotion: Emotion label
            add_noise: Whether to add Gaussian noise

        Returns:
            Valence value in [-1, 1]
        """
        if emotion not in self.emotion_valence_map:
            raise ValueError(f"Unknown emotion: {emotion}")

        valence = self.emotion_valence_map[emotion]

        if add_noise and self.valence_noise_std > 0:
            noise = np.random.normal(0, self.valence_noise_std)
            valence = np.clip(valence + noise, -1.0, 1.0)

        return valence

    def get_dominance(self, emotion: str) -> float:
        """Get dominance value for a given emotion"""
        if emotion not in self.emotion_dominance_map:
            raise ValueError(f"Unknown emotion: {emotion}")
        return self.emotion_dominance_map[emotion]

    def get_quadrant(self, emotion: str) -> str:
        """
        Get Russell quadrant for a given emotion.

        Args:
            emotion: Emotion label

        Returns:
            Quadrant name: 'excited', 'calm', 'panic', or 'depression'
        """
        if emotion not in _EMOTION_TO_QUADRANT:
            raise ValueError(f"Unknown emotion: {emotion}")
        return _EMOTION_TO_QUADRANT[emotion]

    def classify_russell_quadrant(self, arousal: float, valence: float) -> str:
        """
        Classify arousal and valence values into Russell quadrant.

        Args:
            arousal: Arousal value
            valence: Valence value

        Returns:
            Quadrant name: 'excited', 'calm', 'panic', or 'depression'
        """
        if valence >= 0:
            return 'excited' if arousal >= 0 else 'calm'
        else:
            return 'panic' if arousal >= 0 else 'depression'

    def get_quadrants(self) -> List[str]:
        """Return list of all Russell quadrant names"""
        return RUSSELL_QUADRANTS.copy()

    def get_emotions_in_quadrant(self, quadrant: str) -> List[str]:
        """
        Get list of emotions in a specific quadrant.

        Args:
            quadrant: Quadrant name ('excited', 'calm', 'panic', 'depression')

        Returns:
            List of emotion labels in that quadrant
        """
        if quadrant not in RUSSELL_QUADRANT_MAP:
            raise ValueError(f"Unknown quadrant: {quadrant}")
        return RUSSELL_QUADRANT_MAP[quadrant].copy()


# ============================================================================
# Sampling Functions
# ============================================================================

def sample_emotion(config: EmotionConfig = None) -> Tuple[str, float, float]:
    """
    Sample a random emotion and return its VAD values.

    Args:
        config: EmotionConfig object (uses default if None)

    Returns:
        Tuple of (emotion_label, arousal_value, valence_value)
    """
    if config is None:
        config = EmotionConfig()

    emotions = list(config.emotion_probs.keys())
    probs = list(config.emotion_probs.values())

    emotion_label = random.choices(emotions, weights=probs, k=1)[0]
    arousal = config.get_arousal(emotion_label, add_noise=True)
    valence = config.get_valence(emotion_label, add_noise=True)

    return emotion_label, arousal, valence


def sample_emotions_batch(n: int, config: EmotionConfig = None) -> List[Tuple[str, float, float]]:
    """
    Sample multiple emotions at once.

    Args:
        n: Number of emotions to sample
        config: EmotionConfig object

    Returns:
        List of (emotion_label, arousal, valence) tuples
    """
    return [sample_emotion(config) for _ in range(n)]


def sample_emotions_batch_stratified_quadrant(
    num_jobs: int,
    emotion_config: EmotionConfig,
    quadrant_distribution: Dict[str, float] = None
) -> List[Tuple[str, float, float, str]]:
    """
    Batch sample emotions with stratified sampling by Russell quadrant.

    Ensures balanced distribution across the four Russell quadrants
    for fair comparison across emotional states.

    Args:
        num_jobs: Number of jobs to generate
        emotion_config: EmotionConfig object
        quadrant_distribution: Target distribution for each quadrant
                              e.g., {'excited': 0.25, 'calm': 0.25, 'panic': 0.25, 'depression': 0.25}
                              If None, uses uniform distribution (1/4 each)

    Returns:
        List of (emotion_label, arousal, valence, quadrant) tuples
    """
    quadrants = RUSSELL_QUADRANTS

    if quadrant_distribution is None or quadrant_distribution == 'uniform':
        quadrant_distribution = {q: 1/4 for q in quadrants}
    else:
        # Auto-fill missing quadrants
        specified_total = sum(quadrant_distribution.get(q, 0) for q in quadrants)
        unspecified = [q for q in quadrants if q not in quadrant_distribution]
        if unspecified and specified_total < 1:
            remaining = (1 - specified_total) / len(unspecified)
            quadrant_distribution = {q: quadrant_distribution.get(q, remaining) for q in quadrants}

    # Calculate count for each quadrant using Largest Remainder Method
    quadrant_counts = {}
    remainders = {}

    total_allocated = 0
    for q in quadrants:
        expected = num_jobs * quadrant_distribution.get(q, 0)
        floor_val = int(expected)
        quadrant_counts[q] = floor_val
        remainders[q] = expected - floor_val
        total_allocated += floor_val

    # Distribute remaining slots
    remaining = num_jobs - total_allocated
    if remaining > 0:
        sorted_quadrants = sorted(remainders.keys(), key=lambda x: remainders[x], reverse=True)
        for i in range(remaining):
            quadrant_counts[sorted_quadrants[i]] += 1

    # Sample from each quadrant
    emotions_data = []

    for quadrant, count in quadrant_counts.items():
        available_emotions = RUSSELL_QUADRANT_MAP.get(quadrant, [])

        if not available_emotions:
            continue

        for _ in range(count):
            emotion = available_emotions[np.random.randint(len(available_emotions))]
            arousal = emotion_config.get_arousal(emotion, add_noise=True)
            valence = emotion_config.get_valence(emotion, add_noise=True)
            emotions_data.append((emotion, arousal, valence, quadrant))

    # Shuffle to randomize order
    np.random.shuffle(emotions_data)

    return emotions_data


def get_emotion_statistics(config: EmotionConfig = None) -> Dict:
    """
    Get statistics about emotion distributions.

    Args:
        config: EmotionConfig object (uses default if None)

    Returns:
        Dictionary with statistics
    """
    if config is None:
        config = EmotionConfig()

    arousal_values = list(config.emotion_arousal_map.values())
    valence_values = list(config.emotion_valence_map.values())

    # Count by quadrant
    quadrant_counts = {q: len(emotions) for q, emotions in RUSSELL_QUADRANT_MAP.items()}

    return {
        'num_emotions': len(config.emotion_arousal_map),
        'arousal_mean': np.mean(arousal_values),
        'arousal_std': np.std(arousal_values),
        'arousal_min': min(arousal_values),
        'arousal_max': max(arousal_values),
        'valence_mean': np.mean(valence_values),
        'valence_std': np.std(valence_values),
        'valence_min': min(valence_values),
        'valence_max': max(valence_values),
        'quadrant_counts': quadrant_counts,
    }


# ============================================================================
# Utility Functions (kept for backward compatibility)
# ============================================================================

def classify_arousal(arousal: float) -> str:
    """
    Classify arousal value into categories (high/medium/low).
    DEPRECATED: Use classify_russell_quadrant instead.

    Args:
        arousal: Arousal value

    Returns:
        Category label: 'high', 'medium', or 'low'
    """
    if arousal >= 0.3:
        return 'high'
    elif arousal >= -0.3:
        return 'medium'
    else:
        return 'low'


def classify_valence(valence: float) -> str:
    """
    Classify valence into categories (positive/neutral/negative).
    DEPRECATED: Use classify_russell_quadrant instead.

    Args:
        valence: Valence value

    Returns:
        'negative', 'neutral', or 'positive'
    """
    if valence > 0.2:
        return 'positive'
    if valence < -0.2:
        return 'negative'
    return 'neutral'
