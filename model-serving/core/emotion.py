"""
Emotion Module for Emotion-aware LLM Scheduling

This module provides emotion sampling functionality based on the EmpatheticDialogues
dataset, which contains 32 emotion categories. Each emotion is mapped to an arousal
value in the range [-1, 1], where:
  - High arousal (near 1.0): excited, angry, terrified
  - Neutral arousal (near 0.0): content, nostalgic
  - Low arousal (near -1.0): sad, lonely, devastated

References:
- EmpatheticDialogues: https://arxiv.org/abs/1811.00207
- Russell's Circumplex Model of Affect
"""

import random
import numpy as np
from typing import Tuple, Dict, List


# Emotion-to-Arousal mapping based on EmpatheticDialogues dataset (32 emotions)
# Arousal values are normalized to [-1, 1] range based on psychological literature
# Higher values indicate higher emotional activation/arousal
EMOTION_AROUSAL_MAP = {
    # High Arousal Positive (0.6 to 1.0)
    'excited': 0.95,
    'joyful': 0.85,
    'surprised': 0.80,
    'anticipating': 0.75,
    'impressed': 0.70,
    'proud': 0.65,

    # High Arousal Negative (0.6 to 1.0)
    'terrified': 0.95,
    'afraid': 0.90,
    'anxious': 0.85,
    'angry': 0.90,
    'furious': 0.95,
    'annoyed': 0.70,
    'disgusted': 0.75,

    # Medium-High Arousal (0.3 to 0.6)
    'hopeful': 0.55,
    'trusting': 0.45,
    'faithful': 0.45,
    'caring': 0.40,
    'grateful': 0.50,
    'confident': 0.55,
    'jealous': 0.60,
    'embarrassed': 0.65,

    # Medium-Low Arousal (-0.3 to 0.3)
    'sentimental': 0.20,
    'nostalgic': 0.10,
    'content': 0.05,
    'prepared': 0.15,
    'apprehensive': 0.25,
    'guilty': 0.20,
    'ashamed': 0.30,

    # Low Arousal Negative (-1.0 to -0.3)
    'sad': -0.60,
    'lonely': -0.70,
    'disappointed': -0.50,
    'devastated': -0.65,
}

# Default emotion probability distribution (uniform for simplicity)
# Can be customized based on actual EmpatheticDialogues dataset statistics
DEFAULT_EMOTION_PROBS = {emotion: 1.0/len(EMOTION_AROUSAL_MAP)
                         for emotion in EMOTION_AROUSAL_MAP.keys()}


class EmotionConfig:
    """Configuration for emotion sampling and arousal mapping"""

    def __init__(self,
                 emotion_arousal_map: Dict[str, float] = None,
                 emotion_probs: Dict[str, float] = None,
                 arousal_noise_std: float = 0.0):
        """
        Initialize emotion configuration

        Args:
            emotion_arousal_map: Dictionary mapping emotion labels to arousal values
            emotion_probs: Dictionary mapping emotion labels to sampling probabilities
            arousal_noise_std: Standard deviation of Gaussian noise added to arousal
                              (0.0 = no noise, 0.1 = small variation)
        """
        self.emotion_arousal_map = emotion_arousal_map or EMOTION_AROUSAL_MAP
        self.emotion_probs = emotion_probs or DEFAULT_EMOTION_PROBS
        self.arousal_noise_std = arousal_noise_std

        # Validate probabilities sum to 1.0 (with tolerance)
        prob_sum = sum(self.emotion_probs.values())
        if abs(prob_sum - 1.0) > 1e-6:
            # Normalize probabilities
            self.emotion_probs = {k: v/prob_sum for k, v in self.emotion_probs.items()}

        # Validate all emotions in probs have arousal values
        for emotion in self.emotion_probs.keys():
            if emotion not in self.emotion_arousal_map:
                raise ValueError(f"Emotion '{emotion}' in probabilities not found in arousal map")

    def get_emotions(self) -> List[str]:
        """Return list of all emotion labels"""
        return list(self.emotion_arousal_map.keys())

    def get_arousal(self, emotion: str, add_noise: bool = True) -> float:
        """
        Get arousal value for a given emotion

        Args:
            emotion: Emotion label
            add_noise: Whether to add Gaussian noise to arousal value

        Returns:
            Arousal value (typically in [-1, 1] range)
        """
        if emotion not in self.emotion_arousal_map:
            raise ValueError(f"Unknown emotion: {emotion}")

        arousal = self.emotion_arousal_map[emotion]

        if add_noise and self.arousal_noise_std > 0:
            noise = np.random.normal(0, self.arousal_noise_std)
            arousal += noise
            # Clip to reasonable range
            arousal = np.clip(arousal, -1.5, 1.5)

        return arousal

    def classify_arousal(self, arousal: float) -> str:
        """
        Classify arousal value into categories (high/medium/low)

        Args:
            arousal: Arousal value

        Returns:
            Category label: 'high', 'medium', or 'low'
        """
        if arousal >= 0.6:
            return 'high'
        elif arousal >= -0.3:
            return 'medium'
        else:
            return 'low'


def sample_emotion(config: EmotionConfig = None) -> Tuple[str, float]:
    """
    Sample a random emotion and return its corresponding arousal value

    Args:
        config: EmotionConfig object (uses default if None)

    Returns:
        Tuple of (emotion_label, arousal_value)

    Example:
        >>> emotion, arousal = sample_emotion()
        >>> print(f"Emotion: {emotion}, Arousal: {arousal:.2f}")
        Emotion: excited, Arousal: 0.95
    """
    if config is None:
        config = EmotionConfig()

    # Sample emotion according to probability distribution
    emotions = list(config.emotion_probs.keys())
    probs = list(config.emotion_probs.values())

    emotion_label = random.choices(emotions, weights=probs, k=1)[0]
    arousal_value = config.get_arousal(emotion_label, add_noise=True)

    return emotion_label, arousal_value


def sample_emotions_batch(n: int, config: EmotionConfig = None) -> List[Tuple[str, float]]:
    """
    Sample multiple emotions at once

    Args:
        n: Number of emotions to sample
        config: EmotionConfig object (uses default if None)

    Returns:
        List of (emotion_label, arousal_value) tuples
    """
    return [sample_emotion(config) for _ in range(n)]


def get_emotion_statistics(config: EmotionConfig = None) -> Dict:
    """
    Get statistics about emotion and arousal distributions

    Args:
        config: EmotionConfig object (uses default if None)

    Returns:
        Dictionary with statistics
    """
    if config is None:
        config = EmotionConfig()

    arousal_values = list(config.emotion_arousal_map.values())

    # Calculate weighted statistics based on emotion probabilities
    weighted_arousal = sum(config.emotion_arousal_map[e] * config.emotion_probs[e]
                          for e in config.emotion_probs.keys())

    # Count emotions by arousal category
    high_count = sum(1 for a in arousal_values if a >= 0.6)
    medium_count = sum(1 for a in arousal_values if -0.3 <= a < 0.6)
    low_count = sum(1 for a in arousal_values if a < -0.3)

    return {
        'num_emotions': len(config.emotion_arousal_map),
        'arousal_mean': np.mean(arousal_values),
        'arousal_std': np.std(arousal_values),
        'arousal_min': min(arousal_values),
        'arousal_max': max(arousal_values),
        'weighted_arousal_mean': weighted_arousal,
        'high_arousal_count': high_count,
        'medium_arousal_count': medium_count,
        'low_arousal_count': low_count,
    }


# Example usage and testing
if __name__ == '__main__':
    print("=" * 60)
    print("Emotion Module Test")
    print("=" * 60)

    # Create default config
    config = EmotionConfig()

    # Print emotion statistics
    stats = get_emotion_statistics(config)
    print(f"\nEmotion Statistics:")
    print(f"  Total emotions: {stats['num_emotions']}")
    print(f"  Arousal range: [{stats['arousal_min']:.2f}, {stats['arousal_max']:.2f}]")
    print(f"  Arousal mean: {stats['arousal_mean']:.2f}")
    print(f"  Arousal std: {stats['arousal_std']:.2f}")
    print(f"  High arousal emotions: {stats['high_arousal_count']}")
    print(f"  Medium arousal emotions: {stats['medium_arousal_count']}")
    print(f"  Low arousal emotions: {stats['low_arousal_count']}")

    # Sample some emotions
    print(f"\nSample Emotions (n=10):")
    samples = sample_emotions_batch(10, config)
    for i, (emotion, arousal) in enumerate(samples, 1):
        category = config.classify_arousal(arousal)
        print(f"  {i}. {emotion:15s} | Arousal: {arousal:5.2f} | Category: {category}")

    # Test with noise
    print(f"\nTesting arousal with noise (std=0.1):")
    config_noisy = EmotionConfig(arousal_noise_std=0.1)
    print(f"  Base arousal for 'excited': {EMOTION_AROUSAL_MAP['excited']:.2f}")
    for i in range(5):
        arousal = config_noisy.get_arousal('excited', add_noise=True)
        print(f"    Sample {i+1}: {arousal:.3f}")

    print("\n" + "=" * 60)
