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
    'excited': 0.8,
    'joyful': 0.8,
    'surprised': 0.8,
    'anticipating': 0.8,
    'impressed': 0.8,
    'proud': 0.8,

    # High Arousal Negative (0.6 to 1.0)
    'terrified': 0.8,
    'afraid': 0.8,
    'anxious': 0.8,
    'angry': 0.8,
    'furious': 0.8,
    'annoyed': 0.8,
    'disgusted': 0.8,

    # Medium-High Arousal (0.3 to 0.6)
    'hopeful': 0,
    'trusting': 0,
    'faithful': 0,
    'caring': 0,
    'grateful': 0,
    'confident': 0,
    'jealous': 0,
    'embarrassed': 0,

    # Medium-Low Arousal (-0.3 to 0.3)
    'sentimental': -0.8,
    'nostalgic': -0.8,
    'content': -0.8,
    'prepared': -0.8,
    'apprehensive': -0.8,
    'guilty': -0.8,
    'ashamed': -0.8,

    # Low Arousal Negative (-1.0 to -0.3)
    'sad': -0.8,
    'lonely': -0.8,
    'disappointed': -0.8,
    'devastated': -0.8,
}

# Emotion-to-Valence mapping (discrete three-level: -0.8, 0, 0.8)
# Negative valence (e.g., sad/afraid) gets -0.8; neutral gets 0; positive gets 0.8
EMOTION_VALENCE_MAP = {
    # Positive valence
    'excited': 0.8,
    'joyful': 0.8,
    'anticipating': 0.8,
    'proud': 0.8,
    'hopeful': 0.8,
    'trusting': 0.8,
    'faithful': 0.8,
    'grateful': 0.8,
    'confident': 0.8,
    'content': 0.8,

    # Neutral valence
    'sentimental': 0.0,
    'nostalgic': 0.0,
    'prepared': 0.0,
    'impressed': 0.0,
    'caring': 0.0,
    'surprised': 0.0,

    # Negative valence
    'terrified': -0.8,
    'afraid': -0.8,
    'anxious': -0.8,
    'angry': -0.8,
    'furious': -0.8,
    'annoyed': -0.8,
    'disgusted': -0.8,
    'jealous': -0.8,
    'embarrassed': -0.8,
    'apprehensive': -0.8,
    'guilty': -0.8,
    'ashamed': -0.8,
    'sad': -0.8,
    'lonely': -0.8,
    'disappointed': -0.8,
    'devastated': -0.8,
}

# 9-class emotion category map: arousal (high/medium/low) × valence (positive/neutral/negative)
EMOTION_CATEGORY_MAP = {
    'high_positive': ['excited', 'joyful', 'anticipating', 'proud'],
    'high_neutral': ['impressed', 'surprised'],
    'high_negative': ['terrified', 'afraid', 'anxious', 'angry', 'furious', 'annoyed', 'disgusted'],
    'medium_positive': ['hopeful', 'trusting', 'faithful', 'grateful', 'confident'],
    'medium_neutral': ['caring'],
    'medium_negative': ['jealous', 'embarrassed'],
    'low_positive': ['content'],
    'low_neutral': ['sentimental', 'nostalgic', 'prepared'],
    'low_negative': ['apprehensive', 'guilty', 'ashamed', 'sad', 'lonely', 'disappointed', 'devastated'],
}

# All 9 category names
EMOTION_CATEGORIES = list(EMOTION_CATEGORY_MAP.keys())

# Default emotion probability distribution (uniform for simplicity)
# Can be customized based on actual EmpatheticDialogues dataset statistics
DEFAULT_EMOTION_PROBS = {emotion: 1.0/len(EMOTION_AROUSAL_MAP)
                         for emotion in EMOTION_AROUSAL_MAP.keys()}


class EmotionConfig:
    """Configuration for emotion sampling and arousal mapping"""

    def __init__(self,
                 emotion_arousal_map: Dict[str, float] = None,
                 emotion_valence_map: Dict[str, float] = None,
                 emotion_category_map: Dict[str, List[str]] = None,
                 emotion_probs: Dict[str, float] = None,
                 arousal_noise_std: float = 0.0):
        """
        Initialize emotion configuration

        Args:
            emotion_arousal_map: Dictionary mapping emotion labels to arousal values
            emotion_valence_map: Dictionary mapping emotion labels to valence values
            emotion_category_map: Dictionary mapping 9 categories to emotion lists
            emotion_probs: Dictionary mapping emotion labels to sampling probabilities
            arousal_noise_std: Standard deviation of Gaussian noise added to arousal
                              (0.0 = no noise, 0.1 = small variation)
        """
        self.emotion_arousal_map = emotion_arousal_map or EMOTION_AROUSAL_MAP
        self.emotion_valence_map = emotion_valence_map or EMOTION_VALENCE_MAP
        self.emotion_category_map = emotion_category_map or EMOTION_CATEGORY_MAP
        self.emotion_probs = emotion_probs or DEFAULT_EMOTION_PROBS
        self.arousal_noise_std = arousal_noise_std

        # Build reverse mapping: emotion -> category
        self._emotion_to_category = {}
        for category, emotions in self.emotion_category_map.items():
            for emotion in emotions:
                self._emotion_to_category[emotion] = category

        # Validate probabilities sum to 1.0 (with tolerance)
        prob_sum = sum(self.emotion_probs.values())
        if abs(prob_sum - 1.0) > 1e-6:
            # Normalize probabilities
            self.emotion_probs = {k: v/prob_sum for k, v in self.emotion_probs.items()}

        # Validate all emotions in probs have arousal values
        for emotion in self.emotion_probs.keys():
            if emotion not in self.emotion_arousal_map:
                raise ValueError(f"Emotion '{emotion}' in probabilities not found in arousal map")
            if emotion not in self.emotion_valence_map:
                raise ValueError(f"Emotion '{emotion}' in probabilities not found in valence map")

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

    def get_valence(self, emotion: str) -> float:
        """
        Get discrete valence value for a given emotion (-0.8, 0, or 0.8).

        Args:
            emotion: Emotion label

        Returns:
            Valence value (-0.8 for negative, 0 for neutral, 0.8 for positive)
        """
        if emotion not in self.emotion_valence_map:
            raise ValueError(f"Unknown emotion: {emotion}")
        return self.emotion_valence_map[emotion]

    def classify_valence(self, valence: float) -> str:
        """
        Classify valence into discrete categories (negative/neutral/positive).

        Args:
            valence: Valence value

        Returns:
            'negative', 'neutral', or 'positive'
        """
        if valence > 0:
            return 'positive'
        if valence < 0:
            return 'negative'
        return 'neutral'

    def get_category(self, emotion: str) -> str:
        """
        Get the 9-class category for a given emotion.

        Args:
            emotion: Emotion label

        Returns:
            Category name (e.g., 'high_positive', 'medium_neutral', 'low_negative')
        """
        if emotion not in self._emotion_to_category:
            raise ValueError(f"Unknown emotion: {emotion}")
        return self._emotion_to_category[emotion]

    def classify_emotion(self, arousal: float, valence: float) -> str:
        """
        Classify arousal and valence values into a 9-class category.

        Args:
            arousal: Arousal value
            valence: Valence value

        Returns:
            Category name (e.g., 'high_positive', 'medium_neutral', 'low_negative')
        """
        arousal_class = self.classify_arousal(arousal)
        valence_class = self.classify_valence(valence)
        return f"{arousal_class}_{valence_class}"

    def get_categories(self) -> List[str]:
        """Return list of all 9 category names"""
        return list(self.emotion_category_map.keys())


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

def sample_emotions_batch_stratified(
    num_jobs: int,
    emotion_config: EmotionConfig,
    class_distribution: Dict[str, float] = None
) -> List[Tuple[str, float]]:
    """
    Batch sample emotions with stratified sampling by arousal class.
    
    Ensures the target distribution across all samples for fair comparison.
    
    Args:
        num_jobs: Number of jobs to generate
        emotion_config: EmotionConfig object
        class_distribution: Target distribution for each class
                          e.g., {'high': 0.33, 'medium': 0.34, 'low': 0.33}
                          If None, uses uniform distribution (1/3 each)
    
    Returns:
        List of (emotion_label, arousal) tuples
    """
    classes = ['high', 'medium', 'low']

    if class_distribution is None or class_distribution == 'uniform':
        # Default: uniform distribution
        class_distribution = {'high': 1/3, 'medium': 1/3, 'low': 1/3}
    else:
        # Auto-fill missing classes: distribute remaining proportion equally
        specified_total = sum(class_distribution.get(cls, 0) for cls in classes)
        unspecified_classes = [cls for cls in classes if cls not in class_distribution]
        if unspecified_classes and specified_total < 1:
            remaining_proportion = (1 - specified_total) / len(unspecified_classes)
            class_distribution = {cls: class_distribution.get(cls, remaining_proportion) for cls in classes}

    # Calculate count for each class using Largest Remainder Method
    class_counts = {}
    remainders = {}

    # Step 1: Calculate expected values and allocate floor
    total_allocated = 0
    for cls in classes:
        expected = num_jobs * class_distribution.get(cls, 0)
        floor_val = int(expected)
        class_counts[cls] = floor_val
        remainders[cls] = expected - floor_val
        total_allocated += floor_val

    # Step 2: Distribute remaining slots to classes with largest remainders
    remaining = num_jobs - total_allocated
    if remaining > 0:
        sorted_classes = sorted(remainders.keys(), key=lambda x: remainders[x], reverse=True)
        for i in range(remaining):
            class_counts[sorted_classes[i]] += 1
    
    # Group emotions by arousal class
    emotions_by_class = {'high': [], 'medium': [], 'low': []}
    for emotion, base_arousal in emotion_config.emotion_arousal_map.items():
        emotion_class = emotion_config.classify_arousal(base_arousal)
        emotions_by_class[emotion_class].append((emotion, base_arousal))
    
    # Sample from each class
    emotions_arousal = []
    
    for cls, count in class_counts.items():
        available_emotions = emotions_by_class[cls]
        
        if not available_emotions:
            # Fallback: if no emotions in this class, use any emotion
            available_emotions = list(emotion_config.emotion_arousal_map.items())
        
        # Sample 'count' emotions from this class
        for _ in range(count):
            emotion, base_arousal = available_emotions[
                np.random.randint(len(available_emotions))
            ]
            
            # Add noise if configured
            arousal = base_arousal
            if emotion_config.arousal_noise_std > 0:
                noise = np.random.normal(0, emotion_config.arousal_noise_std)
                arousal = arousal + noise
            
            emotions_arousal.append((emotion, arousal))
    
    # Shuffle to randomize order
    np.random.shuffle(emotions_arousal)

    return emotions_arousal


def sample_emotions_batch_stratified_9class(
    num_jobs: int,
    emotion_config: EmotionConfig,
    class_distribution: Dict[str, float] = None
) -> List[Tuple[str, float, float]]:
    """
    Batch sample emotions with stratified sampling by 9-class category.

    Uses 3 arousal levels (high/medium/low) × 3 valence levels (positive/neutral/negative)
    to create 9 balanced categories for fair comparison.

    Args:
        num_jobs: Number of jobs to generate
        emotion_config: EmotionConfig object
        class_distribution: Target distribution for each of the 9 classes
                          e.g., {'high_positive': 1/9, 'high_neutral': 1/9, ...}
                          If None, uses uniform distribution (1/9 each)

    Returns:
        List of (emotion_label, arousal, valence) tuples
    """
    categories = emotion_config.get_categories()

    if class_distribution is None or class_distribution == 'uniform':
        # Default: uniform distribution across 9 classes
        class_distribution = {cat: 1/9 for cat in categories}
    else:
        # Auto-fill missing categories: distribute remaining proportion equally
        specified_total = sum(class_distribution.get(cat, 0) for cat in categories)
        unspecified_cats = [cat for cat in categories if cat not in class_distribution]
        if unspecified_cats and specified_total < 1:
            remaining_proportion = (1 - specified_total) / len(unspecified_cats)
            class_distribution = {cat: class_distribution.get(cat, remaining_proportion) for cat in categories}

    # Calculate count for each class using Largest Remainder Method
    # This ensures fair distribution with at most 1 difference between categories
    class_counts = {}
    remainders = {}

    # Step 1: Calculate expected values and allocate floor
    total_allocated = 0
    for cat in categories:
        expected = num_jobs * class_distribution.get(cat, 0)
        floor_val = int(expected)
        class_counts[cat] = floor_val
        remainders[cat] = expected - floor_val
        total_allocated += floor_val

    # Step 2: Distribute remaining slots to categories with largest remainders
    remaining = num_jobs - total_allocated
    if remaining > 0:
        sorted_cats = sorted(remainders.keys(), key=lambda x: remainders[x], reverse=True)
        for i in range(remaining):
            class_counts[sorted_cats[i]] += 1

    # Sample from each category
    emotions_arousal_valence = []

    for category, count in class_counts.items():
        available_emotions = emotion_config.emotion_category_map.get(category, [])

        if not available_emotions:
            # Fallback: if no emotions in this category, skip
            continue

        # Sample 'count' emotions from this category
        for _ in range(count):
            emotion = available_emotions[np.random.randint(len(available_emotions))]
            base_arousal = emotion_config.emotion_arousal_map[emotion]
            valence = emotion_config.emotion_valence_map[emotion]

            # Add noise if configured
            arousal = base_arousal
            if emotion_config.arousal_noise_std > 0:
                noise = np.random.normal(0, emotion_config.arousal_noise_std)
                arousal = arousal + noise

            emotions_arousal_valence.append((emotion, arousal, valence))

    # Shuffle to randomize order
    np.random.shuffle(emotions_arousal_valence)

    return emotions_arousal_valence


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
    valence_values = list(config.emotion_valence_map.values())

    # Calculate weighted statistics based on emotion probabilities
    weighted_arousal = sum(config.emotion_arousal_map[e] * config.emotion_probs[e]
                          for e in config.emotion_probs.keys())
    weighted_valence = sum(config.emotion_valence_map[e] * config.emotion_probs[e]
                           for e in config.emotion_probs.keys())

    # Count emotions by arousal category
    high_count = sum(1 for a in arousal_values if a >= 0.6)
    medium_count = sum(1 for a in arousal_values if -0.3 <= a < 0.6)
    low_count = sum(1 for a in arousal_values if a < -0.3)

    positive_count = sum(1 for v in valence_values if v > 0)
    neutral_count = sum(1 for v in valence_values if v == 0)
    negative_count = sum(1 for v in valence_values if v < 0)

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
        'valence_mean': np.mean(valence_values),
        'valence_std': np.std(valence_values),
        'weighted_valence_mean': weighted_valence,
        'positive_valence_count': positive_count,
        'neutral_valence_count': neutral_count,
        'negative_valence_count': negative_count,
    }
