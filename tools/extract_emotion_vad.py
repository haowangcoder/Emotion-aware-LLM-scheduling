#!/usr/bin/env python3
"""
Extract VAD (Valence-Arousal-Dominance) values for target emotions from NRC-VAD-Lexicon.

This script extracts emotion VAD values for use in the Emotion-aware LLM Scheduling project.
It reads the NRC-VAD-Lexicon and outputs a CSV file with the target emotions.

Reference:
    Mohammad, S. M. (2018). Obtaining Reliable Human Ratings of Valence, Arousal,
    and Dominance for 20,000 English Words. In Proceedings of ACL 2018.
"""

import csv
import os
from typing import Dict, List, Tuple

# Configuration
LEXICON_PATH = "NRC-VAD-Lexicon-v2.1.txt"
OUTPUT_PATH = "emotion_vad_values.csv"

# Target emotions for the scheduling system
TARGET_EMOTIONS = [
    'excited', 'joyful', 'proud', 'hopeful', 'trusting', 'faithful',
    'grateful', 'confident', 'content', 'sentimental', 'nostalgic',
    'prepared', 'impressed', 'caring', 'surprised', 'anticipating',
    'terrified', 'afraid', 'anxious', 'angry', 'furious', 'annoyed',
    'disgusted', 'jealous', 'embarrassed', 'apprehensive', 'guilty',
    'ashamed', 'sad', 'lonely', 'disappointed', 'devastated'
]

# Substitution mapping for emotions not found in lexicon
SUBSTITUTIONS = {
    'anticipating': 'anticipation'  # 'anticipating' not in lexicon, use noun form
}


def load_lexicon(lexicon_path: str) -> Dict[str, Tuple[float, float, float]]:
    """
    Load the NRC-VAD-Lexicon into a dictionary.

    Args:
        lexicon_path: Path to the NRC-VAD-Lexicon TSV file

    Returns:
        Dictionary mapping term -> (valence, arousal, dominance)
    """
    lexicon = {}
    with open(lexicon_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)  # Skip header
        for row in reader:
            if len(row) >= 4:
                term = row[0].lower()
                valence = float(row[1])
                arousal = float(row[2])
                dominance = float(row[3])
                lexicon[term] = (valence, arousal, dominance)
    return lexicon


def classify_russell_quadrant(valence: float, arousal: float) -> str:
    """
    Classify emotion into Russell's circumplex model quadrant.

    Both valence and arousal are in [-1, 1] range.
    Quadrant boundaries are at 0 for both dimensions.

    Quadrants:
        - excited: valence >= 0 AND arousal >= 0 (high arousal, positive valence)
        - calm: valence >= 0 AND arousal < 0 (low arousal, positive valence)
        - panic: valence < 0 AND arousal >= 0 (high arousal, negative valence)
        - depression: valence < 0 AND arousal < 0 (low arousal, negative valence)

    Args:
        valence: Valence value in [-1, 1]
        arousal: Arousal value in [-1, 1]

    Returns:
        Quadrant name: 'excited', 'calm', 'panic', or 'depression'
    """
    if valence >= 0:
        return 'excited' if arousal >= 0 else 'calm'
    else:
        return 'panic' if arousal >= 0 else 'depression'


def extract_emotions(
    lexicon: Dict[str, Tuple[float, float, float]],
    target_emotions: List[str],
    substitutions: Dict[str, str]
) -> List[Dict]:
    """
    Extract VAD values for target emotions.

    Args:
        lexicon: Dictionary from load_lexicon()
        target_emotions: List of emotion names to extract
        substitutions: Mapping of emotion -> substitute term

    Returns:
        List of dictionaries with emotion data
    """
    results = []
    missing = []

    for emotion in target_emotions:
        lookup_term = substitutions.get(emotion, emotion)

        if lookup_term in lexicon:
            valence, arousal, dominance = lexicon[lookup_term]
            quadrant = classify_russell_quadrant(valence, arousal)
            results.append({
                'emotion': emotion,
                'valence': valence,
                'arousal': arousal,
                'dominance': dominance,
                'russell_quadrant': quadrant
            })
        else:
            missing.append(emotion)
            print(f"WARNING: '{emotion}' (lookup: '{lookup_term}') not found in lexicon")

    if missing:
        print(f"\nMissing emotions: {missing}")

    return results


def save_to_csv(results: List[Dict], output_path: str) -> None:
    """Save extraction results to CSV file."""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['emotion', 'valence', 'arousal', 'dominance', 'russell_quadrant']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {len(results)} emotions to {output_path}")


def main():
    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    lexicon_path = os.path.join(script_dir, LEXICON_PATH)
    output_path = os.path.join(script_dir, OUTPUT_PATH)

    print(f"Loading lexicon from: {lexicon_path}")
    lexicon = load_lexicon(lexicon_path)
    print(f"Loaded {len(lexicon)} terms from lexicon")

    print(f"\nExtracting {len(TARGET_EMOTIONS)} target emotions...")
    results = extract_emotions(lexicon, TARGET_EMOTIONS, SUBSTITUTIONS)

    save_to_csv(results, output_path)

    print(f"\nTotal emotions extracted: {len(results)}/{len(TARGET_EMOTIONS)}")


if __name__ == '__main__':
    main()
