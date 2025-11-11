"""
Base configuration for the emotion-aware LLM scheduling system.

Contains paths, dataset configuration, logging settings, and general experiment parameters.
"""
import os


# ============================================================================
# GENERAL EXPERIMENT CONFIGURATION
# ============================================================================

# Enable/disable emotion-aware features
ENABLE_EMOTION_AWARE = True
ENABLE_EMOTION_AWARE = os.environ.get('ENABLE_EMOTION_ENV', 'True').lower() == 'true'

# Emotion Dataset Path (EmpatheticDialogues)
EMOTION_DATASET_PATH = './dataset'  # Path to EmpatheticDialogues dataset
EMOTION_DATASET_PATH = os.environ.get('EMOTION_DATASET_PATH_ENV', EMOTION_DATASET_PATH)

# Fairness Metrics Configuration
FAIRNESS_METRIC = 'waiting_time'  # 'waiting_time' or 'turnaround_time'
CALCULATE_FAIRNESS = True  # Whether to calculate fairness metrics

# Results and Cache Directories
RESULTS_DIR = 'results'
CACHE_DIR = 'results/cache'
