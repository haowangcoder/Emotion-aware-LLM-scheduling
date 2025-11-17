"""
Tests for Emotion-aware Task Generator

This test file contains tests for the emotion-aware task generation functionality.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.emotion import EmotionConfig
from workload.service_time_mapper import ServiceTimeConfig
from workload.task_generator import create_emotion_aware_jobs, get_emotion_aware_statistics


# ============================================================================
# Test Functions
# ============================================================================

def test_basic_job_creation():
    """Test basic emotion-aware job creation with Poisson arrival"""
    print("\n" + "=" * 70)
    print("Test 1: Basic Emotion-aware Job Creation (n=10, Poisson)")
    print("=" * 70)

    emotion_config = EmotionConfig()
    service_config = ServiceTimeConfig(base_service_time=2.0, alpha=0.5)

    jobs = create_emotion_aware_jobs(
        num_jobs=10,
        arrival_rate=2.0,
        emotion_config=emotion_config,
        service_time_config=service_config
    )

    print(f"\n{'Job ID':<8} {'Emotion':<15} {'Arousal':<10} {'Class':<10} {'Service':<10} {'Arrival':<10}")
    print(f"{'-'*70}")
    for job in jobs[:10]:
        print(f"{job.job_id:<8} {job.emotion_label:<15} {job.arousal:<10.2f} "
              f"{job.emotion_class:<10} {job.execution_duration:<10.2f} {job.arrival_time:<10.2f}")

    # Assertions
    assert len(jobs) == 10, f"Expected 10 jobs, got {len(jobs)}"

    for job in jobs:
        assert job.job_id >= 0, "Job ID should be non-negative"
        assert job.emotion_label is not None, "Emotion label should not be None"
        assert job.arousal is not None, "Arousal should not be None"
        assert job.emotion_class in ['low', 'medium', 'high'], f"Invalid emotion class: {job.emotion_class}"
        assert job.execution_duration > 0, "Service time should be positive"
        assert job.arrival_time >= 0, "Arrival time should be non-negative"

    # Check arrival times are sorted
    for i in range(len(jobs) - 1):
        assert jobs[i].arrival_time <= jobs[i + 1].arrival_time, "Arrival times should be sorted"

    print("\n✓ All assertions passed")


def test_job_statistics():
    """Test job generation statistics calculation"""
    print("\n" + "=" * 70)
    print("Test 2: Job Generation Statistics")
    print("=" * 70)

    emotion_config = EmotionConfig()
    service_config = ServiceTimeConfig(base_service_time=2.0, alpha=0.5)

    jobs = create_emotion_aware_jobs(
        num_jobs=100,
        arrival_rate=2.0,
        emotion_config=emotion_config,
        service_time_config=service_config,
        random_seed=42  # For reproducibility
    )

    stats = get_emotion_aware_statistics(jobs)

    print(f"Total jobs: {stats['num_jobs']}")
    print(f"Arousal mean: {stats['arousal_mean']:.3f}")
    print(f"Service time mean: {stats['service_time_mean']:.3f} ± {stats['service_time_std']:.3f}")
    print(f"Service time range: [{stats['service_time_min']:.3f}, {stats['service_time_max']:.3f}]")
    print(f"Arrival interval mean: {stats['arrival_interval_mean']:.3f}")
    print(f"Emotion class distribution:")
    for cls, count in stats['emotion_class_counts'].items():
        print(f"  {cls}: {count} ({count/stats['num_jobs']*100:.1f}%)")

    # Assertions
    assert stats['num_jobs'] == 100, f"Expected 100 jobs, got {stats['num_jobs']}"
    assert -1 <= stats['arousal_mean'] <= 1, "Arousal mean should be in [-1, 1]"
    assert stats['service_time_mean'] > 0, "Service time mean should be positive"
    assert stats['service_time_min'] > 0, "Service time min should be positive"
    assert stats['service_time_max'] > stats['service_time_min'], "Max should be > min"
    assert stats['arrival_interval_mean'] > 0, "Arrival interval mean should be positive"

    # Check emotion class distribution sums to total
    total_classes = sum(stats['emotion_class_counts'].values())
    assert total_classes == 100, f"Emotion class counts should sum to 100, got {total_classes}"

    print("\n✓ All assertions passed")


def test_with_without_emotion_comparison():
    """Test comparison between emotion-aware and non-emotion-aware job generation"""
    print("\n" + "=" * 70)
    print("Test 3: Comparison: With vs Without Emotion Awareness")
    print("=" * 70)

    jobs_with_emotion = create_emotion_aware_jobs(
        num_jobs=100,
        arrival_rate=2.0,
        enable_emotion=True,
        random_seed=42
    )
    jobs_without_emotion = create_emotion_aware_jobs(
        num_jobs=100,
        arrival_rate=2.0,
        enable_emotion=False,
        random_seed=42
    )

    stats_with = get_emotion_aware_statistics(jobs_with_emotion)
    stats_without = get_emotion_aware_statistics(jobs_without_emotion)

    print(f"\n{'Metric':<30} {'With Emotion':<15} {'Without Emotion':<15}")
    print(f"{'-'*60}")
    print(f"{'Service time mean':<30} {stats_with['service_time_mean']:<15.3f} "
          f"{stats_without['service_time_mean']:<15.3f}")
    print(f"{'Service time std':<30} {stats_with['service_time_std']:<15.3f} "
          f"{stats_without['service_time_std']:<15.3f}")
    print(f"{'Service time range':<30} "
          f"{stats_with['service_time_max']-stats_with['service_time_min']:<15.3f} "
          f"{stats_without['service_time_max']-stats_without['service_time_min']:<15.3f}")

    # Assertions
    # With emotion should have more variance than without
    assert stats_with['service_time_std'] > 0, "With emotion should have variance"

    # Without emotion should have all neutral emotions
    for job in jobs_without_emotion:
        assert job.emotion_label == 'neutral', f"Expected neutral, got {job.emotion_label}"
        assert job.arousal == 0.0, f"Expected arousal 0.0, got {job.arousal}"

    # With emotion should have variety
    emotion_labels_with = set(job.emotion_label for job in jobs_with_emotion)
    assert len(emotion_labels_with) > 1, "With emotion should have multiple emotion types"

    print("\n✓ All assertions passed")


def test_reproducibility_with_seed():
    """Test that random seed produces reproducible results"""
    print("\n" + "=" * 70)
    print("Test 4: Reproducibility with Random Seed")
    print("=" * 70)

    # Generate jobs twice with same seed
    jobs1 = create_emotion_aware_jobs(
        num_jobs=10,
        arrival_rate=2.0,
        random_seed=123
    )
    jobs2 = create_emotion_aware_jobs(
        num_jobs=10,
        arrival_rate=2.0,
        random_seed=123
    )

    print("Comparing two runs with same seed (123):")
    print(f"{'Job ID':<8} {'Emotion 1':<15} {'Emotion 2':<15} {'Match':<10}")
    print(f"{'-'*50}")

    all_match = True
    for j1, j2 in zip(jobs1, jobs2):
        match = j1.emotion_label == j2.emotion_label
        all_match = all_match and match
        print(f"{j1.job_id:<8} {j1.emotion_label:<15} {j2.emotion_label:<15} {'✓' if match else '✗':<10}")

    # Assertions
    assert all_match, "Jobs with same seed should have identical emotions"

    for j1, j2 in zip(jobs1, jobs2):
        assert j1.emotion_label == j2.emotion_label, "Emotions should match"
        assert abs(j1.arousal - j2.arousal) < 0.001, "Arousals should match"
        assert abs(j1.arrival_time - j2.arrival_time) < 0.001, "Arrival times should match"
        assert abs(j1.execution_duration - j2.execution_duration) < 0.001, "Service times should match"

    print("\n✓ All jobs identical - reproducibility confirmed!")


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("TASK GENERATOR TEST SUITE")
    print("=" * 70)

    # Run all tests
    test_basic_job_creation()
    test_job_statistics()
    test_with_without_emotion_comparison()
    test_reproducibility_with_seed()

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
