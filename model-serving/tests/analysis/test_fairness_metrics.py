"""
Tests for fairness_metrics module.
"""

from analysis.fairness_metrics import (
    calculate_jain_fairness_index,
    calculate_per_class_metrics,
    calculate_fairness_across_emotions,
    compare_scheduler_fairness,
)
from core.job import Job


def approx_equal(a, b, rel=1e-4):
    """Helper function for approximate equality comparison."""
    if b == 0:
        return abs(a) < rel
    return abs(a - b) / abs(b) < rel


def test_jain_fairness_index():
    """Tests for Jain Fairness Index calculation."""
    print("\n--- Test: Jain Fairness Index ---")

    # Test perfect fairness with three equal values
    jain = calculate_jain_fairness_index([1, 1, 1])
    assert approx_equal(jain, 1.0), f"Expected ~1.0, got {jain}"
    print("  Perfect fairness (3 equal values): PASSED")

    # Test perfect fairness with five equal values
    jain = calculate_jain_fairness_index([5, 5, 5, 5, 5])
    assert approx_equal(jain, 1.0), f"Expected ~1.0, got {jain}"
    print("  Perfect fairness (5 equal values): PASSED")

    # Test some unfairness
    jain = calculate_jain_fairness_index([1, 2, 3])
    # Expected: (1+2+3)^2 / (3 * (1+4+9)) = 36 / 42 = 0.857
    assert approx_equal(jain, 0.857, rel=1e-2), f"Expected ~0.857, got {jain}"
    print("  Some unfairness [1,2,3]: PASSED")

    # Test high unfairness
    jain = calculate_jain_fairness_index([1, 1, 10])
    # Expected: (12)^2 / (3 * 102) = 144 / 306 = 0.47
    assert approx_equal(jain, 0.47, rel=1e-2), f"Expected ~0.47, got {jain}"
    print("  High unfairness [1,1,10]: PASSED")

    # Test empty list
    jain = calculate_jain_fairness_index([])
    assert jain == 0.0, f"Expected 0.0, got {jain}"
    print("  Empty list: PASSED")

    # Test all zeros
    jain = calculate_jain_fairness_index([0, 0, 0])
    assert jain == 1.0, f"Expected 1.0, got {jain}"
    print("  All zeros: PASSED")


def test_per_class_metrics():
    """Tests for per-class metrics calculation."""
    print("\n--- Test: Per-Class Metrics ---")

    jobs = [
        Job(0, 2.0, 0.0, emotion_label='excited', arousal=0.9, emotion_class='high'),
        Job(1, 1.5, 1.0, emotion_label='sad', arousal=-0.6, emotion_class='low'),
        Job(2, 2.5, 2.0, emotion_label='angry', arousal=0.8, emotion_class='high'),
        Job(3, 1.0, 3.0, emotion_label='calm', arousal=-0.3, emotion_class='low'),
        Job(4, 1.8, 4.0, emotion_label='neutral', arousal=0.0, emotion_class='medium'),
    ]

    # Simulate FCFS scheduling
    current_time = 0
    for job in jobs:
        waiting = current_time - job.arrival_time
        job.waiting_duration = max(0, waiting)
        job.completion_time = current_time + job.execution_duration
        current_time = job.completion_time

    metrics = calculate_per_class_metrics(jobs)

    assert 'high' in metrics, "Missing 'high' class"
    assert 'low' in metrics, "Missing 'low' class"
    assert 'medium' in metrics, "Missing 'medium' class"
    assert 'overall' in metrics, "Missing 'overall' key"
    print("  All emotion classes present: PASSED")

    assert metrics['high']['count'] == 2, f"Expected high count=2, got {metrics['high']['count']}"
    assert metrics['low']['count'] == 2, f"Expected low count=2, got {metrics['low']['count']}"
    assert metrics['medium']['count'] == 1, f"Expected medium count=1, got {metrics['medium']['count']}"
    print("  Correct job counts per class: PASSED")

    # Test empty list
    metrics = calculate_per_class_metrics([])
    assert metrics == {}, f"Expected empty dict, got {metrics}"
    print("  Empty list returns empty dict: PASSED")


def test_fairness_across_emotions():
    """Tests for fairness calculation across emotion classes."""
    print("\n--- Test: Fairness Across Emotions ---")

    jobs = [
        Job(0, 2.0, 0.0, emotion_label='excited', arousal=0.9, emotion_class='high'),
        Job(1, 1.5, 1.0, emotion_label='sad', arousal=-0.6, emotion_class='low'),
        Job(2, 2.5, 2.0, emotion_label='angry', arousal=0.8, emotion_class='high'),
        Job(3, 1.0, 3.0, emotion_label='calm', arousal=-0.3, emotion_class='low'),
        Job(4, 1.8, 4.0, emotion_label='neutral', arousal=0.0, emotion_class='medium'),
    ]

    current_time = 0
    for job in jobs:
        waiting = current_time - job.arrival_time
        job.waiting_duration = max(0, waiting)
        job.completion_time = current_time + job.execution_duration
        current_time = job.completion_time

    fairness = calculate_fairness_across_emotions(jobs, metric='waiting_time')

    assert 'jain_index' in fairness, "Missing 'jain_index'"
    assert 'per_class_values' in fairness, "Missing 'per_class_values'"
    assert 'coefficient_of_variation' in fairness, "Missing 'coefficient_of_variation'"
    assert 'max_min_ratio' in fairness, "Missing 'max_min_ratio'"
    print("  All fairness keys present: PASSED")

    # Jain index should be between 0 and 1
    assert 0 <= fairness['jain_index'] <= 1, f"Jain index out of range: {fairness['jain_index']}"
    print("  Jain index in valid range: PASSED")


def test_scheduler_fairness_comparison():
    """Tests for comparing fairness across different schedulers."""
    print("\n--- Test: Scheduler Fairness Comparison ---")

    # Scenario 1: Fair scheduler (FCFS-like)
    jobs_fair = []
    for i in range(15):
        emotion_class = ['high', 'medium', 'low'][i % 3]
        arousal = [0.8, 0.0, -0.6][i % 3]
        job = Job(i, 2.0, i * 1.0, emotion_class=emotion_class, arousal=arousal)
        job.waiting_duration = 0
        job.completion_time = job.arrival_time + job.execution_duration
        jobs_fair.append(job)

    # Scenario 2: Unfair scheduler (prioritizes high arousal)
    high_jobs = []
    other_jobs = []
    for i in range(15):
        emotion_class = ['high', 'medium', 'low'][i % 3]
        arousal = [0.8, 0.0, -0.6][i % 3]
        job = Job(i, 2.0, i * 1.0, emotion_class=emotion_class, arousal=arousal)
        if emotion_class == 'high':
            high_jobs.append(job)
        else:
            other_jobs.append(job)

    # Schedule high arousal first
    current_time = 0
    for job in high_jobs + other_jobs:
        waiting = max(0, current_time - job.arrival_time)
        job.waiting_duration = waiting
        job.completion_time = current_time + job.execution_duration
        current_time = job.completion_time
    jobs_unfair = high_jobs + other_jobs

    comparison = compare_scheduler_fairness({
        'Fair (FCFS)': jobs_fair,
        'Unfair (Priority)': jobs_unfair
    })

    assert 'summary' in comparison, "Missing 'summary'"
    assert 'jain_index_waiting_time' in comparison['summary'], "Missing 'jain_index_waiting_time'"
    print("  Comparison structure correct: PASSED")

    # Fair scheduler should have higher or equal Jain index
    fair_jain = comparison['summary']['jain_index_waiting_time']['Fair (FCFS)']
    unfair_jain = comparison['summary']['jain_index_waiting_time']['Unfair (Priority)']

    assert fair_jain >= unfair_jain or abs(fair_jain - unfair_jain) < 0.1, \
        f"Fair scheduler should have higher fairness: fair={fair_jain}, unfair={unfair_jain}"
    print(f"  Fair scheduler Jain index ({fair_jain:.4f}) >= Unfair ({unfair_jain:.4f}): PASSED")


def test_fairness_metrics():
    """Integration test for fairness metrics (legacy test)."""
    print("=" * 70)
    print("Fairness Metrics Test")
    print("=" * 70)

    # Test 1: Jain Fairness Index calculation
    print("\n1. Jain Fairness Index Calculation")
    test_cases = [
        [1, 1, 1],  # Perfect fairness
        [1, 2, 3],  # Some unfairness
        [1, 1, 10],  # High unfairness
        [5, 5, 5, 5, 5]  # Perfect fairness, more groups
    ]

    for values in test_cases:
        jain = calculate_jain_fairness_index(values)
        print(f"   Values: {values} -> Jain Index: {jain:.4f}")

    # Test 2: Create test jobs with emotions
    print("\n2. Emotion-aware Job Fairness Analysis")
    jobs = [
        Job(0, 2.0, 0.0, emotion_label='excited', arousal=0.9, emotion_class='high'),
        Job(1, 1.5, 1.0, emotion_label='sad', arousal=-0.6, emotion_class='low'),
        Job(2, 2.5, 2.0, emotion_label='angry', arousal=0.8, emotion_class='high'),
        Job(3, 1.0, 3.0, emotion_label='calm', arousal=-0.3, emotion_class='low'),
        Job(4, 1.8, 4.0, emotion_label='neutral', arousal=0.0, emotion_class='medium'),
    ]

    # Compute completion metrics
    current_time = 0
    for job in jobs:
        waiting = current_time - job.arrival_time
        job.waiting_duration = max(0, waiting)
        job.completion_time = current_time + job.execution_duration
        current_time = job.completion_time

    # Calculate per-class metrics
    metrics = calculate_per_class_metrics(jobs)
    print("\n   Per-Class Metrics:")
    for emotion_class, class_metrics in metrics.items():
        if emotion_class != 'overall':
            print(f"     {emotion_class}:")
            print(f"       Count: {class_metrics['count']}")
            print(f"       Avg waiting: {class_metrics['avg_waiting_time']:.3f}")
            print(f"       Avg turnaround: {class_metrics['avg_turnaround_time']:.3f}")

    # Calculate fairness
    print("\n3. Fairness Analysis")
    fairness = calculate_fairness_across_emotions(jobs, metric='waiting_time')
    print(f"   Waiting Time Fairness:")
    print(f"     Jain Index: {fairness['jain_index']:.4f}")
    print(f"     Coefficient of Variation: {fairness['coefficient_of_variation']:.4f}")
    print(f"     Max/Min Ratio: {fairness['max_min_ratio']:.4f}")
    print(f"     Per-class values:")
    for cls, val in fairness['per_class_values'].items():
        print(f"       {cls}: {val:.3f}")

    # Test 3: Compare different scenarios
    print("\n4. Scheduler Fairness Comparison")

    # Scenario 1: Fair scheduler (FCFS-like)
    jobs_fair = []
    for i in range(15):
        emotion_class = ['high', 'medium', 'low'][i % 3]
        arousal = [0.8, 0.0, -0.6][i % 3]
        job = Job(i, 2.0, i*1.0, emotion_class=emotion_class, arousal=arousal)
        job.waiting_duration = 0
        job.completion_time = job.arrival_time + job.execution_duration
        jobs_fair.append(job)

    # Scenario 2: Unfair scheduler (prioritizes high arousal)
    high_jobs = []
    other_jobs = []
    for i in range(15):
        emotion_class = ['high', 'medium', 'low'][i % 3]
        arousal = [0.8, 0.0, -0.6][i % 3]
        job = Job(i, 2.0, i*1.0, emotion_class=emotion_class, arousal=arousal)
        if emotion_class == 'high':
            high_jobs.append(job)
        else:
            other_jobs.append(job)

    # Schedule high arousal first
    current_time = 0
    for job in high_jobs + other_jobs:
        waiting = max(0, current_time - job.arrival_time)
        job.waiting_duration = waiting
        job.completion_time = current_time + job.execution_duration
        current_time = job.completion_time
    jobs_unfair = high_jobs + other_jobs

    # Compare fairness
    comparison = compare_scheduler_fairness({
        'Fair (FCFS)': jobs_fair,
        'Unfair (Priority)': jobs_unfair
    })

    print("\n   Fairness Comparison:")
    print(f"   {'Scheduler':<20} {'Jain Index (Waiting)':<25}")
    print(f"   {'-'*45}")
    for scheduler, jain in comparison['summary']['jain_index_waiting_time'].items():
        print(f"   {scheduler:<20} {jain:<25.4f}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    test_fairness_metrics()
