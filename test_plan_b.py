#!/usr/bin/env python3
"""
Simple test script for 方案 B implementation
Tests the fixed-rate arrival model with both FCFS and SSJF schedulers
"""

import sys
import os

# Add model-serving directory to Python path
model_serving_dir = os.path.join(os.path.dirname(__file__), 'model-serving')
sys.path.insert(0, model_serving_dir)

import numpy as np
from core.emotion import EmotionConfig
from workload.service_time_mapper import ServiceTimeConfig
from core.scheduler_base import FCFSScheduler
from core.ssjf_emotion import SSJFEmotionScheduler
from simulator import run_scheduling_loop


def test_basic_simulation():
    """Test basic simulation with FCFS and SSJF"""
    print("=" * 80)
    print("Testing 方案 B: Fixed-Rate Arrival Model")
    print("=" * 80)

    # Configuration
    arrival_rate = 0.5  # 0.5 req/sec
    simulation_time = 20.0  # 20 seconds
    random_seed = 42

    # Set random seed
    np.random.seed(random_seed)
    import random
    random.seed(random_seed)

    # Create configs
    emotion_config = EmotionConfig(arousal_noise_std=0.0)
    service_config = ServiceTimeConfig(
        base_service_time=2.0,
        alpha=0.5,
        rho=1.0
    )

    print(f"\nConfiguration:")
    print(f"  Arrival rate: {arrival_rate} req/sec")
    print(f"  Simulation time: {simulation_time}s")
    print(f"  Expected jobs: ~{int(arrival_rate * simulation_time)}")
    print(f"  Random seed: {random_seed}")

    # Test FCFS
    print(f"\n{'='*80}")
    print("Test 1: FCFS Scheduler")
    print(f"{'='*80}")

    # Reset random seed for fair comparison
    np.random.seed(random_seed)
    random.seed(random_seed)

    fcfs_scheduler = FCFSScheduler()
    fcfs_jobs, fcfs_metrics = run_scheduling_loop(
        scheduler=fcfs_scheduler,
        arrival_rate=arrival_rate,
        simulation_time=simulation_time,
        emotion_config=emotion_config,
        service_time_config=service_config,
        enable_emotion=True,
        verbose=True,
        llm_handler=None,
        llm_skip_on_error=True
    )

    print(f"\nFCFS Results:")
    print(f"  Total jobs: {fcfs_metrics['total_jobs']}")
    print(f"  Jobs by deadline: {fcfs_metrics['jobs_by_deadline']}")
    print(f"  Jobs after deadline: {fcfs_metrics['jobs_after_deadline']}")
    print(f"  Effective throughput: {fcfs_metrics['effective_throughput']:.3f} jobs/sec")
    print(f"  Total time: {fcfs_metrics['total_time']:.2f}s")

    # Test SSJF
    print(f"\n{'='*80}")
    print("Test 2: SSJF-Emotion Scheduler")
    print(f"{'='*80}")

    # Reset random seed for fair comparison
    np.random.seed(random_seed)
    random.seed(random_seed)

    ssjf_scheduler = SSJFEmotionScheduler(
        starvation_threshold=float('inf'),
        starvation_coefficient=3.0
    )
    ssjf_jobs, ssjf_metrics = run_scheduling_loop(
        scheduler=ssjf_scheduler,
        arrival_rate=arrival_rate,
        simulation_time=simulation_time,
        emotion_config=emotion_config,
        service_time_config=service_config,
        enable_emotion=True,
        verbose=True,
        llm_handler=None,
        llm_skip_on_error=True
    )

    print(f"\nSSJF Results:")
    print(f"  Total jobs: {ssjf_metrics['total_jobs']}")
    print(f"  Jobs by deadline: {ssjf_metrics['jobs_by_deadline']}")
    print(f"  Jobs after deadline: {ssjf_metrics['jobs_after_deadline']}")
    print(f"  Effective throughput: {ssjf_metrics['effective_throughput']:.3f} jobs/sec")
    print(f"  Total time: {ssjf_metrics['total_time']:.2f}s")

    # Compare
    print(f"\n{'='*80}")
    print("Comparison")
    print(f"{'='*80}")

    throughput_improvement = (ssjf_metrics['effective_throughput'] - fcfs_metrics['effective_throughput']) / fcfs_metrics['effective_throughput'] * 100

    print(f"\nThroughput Comparison:")
    print(f"  FCFS: {fcfs_metrics['effective_throughput']:.3f} jobs/sec")
    print(f"  SSJF: {ssjf_metrics['effective_throughput']:.3f} jobs/sec")
    print(f"  Improvement: {throughput_improvement:+.1f}%")

    print(f"\nJobs Completed by Deadline:")
    print(f"  FCFS: {fcfs_metrics['jobs_by_deadline']}")
    print(f"  SSJF: {ssjf_metrics['jobs_by_deadline']}")
    print(f"  Difference: {ssjf_metrics['jobs_by_deadline'] - fcfs_metrics['jobs_by_deadline']:+d}")

    # Verify basic expectations
    assert fcfs_metrics['total_jobs'] > 0, "Should complete at least one job"
    assert ssjf_metrics['total_jobs'] > 0, "Should complete at least one job"
    assert fcfs_metrics['effective_throughput'] > 0, "Throughput should be positive"
    assert ssjf_metrics['effective_throughput'] > 0, "Throughput should be positive"

    print(f"\n{'='*80}")
    print("✓ All tests passed!")
    print(f"{'='*80}")

    return fcfs_metrics, ssjf_metrics


if __name__ == '__main__':
    try:
        fcfs_metrics, ssjf_metrics = test_basic_simulation()
        print("\n✓ Test completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
