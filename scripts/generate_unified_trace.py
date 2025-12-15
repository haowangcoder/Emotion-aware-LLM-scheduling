#!/usr/bin/env python3
"""
Generate a unified job trace for load sweep experiments.

This script creates a base trace with job attributes (emotion, service_time, etc.)
that can be reused across different load levels. The arrival_times will be
recalculated based on the actual arrival_rate for each load level.

Usage:
    python scripts/generate_unified_trace.py --output_dir results/experiments/exp8_load_sweep_timewindow
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "model-serving"))

from core.emotion import EmotionConfig
from workload.task_generator import sample_emotions_batch_stratified_quadrant
from core.affect_weight import compute_urgency_and_weight


def generate_base_trace(
    num_jobs: int,
    emotion_config: EmotionConfig,
    default_service_time: float = 2.0,
    w_max: float = 2.0,
    p: float = 1.0,
    q: float = 1.0,
    random_seed: int = 42,
) -> list:
    """
    Generate a base trace with job attributes but without arrival times.

    The arrival times will be calculated separately for each load level.
    """
    np.random.seed(random_seed)
    import random
    random.seed(random_seed)

    # Generate emotions with stratified sampling
    emotions_data = sample_emotions_batch_stratified_quadrant(
        num_jobs, emotion_config, None
    )

    trace = []
    for job_id in range(num_jobs):
        emotion_label, arousal, valence, quadrant = emotions_data[job_id]

        # Compute affect weight
        urgency, affect_weight = compute_urgency_and_weight(
            arousal=arousal,
            valence=valence,
            w_max=w_max,
            p=p,
            q=q,
        )

        job_entry = {
            'job_id': job_id,
            'emotion': emotion_label,
            'arousal': float(arousal),
            'valence': float(valence),
            'russell_quadrant': quadrant,
            'service_time': float(default_service_time),
            'affect_weight': float(affect_weight),
            'urgency': float(urgency),
            # arrival_time will be set later based on load
        }

        trace.append(job_entry)

    return trace


def add_arrival_times(
    base_trace: list,
    arrival_rate: float,
    random_seed: int = 42,
) -> list:
    """
    Add arrival times to a base trace based on the specified arrival rate.

    Uses the same random seed offset for consistent relative ordering.
    """
    # Use a separate seed for arrival times to ensure consistency
    np.random.seed(random_seed + 1000)  # Offset to not interfere with emotion sampling

    num_jobs = len(base_trace)
    intervals = np.random.exponential(1.0 / arrival_rate, num_jobs)
    arrival_times = np.cumsum(intervals)

    trace = []
    for i, job in enumerate(base_trace):
        new_job = job.copy()
        new_job['arrival_time'] = float(arrival_times[i])
        trace.append(new_job)

    return trace


def main():
    parser = argparse.ArgumentParser(
        description="Generate unified job trace for load sweep experiments"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/experiments/exp8_load_sweep_timewindow",
        help="Base output directory",
    )
    parser.add_argument(
        "--num_jobs",
        type=int,
        default=160,  # 2x trace size for time_window mode
        help="Number of jobs in the trace",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--load_values",
        type=str,
        default="0.5,0.7,0.9,1.0,1.2",
        help="Comma-separated load values",
    )
    parser.add_argument(
        "--base_service_time",
        type=float,
        default=2.0,
        help="Base service time for arrival rate calculation",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_values = [float(x) for x in args.load_values.split(',')]

    print("=" * 60)
    print("Generating Unified Job Trace for Load Sweep")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Number of jobs: {args.num_jobs}")
    print(f"Random seed: {args.random_seed}")
    print(f"Load values: {load_values}")
    print()

    # Generate base trace (without arrival times)
    print("Generating base trace with job attributes...")
    emotion_config = EmotionConfig()
    base_trace = generate_base_trace(
        num_jobs=args.num_jobs,
        emotion_config=emotion_config,
        default_service_time=args.base_service_time,
        random_seed=args.random_seed,
    )

    # Print quadrant distribution
    quadrant_counts = {'excited': 0, 'calm': 0, 'panic': 0, 'depression': 0}
    for entry in base_trace:
        quadrant = entry['russell_quadrant']
        if quadrant in quadrant_counts:
            quadrant_counts[quadrant] += 1

    print(f"  Russell quadrant distribution:")
    for quadrant in ['excited', 'calm', 'panic', 'depression']:
        count = quadrant_counts[quadrant]
        pct = count / args.num_jobs * 100
        print(f"    {quadrant}: {count} ({pct:.1f}%)")

    # Save base trace
    base_trace_path = output_dir / "unified_base_trace.json"
    with open(base_trace_path, 'w') as f:
        json.dump({
            'metadata': {
                'num_jobs': args.num_jobs,
                'random_seed': args.random_seed,
                'base_service_time': args.base_service_time,
                'description': 'Base trace without arrival times for load sweep experiments',
            },
            'trace': base_trace,
        }, f, indent=2)
    print(f"\nSaved base trace to: {base_trace_path}")

    # Generate traces for each load level
    print("\nGenerating load-specific traces...")
    for load in load_values:
        arrival_rate = load / args.base_service_time

        # Add arrival times for this load level
        load_trace = add_arrival_times(
            base_trace=base_trace,
            arrival_rate=arrival_rate,
            random_seed=args.random_seed,
        )

        # Create load directory
        load_dir = output_dir / f"load{load}"
        load_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = load_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Save trace with metadata that matches experiment.py expectations
        trace_metadata = {
            'num_jobs': 80,  # Expected num_jobs in config
            'trace_size': args.num_jobs,
            'simulation_duration': 100.0,
            'arrival_rate': arrival_rate,
            'system_load': load,
            'base_service_time': args.base_service_time,
            'default_service_time': args.base_service_time,
            'enable_emotion': True,
            'use_stratified_sampling': True,
            'random_seed': args.random_seed,
            'unified_trace': True,  # Mark as unified trace
        }

        trace_path = cache_dir / "time_window_trace.json"
        with open(trace_path, 'w') as f:
            json.dump({
                'metadata': trace_metadata,
                'trace': load_trace,
            }, f, indent=2)

        print(f"  load={load}: arrival_rate={arrival_rate:.3f}, saved to {trace_path}")

    print("\n" + "=" * 60)
    print("Unified trace generation complete!")
    print("=" * 60)
    print(f"\nNow run the experiment with:")
    print(f"  bash scripts/run_exp8_load_sweep_timewindow.sh")
    print("\nNote: The script should NOT clear the cache directories.")


if __name__ == "__main__":
    main()
