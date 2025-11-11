"""
Standalone Emotion-aware LLM Scheduling Runner

This runner provides a simple command-line interface for executing emotion-aware
scheduling experiments. It integrates all the emotion-aware components:
- Emotion sampling and arousal mapping
- Service time mapping based on arousal
- FCFS and SSJF-Emotion scheduling strategies
- Fairness metrics and comprehensive logging

Usage:
    python simulator_emotion.py --scheduler FCFS --num_jobs 100 --system_load 0.6
    python simulator_emotion.py --scheduler SSJF-Emotion --num_jobs 200 --alpha 0.7
"""

import argparse
import os
import sys
from typing import List

from core.emotion import EmotionConfig
from workload.service_time_mapper import ServiceTimeConfig
from workload.task_generator import create_emotion_aware_jobs, get_emotion_aware_statistics
from core.scheduler_base import FCFSScheduler
from core.ssjf_emotion import SSJFEmotionScheduler
from analysis.logger import EmotionAwareLogger
from analysis.fairness_metrics import analyze_fairness_comprehensive
from core.job import Job
from config import *


def run_scheduling_loop(
        scheduler,
        job_list: List[Job],
        verbose: bool = False,
        llm_handler=None) -> List[Job]:
    """
    Run emotion-aware job scheduling loop

    Args:
        scheduler: Scheduler instance
        job_list: List of Job objects
        verbose: Print detailed progress
        llm_handler: Optional LLMInferenceHandler for real model inference

    Returns:
        List of completed jobs
    """
    if verbose:
        print(f"\nStarting scheduling run with {scheduler.name} scheduler")
        print(f"Initial queue size: {len(job_list)}")

    # Sort jobs by arrival time
    job_list = sorted(job_list, key=lambda j: j.arrival_time)

    # Scheduling state
    current_time = 0
    waiting_queue = []
    completed_jobs = []
    job_index = 0

    # Main scheduling loop
    while job_index < len(job_list) or waiting_queue:
        # Add newly arrived jobs to waiting queue
        while job_index < len(job_list) and job_list[job_index].arrival_time <= current_time:
            job = job_list[job_index]
            waiting_queue.append(job)
            if verbose and job_index % 50 == 0:
                print(f"  Time {current_time:.2f}: Added job {job.job_id} to queue (queue size: {len(waiting_queue)})")
            job_index += 1

        # If queue is empty, advance time to next arrival
        if not waiting_queue:
            if job_index < len(job_list):
                current_time = job_list[job_index].arrival_time
            continue

        # Schedule next job
        selected_job = scheduler.schedule(waiting_queue, current_time=current_time)

        if selected_job is None:
            print(f"Warning: Scheduler returned None with non-empty queue at time {current_time}")
            break

        # Remove from queue
        waiting_queue.remove(selected_job)

        # Execute job
        selected_job.status = 'RUNNING'
        selected_job.waiting_duration = current_time - selected_job.arrival_time
        scheduler.on_job_scheduled(selected_job, current_time)

        # Use real LLM inference if handler provided
        if llm_handler is not None:
            if verbose and len(completed_jobs) % 10 == 0:
                print(f"  Time {current_time:.2f}: Executing job {selected_job.job_id} with LLM "
                      f"(emotion: {selected_job.emotion_label}, predicted: {selected_job.execution_duration:.2f}s)")

            # Execute with real LLM model
            success = llm_handler.execute_job(selected_job)

            if not success and LLM_SKIP_ON_ERROR:
                # Skip this job if it failed
                if verbose:
                    print(f"  WARNING: Job {selected_job.job_id} failed, skipping: {selected_job.error_msg}")
                continue
            elif not success:
                # Fail entire run
                print(f"ERROR: Job {selected_job.job_id} failed: {selected_job.error_msg}")
                print("Set LLM_SKIP_ON_ERROR=True to skip failed jobs instead")
                break

            # Note: llm_handler.execute_job() updates selected_job.execution_duration
            # with actual inference time, so we use that for the scheduling clock
        else:
            # In LLM-only mode this branch is not used
            pass

        # Advance time (uses actual LLM time if llm_handler was used)
        current_time += selected_job.execution_duration
        selected_job.completion_time = current_time
        selected_job.status = 'COMPLETED'

        scheduler.on_job_completed(selected_job, current_time)
        completed_jobs.append(selected_job)

    if verbose:
        print(f"\nRun completed at time {current_time:.2f}")
        print(f"Completed jobs: {len(completed_jobs)}")

    return completed_jobs


def run_emotion_aware_experiment(args):
    """
    Run a complete emotion-aware scheduling experiment

    Args:
        args: Command-line arguments
    """
    print("=" * 80)
    print("Emotion-aware LLM Scheduling Simulator")
    print("=" * 80)

    # Create configurations
    emotion_config = EmotionConfig(arousal_noise_std=args.arousal_noise)
    service_config = ServiceTimeConfig(
        base_service_time=args.base_service_time,
        alpha=args.alpha,
        gamma=1.0,
        rho=args.rho,
        mapping_func=args.mapping_func
    )

    print(f"\nExperiment Configuration:")
    print(f"  Scheduler: {args.scheduler}")
    print(f"  Number of jobs: {args.num_jobs}")
    print(f"  System load (ρ): {args.system_load}")
    print(f"  Base service time (L_0): {args.base_service_time}")
    print(f"  Alpha (α): {args.alpha}")
    print(f"  Gamma (γ): {args.gamma}")
    print(f"  Distribution: {args.distribution}")

    # Calculate arrival rate
    expected_service_time = args.base_service_time
    arrival_rate = args.system_load / expected_service_time

    print(f"  Calculated arrival rate (λ): {arrival_rate:.3f}")

    # Generate jobs
    print(f"\nGenerating {args.num_jobs} emotion-aware jobs...")
    jobs = create_emotion_aware_jobs(
        num_jobs=args.num_jobs,
        arrival_rate=arrival_rate,
        distribution=args.distribution,
        emotion_config=emotion_config,
        service_time_config=service_config,
        gamma=args.gamma,
        enable_emotion=args.enable_emotion
    )

    # Print job statistics
    job_stats = get_emotion_aware_statistics(jobs)
    print(f"\nJob Generation Statistics:")
    print(f"  Total jobs: {job_stats['num_jobs']}")
    print(f"  Arousal mean: {job_stats['arousal_mean']:.3f}")
    print(f"  Arousal std: {job_stats['arousal_std']:.3f}")
    print(f"  Service time mean: {job_stats['service_time_mean']:.3f} ± {job_stats['service_time_std']:.3f}")
    print(f"  Service time range: [{job_stats['service_time_min']:.3f}, {job_stats['service_time_max']:.3f}]")
    print(f"  Emotion class distribution:")
    for emotion_class, count in job_stats['emotion_class_counts'].items():
        pct = (count / job_stats['num_jobs']) * 100
        print(f"    {emotion_class}: {count} ({pct:.1f}%)")

    # Create scheduler
    print(f"\nCreating {args.scheduler} scheduler...")
    if args.scheduler == 'FCFS':
        scheduler = FCFSScheduler()
    elif args.scheduler == 'SSJF-Emotion':
        scheduler = SSJFEmotionScheduler(
            starvation_threshold=args.starvation_threshold,
            starvation_coefficient=args.starvation_coefficient
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")

    # Initialize LLM handler (LLM-only mode)
    llm_handler = None
    print(f"\nInitializing LLM Inference Handler...")
    print(f"  Model: {args.model_name}")
    print(f"  Dataset: {args.dataset_path}")
    print(f"  Cache: {'Enabled' if args.use_cache else 'Disabled'}")

    try:
        from llm.inference_handler import LLMInferenceHandler

        llm_handler = LLMInferenceHandler(
            model_name=args.model_name,
            dataset_path=args.dataset_path,
            cache_path=args.cache_path,
            use_cache=args.use_cache,
            force_regenerate=args.force_regenerate,
            device_map=args.device_map,
            dtype=args.dtype,
            load_in_8bit=args.load_in_8bit
        )

        print(f"  LLM handler initialized successfully")
        model_info = llm_handler.get_model_info()
        print(f"  Model loaded on device: {model_info['device']}")

    except Exception as e:
        print(f"ERROR: Failed to initialize LLM handler: {e}")
        print("LLM-only mode is enabled; no fallback is available.")
        print("Please fix the LLM setup (model, dataset, or device settings) and retry.")
        sys.exit(1)

    # Run scheduling
    print(f"\nRunning scheduling...")
    completed_jobs = run_scheduling_loop(
        scheduler, jobs, verbose=args.verbose, llm_handler=llm_handler
    )

    # Save cache if LLM was used
    if llm_handler is not None:
        print(f"\nSaving LLM response cache...")
        llm_handler.save_cache()
        cache_stats = llm_handler.get_cache_stats()
        print(f"  Cache stats: {cache_stats['num_entries']} entries, "
              f"{cache_stats['hit_rate']:.1%} hit rate")

    # Calculate metrics
    import numpy as np
    waiting_times = [j.waiting_duration for j in completed_jobs if j.waiting_duration is not None]
    turnaround_times = [(j.completion_time - j.arrival_time) for j in completed_jobs]
    service_times = [j.execution_duration for j in completed_jobs]

    print(f"\n" + "=" * 80)
    print("Results")
    print("=" * 80)

    print(f"\nOverall Performance Metrics:")
    print(f"  Completed jobs: {len(completed_jobs)}")
    print(f"  Total run time: {max([j.completion_time for j in completed_jobs]):.2f}")
    print(f"  Throughput: {len(completed_jobs) / max([j.completion_time for j in completed_jobs]):.3f} jobs/sec")
    print(f"\nLatency Metrics:")
    print(f"  Avg waiting time: {np.mean(waiting_times):.3f}")
    print(f"  Std waiting time: {np.std(waiting_times):.3f}")
    print(f"  P50 waiting time: {np.percentile(waiting_times, 50):.3f}")
    print(f"  P95 waiting time: {np.percentile(waiting_times, 95):.3f}")
    print(f"  P99 waiting time: {np.percentile(waiting_times, 99):.3f}")
    print(f"\nTurnaround Time:")
    print(f"  Avg: {np.mean(turnaround_times):.3f}")
    print(f"  P99: {np.percentile(turnaround_times, 99):.3f}")

    # Fairness analysis
    print(f"\nFairness Analysis:")
    fairness_analysis = analyze_fairness_comprehensive(completed_jobs)

    waiting_fairness = fairness_analysis['waiting_time_fairness']
    print(f"  Waiting Time Fairness:")
    print(f"    Jain Index: {waiting_fairness['jain_index']:.4f}")
    print(f"    CV: {waiting_fairness['coefficient_of_variation']:.4f}")
    print(f"    Max/Min Ratio: {waiting_fairness['max_min_ratio']:.4f}")

    print(f"\n  Per-Emotion-Class Waiting Time:")
    for emotion_class, avg_wait in waiting_fairness['per_class_values'].items():
        print(f"    {emotion_class}: {avg_wait:.3f}")

    # Per-class metrics
    per_class = fairness_analysis['per_class_metrics']
    print(f"\nPer-Emotion-Class Detailed Metrics:")
    print(f"  {'Class':<10} {'Count':<8} {'Avg Wait':<12} {'P99 Wait':<12} {'Avg Service':<12}")
    print(f"  {'-'*60}")
    for emotion_class, metrics in per_class.items():
        if emotion_class != 'overall':
            print(f"  {emotion_class:<10} {metrics['count']:<8} "
                  f"{metrics['avg_waiting_time']:<12.3f} {metrics['p99_waiting_time']:<12.3f} "
                  f"{metrics['avg_execution_time']:<12.3f}")

    # Save results
    if args.output_dir:
        print(f"\nSaving results to: {args.output_dir}")
        os.makedirs(args.output_dir, exist_ok=True)

        logger = EmotionAwareLogger(
            output_dir=args.output_dir,
            experiment_name=f"{args.scheduler}_{args.num_jobs}jobs_load{args.system_load:.2f}"
        )

        # Set metadata
        metadata = vars(args)
        metadata['arrival_rate'] = arrival_rate
        metadata['job_stats'] = job_stats
        logger.set_metadata(metadata)

        # Log and save
        logger.log_jobs_batch(completed_jobs)
        logger.save_job_logs()
        logger.save_summary_statistics(completed_jobs)

        print(f"Results saved successfully!")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Emotion-aware LLM Scheduling Runner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--scheduler', type=str, default='FCFS',
                        choices=['FCFS', 'SSJF-Emotion'],
                        help='Scheduling algorithm')
    parser.add_argument('--num_jobs', type=int, default=100,
                        help='Number of jobs to run')

    # System configuration
    parser.add_argument('--system_load', type=float, default=0.6,
                        help='Target system load (rho)')
    parser.add_argument('--base_service_time', type=float, default=2.0,
                        help='Base service time (L_0)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Alpha parameter for arousal-service time mapping')
    parser.add_argument('--gamma', type=float, default=0.3,
                        help='Gamma parameter for arousal-arrival rate mapping')
    parser.add_argument('--rho', type=float, default=1.0,
                        help='Correlation strength (rho)')

    # Job generation
    parser.add_argument('--distribution', type=str, default='poisson',
                        choices=['poisson', 'uniform', 'gamma'],
                        help='Arrival distribution')
    parser.add_argument('--enable_emotion', action='store_true', default=True,
                        help='Enable emotion-aware features')
    parser.add_argument('--arousal_noise', type=float, default=0.0,
                        help='Standard deviation of arousal noise')
    parser.add_argument('--mapping_func', type=str, default='linear',
                        choices=['linear', 'exponential', 'gamma_dist'],
                        help='Service time mapping function')

    # Scheduler configuration
    parser.add_argument('--starvation_threshold', type=float, default=float('inf'),
                        help='Absolute starvation threshold for SSJF')
    parser.add_argument('--starvation_coefficient', type=float, default=3.0,
                        help='Relative starvation coefficient for SSJF')

    # Output configuration
    parser.add_argument('--output_dir', type=str, default='results/llm_runs/',
                        help='Directory to save results')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed scheduling progress')

    # LLM Inference Configuration (LLM-only)
    parser.add_argument('--model_name', type=str, default=LLM_MODEL_NAME,
                        help='HuggingFace model identifier')
    parser.add_argument('--dataset_path', type=str, default=EMOTION_DATASET_PATH,
                        help='Path to EmpatheticDialogues dataset')
    parser.add_argument('--cache_path', type=str, default=RESPONSE_CACHE_PATH,
                        help='Path to response cache file')
    parser.add_argument('--use_cache', action='store_true', default=USE_RESPONSE_CACHE,
                        help='Enable response caching')
    parser.add_argument('--force_regenerate', action='store_true', default=FORCE_REGENERATE,
                        help='Force regenerate responses (ignore cache)')
    parser.add_argument('--device_map', type=str, default=LLM_DEVICE_MAP,
                        help='Device mapping for model (auto, cuda, cpu)')
    parser.add_argument('--dtype', type=str, default=LLM_DTYPE,
                        help='Data type for model weights (auto, float16, bfloat16, float32)')
    parser.add_argument('--load_in_8bit', action='store_true', default=LLM_LOAD_IN_8BIT,
                        help='Use 8-bit quantization')

    args = parser.parse_args()

    # Run experiment
    run_emotion_aware_experiment(args)


if __name__ == '__main__':
    main()
