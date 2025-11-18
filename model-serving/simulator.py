"""
Standalone Emotion-aware LLM Scheduling Runner

This runner implements a fixed-rate arrival simulation model:
- Jobs arrive continuously at rate λ = system_load / E[S] req/sec
- Simulation runs for a fixed duration (simulation_time)
- After deadline T, new arrivals stop and remaining queue drains
- Metrics track: jobs completed by deadline vs total jobs

Integrated components:
- Emotion sampling and arousal mapping
- Service time mapping based on arousal
- On-demand job generation (no pre-generated job list)
- FCFS and SSJF-Emotion scheduling strategies
- Fairness metrics and comprehensive logging

Usage:
    python simulator.py --scheduler FCFS --simulation_time 200 --system_load 0.6
    python simulator.py --scheduler SSJF-Emotion --simulation_time 300 --system_load 0.8
"""

import argparse
import os
import sys
import numpy as np
from typing import List

from core.emotion import EmotionConfig
from workload.service_time_mapper import ServiceTimeConfig
from core.scheduler_base import FCFSScheduler
from core.ssjf_emotion import SSJFEmotionScheduler
from analysis.logger import EmotionAwareLogger
from analysis.fairness_metrics import analyze_fairness_comprehensive
from core.job import Job


def run_scheduling_loop(
        scheduler,
        arrival_rate: float,
        simulation_time: float,
        emotion_config: EmotionConfig,
        service_time_config: ServiceTimeConfig,
        enable_emotion: bool = True,
        verbose: bool = False,
        llm_handler=None,
        llm_skip_on_error: bool = True) -> tuple[List[Job], dict]:
    """
    Run emotion-aware job scheduling loop with fixed-rate arrivals

    Args:
        scheduler: Scheduler instance
        arrival_rate: Arrival rate λ (requests/sec)
        simulation_time: Time limit for new arrivals (seconds)
        emotion_config: EmotionConfig for job generation
        service_time_config: ServiceTimeConfig for job generation
        enable_emotion: Whether to enable emotion-aware features
        verbose: Print detailed progress
        llm_handler: Optional LLMInferenceHandler for real model inference
        llm_skip_on_error: Skip failed LLM jobs instead of aborting

    Returns:
        Tuple of (completed_jobs list, metrics dict)
    """
    from workload.task_generator import generate_job_on_demand

    if verbose:
        print(f"\nStarting scheduling run with {scheduler.name} scheduler")
        print(f"  Arrival rate: {arrival_rate:.3f} req/sec")
        print(f"  Simulation time: {simulation_time:.2f}s")

    # Scheduling state
    current_time = 0.0
    waiting_queue = []
    completed_jobs = []
    next_job_id = 0
    next_arrival_time = 0.0

    # Track jobs completed before deadline
    jobs_by_deadline = []

    # Phase 1: Generate arrivals until simulation_time
    if verbose:
        print(f"\n=== Phase 1: Arrival Phase (0 - {simulation_time:.2f}s) ===")

    while current_time < simulation_time or waiting_queue:
        # Generate new arrivals if we're still in the arrival phase
        while current_time >= next_arrival_time and next_arrival_time < simulation_time:
            # Generate job on-demand
            new_job = generate_job_on_demand(
                job_id=next_job_id,
                arrival_time=next_arrival_time,
                emotion_config=emotion_config,
                service_time_config=service_time_config,
                enable_emotion=enable_emotion
            )
            waiting_queue.append(new_job)

            if verbose and next_job_id % 50 == 0:
                print(f"  Time {next_arrival_time:.2f}: Job {next_job_id} arrived "
                      f"(emotion: {new_job.emotion_label}, queue: {len(waiting_queue)})")

            # Schedule next arrival using exponential distribution
            inter_arrival_time = np.random.exponential(1.0 / arrival_rate)
            next_arrival_time += inter_arrival_time
            next_job_id += 1

        # Check if we've transitioned to drain phase
        if current_time >= simulation_time and len(jobs_by_deadline) == 0:
            jobs_by_deadline = completed_jobs.copy()
            if verbose:
                print(f"\n=== Phase 2: Drain Phase (starting at {current_time:.2f}s) ===")
                print(f"  Jobs completed by deadline: {len(jobs_by_deadline)}")
                print(f"  Jobs still waiting: {len(waiting_queue)}")

        # If queue is empty, advance time
        if not waiting_queue:
            if next_arrival_time < simulation_time:
                # Jump to next arrival
                current_time = next_arrival_time
            else:
                # No more arrivals and queue empty - done
                break
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

            if not success and llm_skip_on_error:
                # Skip this job if it failed
                if verbose:
                    print(f"  WARNING: Job {selected_job.job_id} failed, skipping: {selected_job.error_msg}")
                continue
            elif not success:
                # Fail entire run
                print(f"ERROR: Job {selected_job.job_id} failed: {selected_job.error_msg}")
                print("Set LLM_SKIP_ON_ERROR=True to skip failed jobs instead")
                break

        # Advance time (uses actual LLM time if llm_handler was used)
        current_time += selected_job.execution_duration
        selected_job.completion_time = current_time
        selected_job.status = 'COMPLETED'

        scheduler.on_job_completed(selected_job, current_time)
        completed_jobs.append(selected_job)

    # Calculate metrics for both phases
    total_jobs = len(completed_jobs)
    jobs_by_deadline_count = len(jobs_by_deadline) if jobs_by_deadline else total_jobs
    jobs_after_deadline = total_jobs - jobs_by_deadline_count

    metrics = {
        'total_jobs': total_jobs,
        'jobs_by_deadline': jobs_by_deadline_count,
        'jobs_after_deadline': jobs_after_deadline,
        'simulation_time': simulation_time,
        'total_time': current_time,
        'effective_throughput': jobs_by_deadline_count / simulation_time if simulation_time > 0 else 0,
        'total_throughput': total_jobs / current_time if current_time > 0 else 0,
        'arrival_rate': arrival_rate,
    }

    if verbose:
        print(f"\n=== Run Completed ===")
        print(f"  Total simulation time: {current_time:.2f}s")
        print(f"  Jobs completed by deadline: {jobs_by_deadline_count}")
        print(f"  Jobs completed after deadline: {jobs_after_deadline}")
        print(f"  Total completed jobs: {total_jobs}")
        print(f"  Effective throughput: {metrics['effective_throughput']:.3f} jobs/sec")

    return completed_jobs, metrics


def run_emotion_aware_experiment(args):
    """
    Run a complete emotion-aware scheduling experiment

    Args:
        args: Command-line arguments
    """
    print("=" * 80)
    print("Emotion-aware LLM Scheduling Simulator")
    print("=" * 80)

    # Load hierarchical configuration (YAML → env → CLI)
    from config.config_loader import load_config, get_alpha  # Local import to avoid circulars

    cli_args = vars(args)
    config = load_config(cli_args=cli_args)

    # Create configurations using loaded config
    alpha = get_alpha(config)
    emotion_config = EmotionConfig(arousal_noise_std=config.workload.emotion.arousal_noise_std)
    service_config = ServiceTimeConfig(
        base_service_time=config.workload.service_time.base_service_time,
        alpha=alpha,
        rho=config.workload.service_time.rho
    )

    print(f"\nExperiment Configuration:")
    print(f"  Scheduler: {args.scheduler}")
    print(f"  Simulation time: {config.experiment.simulation_time:.2f}s")
    print(f"  System load (ρ): {config.scheduler.system_load}")
    print(f"  Base service time (L_0): {config.workload.service_time.base_service_time}")
    print(f"  Alpha (α): {alpha}")
    print(f"  Arrival model: Fixed-rate")

    # Calculate arrival rate from system_load
    # λ = ρ / E[S] where E[S] is expected service time
    expected_service_time = config.workload.service_time.base_service_time
    arrival_rate = config.scheduler.system_load / expected_service_time

    print(f"  Calculated arrival rate (λ): {arrival_rate:.3f} req/sec")
    print(f"  Expected jobs: ~{int(arrival_rate * config.experiment.simulation_time)}")

    # Set random seed if provided (for reproducible job generation)
    if config.experiment.random_seed is not None:
        np.random.seed(config.experiment.random_seed)
        import random
        random.seed(config.experiment.random_seed)
        print(f"  Random seed: {config.experiment.random_seed}")

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
    print(f"  Model: {config.llm.model.name}")
    print(f"  Dataset: {config.dataset.emotion_dataset_path}")
    print(f"  Cache: {'Enabled' if config.llm.cache.use_response_cache else 'Disabled'}")

    try:
        from llm.inference_handler import LLMInferenceHandler

        llm_handler = LLMInferenceHandler(
            model_name=config.llm.model.name,
            dataset_path=config.dataset.emotion_dataset_path,
            cache_path=os.path.join(config.llm.cache.cache_dir, config.llm.cache.cache_file),
            use_cache=config.llm.cache.use_response_cache,
            force_regenerate=config.llm.cache.force_regenerate,
            device_map=config.llm.model.device_map,
            dtype=config.llm.model.dtype,
            load_in_8bit=config.llm.model.load_in_8bit,
            include_emotion_hint=config.llm.prompt.include_emotion_hint,
            enable_emotion_length_control=config.llm.prompt.emotion_length_control.enabled,
            base_response_length=config.llm.prompt.emotion_length_control.base_response_length,
            alpha=alpha,
            max_conversation_turns=config.llm.prompt.max_conversation_turns,
            max_new_tokens=config.llm.generation.max_new_tokens,
            temperature=config.llm.generation.temperature,
            top_p=config.llm.generation.top_p,
            do_sample=config.llm.generation.do_sample,
            repetition_penalty=config.llm.generation.repetition_penalty,
            max_retries=config.llm.error_handling.max_retries,
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
    completed_jobs, run_metrics = run_scheduling_loop(
        scheduler=scheduler,
        arrival_rate=arrival_rate,
        simulation_time=config.experiment.simulation_time,
        emotion_config=emotion_config,
        service_time_config=service_config,
        enable_emotion=config.workload.emotion.enable_emotion_aware,
        verbose=config.output.verbose,
        llm_handler=llm_handler,
        llm_skip_on_error=config.llm.error_handling.skip_on_error,
    )

    # Save cache if LLM was used
    if llm_handler is not None:
        print(f"\nSaving LLM response cache...")
        llm_handler.save_cache()
        cache_stats = llm_handler.get_cache_stats()
        print(f"  Cache stats: {cache_stats['num_entries']} entries, "
              f"{cache_stats['hit_rate']:.1%} hit rate")

    # Calculate metrics
    waiting_times = [j.waiting_duration for j in completed_jobs if j.waiting_duration is not None]
    turnaround_times = [(j.completion_time - j.arrival_time) for j in completed_jobs]

    print(f"\n" + "=" * 80)
    print("Results (Fixed-Rate Arrival)")
    print("=" * 80)

    print(f"\nOverall Performance Metrics:")
    print(f"  Total completed jobs: {run_metrics['total_jobs']}")
    print(f"  Jobs by deadline ({run_metrics['simulation_time']:.2f}s): {run_metrics['jobs_by_deadline']}")
    print(f"  Jobs after deadline: {run_metrics['jobs_after_deadline']}")
    print(f"  Total run time: {run_metrics['total_time']:.2f}s")
    print(f"  Effective throughput (by deadline): {run_metrics['effective_throughput']:.3f} jobs/sec")
    print(f"  Total throughput: {run_metrics['total_throughput']:.3f} jobs/sec")
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
            experiment_name=f"{args.scheduler}_{run_metrics['total_jobs']}jobs_load{config.scheduler.system_load:.2f}_time{config.experiment.simulation_time:.0f}s"
        )

        # Set metadata
        metadata = vars(args)
        metadata['arrival_rate'] = arrival_rate
        metadata['run_metrics'] = run_metrics
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
    parser.add_argument('--simulation_time', type=float, default=None,
                        help='Simulation time duration in seconds')

    # System configuration
    parser.add_argument('--system_load', type=float, default=0.6,
                        help='Target system load (rho)')
    parser.add_argument('--base_service_time', type=float, default=2.0,
                        help='Base service time (L_0)')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Alpha parameter for arousal-service time mapping')
    parser.add_argument('--rho', type=float, default=1.0,
                        help='Correlation strength (rho)')

    # Job generation
    parser.add_argument('--enable_emotion', action='store_true', default=True,
                        help='Enable emotion-aware features')
    parser.add_argument('--arousal_noise', type=float, default=0.0,
                        help='Standard deviation of arousal noise')
    parser.add_argument('--random_seed', type=int, default=None,
                        help='Random seed for reproducibility (default: None, uses system entropy)')

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
    # Note: These are loaded from config by default
    parser.add_argument('--model_name', type=str, default=None,
                        help='HuggingFace model identifier (overrides config)')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to EmpatheticDialogues dataset (overrides config)')
    parser.add_argument('--use_cache', action='store_true', default=None,
                        help='Enable response caching (overrides config)')
    parser.add_argument('--force_regenerate', action='store_true', default=None,
                        help='Force regenerate responses (overrides config)')
    parser.add_argument('--device_map', type=str, default=None,
                        help='Device mapping for model (overrides config)')
    parser.add_argument('--dtype', type=str, default=None,
                        help='Data type for model weights (overrides config)')
    parser.add_argument('--load_in_8bit', action='store_true', default=None,
                        help='Use 8-bit quantization (overrides config)')

    args = parser.parse_args()

    # Run experiment
    run_emotion_aware_experiment(args)


if __name__ == '__main__':
    main()
