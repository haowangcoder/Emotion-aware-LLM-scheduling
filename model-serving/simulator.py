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
        llm_skip_on_error: bool = True,
        pre_generated_jobs: List[Job] = None) -> tuple[List[Job], dict]:
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
        pre_generated_jobs: Optional pre-generated job list (for reproducibility)

    Returns:
        Tuple of (completed_jobs list, metrics dict)
    """
    from workload.task_generator import generate_job_on_demand

    if verbose:
        print(f"\nStarting scheduling run with {scheduler.name} scheduler")
        print(f"  Arrival rate: {arrival_rate:.3f} req/sec")
        print(f"  Simulation time: {simulation_time:.2f}s")
        if pre_generated_jobs:
            print(f"  Using pre-generated jobs: {len(pre_generated_jobs)}")

    # Scheduling state
    current_time = 0.0
    waiting_queue = []
    completed_jobs = []
    next_job_id = 0
    next_arrival_time = 0.0

    # Track jobs completed before deadline
    jobs_by_deadline = []

    # Pre-generated jobs setup
    if pre_generated_jobs:
        # Sort by arrival time to ensure correct order
        pre_generated_jobs = sorted(pre_generated_jobs, key=lambda j: j.arrival_time)
        job_index = 0
        total_jobs = len(pre_generated_jobs)
        if total_jobs > 0:
            next_arrival_time = pre_generated_jobs[0].arrival_time
    else:
        job_index = None
        total_jobs = None

    # Phase 1: Generate arrivals until simulation_time
    if verbose:
        print(f"\n=== Phase 1: Arrival Phase (0 - {simulation_time:.2f}s) ===")

    while current_time < simulation_time or waiting_queue:
        # Generate new arrivals if we're still in the arrival phase
        while current_time >= next_arrival_time and next_arrival_time < simulation_time:
            if pre_generated_jobs and job_index < total_jobs:
                # Use pre-generated job
                new_job = pre_generated_jobs[job_index]
                waiting_queue.append(new_job)

                if verbose and job_index % 50 == 0:
                    print(f"  Time {new_job.arrival_time:.2f}: Job {new_job.job_id} arrived "
                          f"(emotion: {new_job.emotion_label}, queue: {len(waiting_queue)})")

                job_index += 1
                if job_index < total_jobs:
                    next_arrival_time = pre_generated_jobs[job_index].arrival_time
                else:
                    next_arrival_time = simulation_time  # No more jobs
            else:
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

    # Only pass CLI arguments that were explicitly set (value is not None)
    cli_args = {k: v for k, v in vars(args).items() if v is not None}
    config = load_config(cli_args=cli_args)

    # Create configurations using loaded config
    alpha = get_alpha(config)
    emotion_config = EmotionConfig(arousal_noise_std=config.workload.emotion.arousal_noise_std)
    service_config = ServiceTimeConfig(
        base_service_time=config.workload.service_time.base_service_time,
        alpha=alpha,
        rho=config.workload.service_time.rho,
        min_service_time=config.workload.service_time.min_service_time,
    )

    print(f"\nExperiment Configuration:")
    print(f"  Scheduler: {config.scheduler.algorithm}")
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

    # ========================================================================
    # Job Configuration Management (for reproducibility)
    # ========================================================================
    from core.job_config_manager import JobConfigManager
    from workload.task_generator import create_emotion_aware_jobs

    # Build job config file path
    job_config_path = os.path.join(config.llm.cache.cache_dir, config.llm.cache.job_config_file)
    job_config_manager = JobConfigManager(job_config_path)

    pre_generated_jobs = None
    use_saved = config.llm.cache.use_saved_job_config
    force_new = config.llm.cache.force_new_job_config

    print(f"\nJob Configuration:")
    print(f"  Use saved config: {use_saved}")
    print(f"  Force new config: {force_new}")
    print(f"  Config file: {job_config_path}")

    # Try to load existing job configuration
    if use_saved and not force_new:
        print(f"  Attempting to load existing job configuration...")
        config_data = job_config_manager.load_job_configs()

        if config_data is not None:
            # Extract job configurations
            job_configs = config_data.get('jobs', [])

            # Generate jobs from loaded configuration
            # Note: We generate enough jobs to cover the simulation time
            # The actual number may be more/less than loaded configs
            print(f"  Generating jobs from loaded configuration...")
            pre_generated_jobs = create_emotion_aware_jobs(
                num_jobs=len(job_configs),
                arrival_rate=arrival_rate,
                emotion_config=emotion_config,
                service_time_config=service_config,
                enable_emotion=config.workload.emotion.enable_emotion_aware,
                job_configs=job_configs,
                random_seed=None  # Don't reset seed here, already set above
            )
            print(f"  ✓ Loaded {len(pre_generated_jobs)} jobs from saved configuration")
        else:
            print(f"  No existing configuration found, will generate new jobs")
    else:
        if force_new:
            print(f"  Force new config enabled, will generate new jobs")
        else:
            print(f"  Saved config disabled, will generate new jobs")

    # Create scheduler
    algorithm = config.scheduler.algorithm
    print(f"\nCreating {algorithm} scheduler...")
    if algorithm == 'FCFS':
        scheduler = FCFSScheduler()
    elif algorithm == 'SSJF-Emotion':
        scheduler = SSJFEmotionScheduler(
            starvation_threshold=config.scheduler.starvation_prevention.threshold,
            starvation_coefficient=config.scheduler.starvation_prevention.coefficient
        )
    else:
        raise ValueError(f"Unknown scheduler: {algorithm}")

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
        pre_generated_jobs=pre_generated_jobs,
    )

    # Save cache if LLM was used
    if llm_handler is not None:
        print(f"\nSaving LLM response cache...")
        llm_handler.save_cache()
        cache_stats = llm_handler.get_cache_stats()
        print(f"  Cache stats: {cache_stats['num_entries']} entries, "
              f"{cache_stats['hit_rate']:.1%} hit rate")

    # ========================================================================
    # Save Job Configuration (for reproducibility across schedulers)
    # ========================================================================
    # Save job configuration if:
    # 1. We didn't load from an existing config (pre_generated_jobs is None), OR
    # 2. force_new_job_config is True (explicitly requested to save new config)
    if pre_generated_jobs is None or force_new:
        print(f"\nSaving job configuration for future reproducibility...")

        # Prepare metadata
        job_config_metadata = {
            'num_jobs': run_metrics['total_jobs'],
            'simulation_time': config.experiment.simulation_time,
            'arrival_rate': arrival_rate,
            'system_load': config.scheduler.system_load,
            'random_seed': config.experiment.random_seed,
            'base_service_time': config.workload.service_time.base_service_time,
            'alpha': alpha,
            'rho': config.workload.service_time.rho,
            'enable_emotion_aware': config.workload.emotion.enable_emotion_aware,
            'model_name': config.llm.model.name,
        }

        # Save job configurations with actual execution times
        job_config_manager.save_job_configs(
            jobs=completed_jobs,
            metadata=job_config_metadata
        )
        print(f"  ✓ Saved {len(completed_jobs)} job configurations to {job_config_path}")
        print(f"  Note: Different schedulers can now use this config for fair comparison")
    else:
        print(f"\nSkipping job config save (loaded from existing configuration)")

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
    # Prefer explicit CLI output_dir; fall back to config.output.results_dir
    output_dir = args.output_dir if args.output_dir is not None else config.output.results_dir
    if output_dir:
        print(f"\nSaving results to: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        logger = EmotionAwareLogger(
            output_dir=output_dir,
            experiment_name=f"{config.scheduler.algorithm}_{run_metrics['total_jobs']}jobs_load{config.scheduler.system_load:.2f}_time{config.experiment.simulation_time:.0f}s"
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
    parser.add_argument('--scheduler', type=str, default=None,
                        choices=['FCFS', 'SSJF-Emotion'],
                        help='Scheduling algorithm (overrides config.scheduler.algorithm)')
    parser.add_argument('--simulation_time', type=float, default=None,
                        help='Simulation time duration in seconds')

    # System configuration
    parser.add_argument('--system_load', type=float, default=None,
                        help='Target system load (rho, overrides config.scheduler.system_load)')
    parser.add_argument('--base_service_time', type=float, default=None,
                        help='Base service time L_0 (overrides config.workload.service_time.base_service_time)')
    parser.add_argument('--alpha', type=float, default=None,
                        help='Alpha parameter for arousal-service time mapping (overrides config.workload.service_time.alpha)')
    parser.add_argument('--rho', type=float, default=None,
                        help='Correlation strength rho (overrides config.workload.service_time.rho)')

    # Job generation
    emotion_group = parser.add_mutually_exclusive_group()
    emotion_group.add_argument('--enable_emotion', dest='enable_emotion',
                               action='store_true', default=None,
                               help='Enable emotion-aware features (overrides config.workload.emotion.enable_emotion_aware)')
    emotion_group.add_argument('--disable_emotion', dest='enable_emotion',
                               action='store_false',
                               help='Disable emotion-aware features (overrides config.workload.emotion.enable_emotion_aware)')
    parser.add_argument('--arousal_noise', type=float, default=None,
                        help='Standard deviation of arousal noise (overrides config.workload.emotion.arousal_noise_std)')
    parser.add_argument('--random_seed', type=int, default=None,
                        help='Random seed for reproducibility (overrides config.experiment.random_seed)')

    # Scheduler configuration
    parser.add_argument('--starvation_threshold', type=float, default=None,
                        help='Absolute starvation threshold for SSJF (overrides config.scheduler.starvation_prevention.threshold)')
    parser.add_argument('--starvation_coefficient', type=float, default=None,
                        help='Relative starvation coefficient for SSJF (overrides config.scheduler.starvation_prevention.coefficient)')

    # Output configuration
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results (overrides config.output.results_dir)')
    parser.add_argument('--verbose', action='store_true', default=None,
                        help='Print detailed scheduling progress (overrides config.output.verbose)')

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
