import os
import sys
from datetime import datetime

import numpy as np

from core.emotion import EmotionConfig
from workload.service_time_mapper import ServiceTimeConfig
from workload.task_generator import create_emotion_aware_jobs

from .job_config import load_pre_generated_jobs, save_job_config_if_needed
from .llm_runtime import create_scheduler, init_llm_handler, save_cache_if_needed
from .loop import run_scheduling_loop
from .reporting import report_and_save_results


class TeeLogger:
    """Write to both stdout and a log file."""

    def __init__(self, log_file, original_stdout):
        self.log_file = log_file
        self.original_stdout = original_stdout

    def write(self, message):
        self.original_stdout.write(message)
        self.log_file.write(message)

    def flush(self):
        self.original_stdout.flush()
        self.log_file.flush()


def run_emotion_aware_experiment(args) -> None:
    """
    Run a complete emotion-aware scheduling experiment with fixed job count.
    """
    # Load hierarchical configuration (YAML → env → CLI)
    from config.config_loader import load_config, get_alpha  # Local import to avoid circulars

    cli_args = {k: v for k, v in vars(args).items() if v is not None}
    config = load_config(cli_args=cli_args)

    # Setup logging to file
    output_dir = args.output_dir if args.output_dir is not None else config.output.results_dir
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(output_dir, f'experiment_{timestamp}.log')
    log_file = open(log_path, 'w')
    original_stdout = sys.stdout
    sys.stdout = TeeLogger(log_file, original_stdout)

    try:
        print("=" * 80)
        print("Emotion-aware LLM Scheduling Simulator (Fixed-Jobs Mode)")
        print("=" * 80)
        print(f"Log file: {log_path}")

        # Create configurations using loaded config
        alpha = get_alpha(config)
        emotion_config = EmotionConfig(
            arousal_noise_std=config.workload.emotion.arousal_noise_std
        )
        service_config = ServiceTimeConfig(
            base_service_time=config.workload.service_time.base_service_time,
            alpha=alpha,
            rho=config.workload.service_time.rho,
            min_service_time=config.workload.service_time.min_service_time,
        )

        # Get num_jobs from config
        num_jobs = config.experiment.num_jobs

        print(f"\nExperiment Configuration:")
        print(f"  Scheduler: {config.scheduler.algorithm}")
        print(f"  Number of jobs: {num_jobs}")
        print(f"  System load (ρ): {config.scheduler.system_load}")
        print(f"  Base service time (L_0): {config.workload.service_time.base_service_time}")
        print(f"  Alpha (α): {alpha}")
        print(f"  Experiment mode: Fixed-Jobs")

        # Calculate arrival rate from system_load
        expected_service_time = config.workload.service_time.base_service_time
        arrival_rate = config.scheduler.system_load / expected_service_time

        print(f"  Calculated arrival rate (λ): {arrival_rate:.3f} req/sec")

        # Set random seed if provided (for reproducible job generation)
        if config.experiment.random_seed is not None:
            np.random.seed(config.experiment.random_seed)
            import random

            random.seed(config.experiment.random_seed)
            print(f"  Random seed: {config.experiment.random_seed}")

        # Job configuration management (for reproducibility)
        pre_generated_jobs, _use_saved, force_new = load_pre_generated_jobs(
            config=config,
            emotion_config=emotion_config,
            service_config=service_config,
            arrival_rate=arrival_rate,
        )

        # Track if we loaded from cache (for save decision later)
        loaded_from_cache = pre_generated_jobs is not None

        # Generate jobs if not loaded from saved config
        if pre_generated_jobs is None:
            print(f"\nGenerating {num_jobs} jobs...")
            pre_generated_jobs = create_emotion_aware_jobs(
                num_jobs=num_jobs,
                arrival_rate=arrival_rate,
                emotion_config=emotion_config,
                service_time_config=service_config,
                enable_emotion=config.workload.emotion.enable_emotion_aware,
                random_seed=None,  # Seed already set above
            )
            print(f"  ✓ Generated {len(pre_generated_jobs)} jobs")

        # Create scheduler
        scheduler = create_scheduler(config)

        # Initialize LLM handler (LLM-only mode)
        llm_handler = init_llm_handler(config, alpha)

        # Run scheduling with fixed job count
        print(f"\nRunning scheduling...")
        completed_jobs, run_metrics = run_scheduling_loop(
            scheduler=scheduler,
            jobs=pre_generated_jobs,
            verbose=config.output.verbose,
            llm_handler=llm_handler,
            llm_skip_on_error=config.llm.error_handling.skip_on_error,
        )

        # Save cache if LLM was used
        save_cache_if_needed(llm_handler)

        # Save job configuration for reproducibility across runs/schedulers
        # Pass None if we didn't load from cache, so it will save
        save_job_config_if_needed(
            config=config,
            completed_jobs=completed_jobs,
            arrival_rate=arrival_rate,
            alpha=alpha,
            pre_generated_jobs=pre_generated_jobs if loaded_from_cache else None,
            force_new=force_new,
        )

        # Print and save results
        report_and_save_results(
            args=args,
            config=config,
            completed_jobs=completed_jobs,
            run_metrics=run_metrics,
            arrival_rate=arrival_rate,
        )

    finally:
        # Restore stdout and close log file
        sys.stdout = original_stdout
        log_file.close()
        print(f"Log saved to: {log_path}")


__all__ = ["run_emotion_aware_experiment"]

