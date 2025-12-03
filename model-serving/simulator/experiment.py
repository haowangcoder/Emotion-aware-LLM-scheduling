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
import json  # used to persist time-window traces across schedulers


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
        mode = config.experiment.mode
        mode_display = "Fixed-Jobs" if mode == "fixed_jobs" else "Time-Window"
        print(f"Emotion-aware LLM Scheduling Simulator ({mode_display} Mode)")
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
            emotion_correlation=config.workload.service_time.emotion_correlation,
            min_service_time=config.workload.service_time.min_service_time,
        )

        # Get experiment parameters
        mode = config.experiment.mode
        num_jobs = config.experiment.num_jobs
        simulation_duration = config.experiment.simulation_duration

        print(f"\nExperiment Configuration:")
        print(f"  Scheduler: {config.scheduler.algorithm}")
        if config.scheduler.algorithm == "SSJF-Valence":
            print(f"  Valence beta (π_i = 1 + β(-v_i)): {config.scheduler.valence_priority.beta}")
        print(f"  Experiment mode: {mode}")

        if mode == "fixed_jobs":
            print(f"  Number of jobs: {num_jobs}")
        else:  # time_window
            print(f"  Simulation duration: {simulation_duration}s")
            print(f"  Trace size: {num_jobs} jobs (pre-generated)")

        print(f"  System load (ρ): {config.scheduler.system_load}")
        print(f"  Base service time (L_0): {config.workload.service_time.base_service_time}")
        print(f"  Alpha (α): {alpha}")

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

        # Generate or load job trace
        from workload.task_generator import generate_job_trace, create_jobs_from_trace

        if mode == "fixed_jobs":
            # ------------------------------------------------------------------
            # Fixed-jobs mode: keep the original JobConfigManager behaviour.
            # ------------------------------------------------------------------
            pre_generated_jobs, _use_saved, force_new = load_pre_generated_jobs(
                config=config,
                emotion_config=emotion_config,
                service_config=service_config,
                arrival_rate=arrival_rate,
            )

            # Track if loaded from cache (used when deciding whether to save)
            loaded_from_cache = pre_generated_jobs is not None
            job_trace = None

            if pre_generated_jobs is None:
                print(f"\nGenerating job trace (fixed_jobs mode)...")

                trace_size = num_jobs
                job_trace = generate_job_trace(
                    num_jobs=trace_size,
                    arrival_rate=arrival_rate,
                    emotion_config=emotion_config,
                    service_time_config=service_config,
                    enable_emotion=config.workload.emotion.enable_emotion_aware,
                    random_seed=None,  # Seed already set globally
                    use_stratified_sampling=config.workload.emotion.use_stratified_sampling,
                    class_distribution=config.workload.emotion.class_distribution,
                )

                # Convert trace dicts to Job objects for the fixed-jobs runner
                pre_generated_jobs = create_jobs_from_trace(
                    job_trace, emotion_config, max_jobs=num_jobs
                )

                print(f"  ✓ Generated trace with {len(job_trace)} jobs")

        else:  # mode == "time_window"
            # ------------------------------------------------------------------
            # Time-window mode:
            # Use a persistent JSON trace so that all schedulers (FCFS,
            # SSJF-Emotion, etc.) see exactly the same arrival pattern.
            # ------------------------------------------------------------------
            cache_dir = os.path.join(output_dir, "cache")
            os.makedirs(cache_dir, exist_ok=True)
            trace_file = os.path.join(cache_dir, "time_window_trace.json")

            job_trace = None

            # In time-window mode we do not rely on pre_generated_jobs;
            # the scheduler loop consumes job_trace directly.
            pre_generated_jobs = []
            loaded_from_cache = False
            force_new = False  # Not used in this mode

            if os.path.exists(trace_file):
                # Reuse an existing trace (e.g., generated by the first scheduler run)
                if config.output.verbose:
                    print(f"\nLoading time-window job trace from: {trace_file}")
                with open(trace_file, "r") as f:
                    job_trace = json.load(f)
                print(f"  ✓ Loaded trace with {len(job_trace)} jobs")
            else:
                # No trace yet: generate and save one
                print(f"\nGenerating job trace (time_window mode)...")

                # Use 2x jobs to ensure the time window has enough arrivals
                trace_size = int(num_jobs * 2)
                job_trace = generate_job_trace(
                    num_jobs=trace_size,
                    arrival_rate=arrival_rate,
                    emotion_config=emotion_config,
                    service_time_config=service_config,
                    enable_emotion=config.workload.emotion.enable_emotion_aware,
                    random_seed=None,  # Seed already set globally
                    use_stratified_sampling=config.workload.emotion.use_stratified_sampling,
                    class_distribution=config.workload.emotion.class_distribution,
                )

                # 保存到 output_dir/cache/ 下
                with open(trace_file, "w") as f:
                    json.dump(job_trace, f, indent=2)

                print(
                    f"  ✓ Generated trace with {len(job_trace)} jobs "
                    f"and saved to {trace_file}"
                )

        # Create scheduler
        scheduler = create_scheduler(config)

        # Initialize LLM handler (LLM-only mode)
        llm_handler = init_llm_handler(config, alpha)

        # Run scheduling based on mode
        print(f"\nRunning scheduling ({mode} mode)...")

        if mode == "fixed_jobs":
            completed_jobs, run_metrics = run_scheduling_loop(
                scheduler=scheduler,
                jobs=pre_generated_jobs,
                verbose=config.output.verbose,
                llm_handler=llm_handler,
                llm_skip_on_error=config.llm.error_handling.skip_on_error,
            )
        else:  # time_window
            from simulator.loop import run_scheduling_loop_time_window
            
            completed_jobs, run_metrics = run_scheduling_loop_time_window(
                scheduler=scheduler,
                job_trace=job_trace,
                simulation_duration=simulation_duration,
                emotion_config=emotion_config,
                verbose=config.output.verbose,
                llm_handler=llm_handler,
                llm_skip_on_error=config.llm.error_handling.skip_on_error,
            )

        # Save cache if LLM was used
        save_cache_if_needed(llm_handler)

        # Save job configuration only for fixed-jobs mode.
        # Time-window mode uses a dedicated JSON trace file instead.
        if mode == "fixed_jobs":
            save_job_config_if_needed(
                config=config,
                completed_jobs=completed_jobs,
                arrival_rate=arrival_rate,
                alpha=alpha,
                pre_generated_jobs=pre_generated_jobs if loaded_from_cache else None,
                force_new=force_new,
            )
        else:
            if config.output.verbose:
                print("\nSkipping job config save in time_window mode "
                      "(trace is stored in results/cache/time_window_trace.json).")

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
