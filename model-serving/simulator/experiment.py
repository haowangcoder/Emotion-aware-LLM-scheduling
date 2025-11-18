import numpy as np

from core.emotion import EmotionConfig
from workload.service_time_mapper import ServiceTimeConfig

from .job_config import load_pre_generated_jobs, save_job_config_if_needed
from .llm_runtime import create_scheduler, init_llm_handler, save_cache_if_needed
from .loop import run_scheduling_loop
from .reporting import report_and_save_results


def run_emotion_aware_experiment(args) -> None:
    """
    Run a complete emotion-aware scheduling experiment.
    """
    print("=" * 80)
    print("Emotion-aware LLM Scheduling Simulator")
    print("=" * 80)

    # Load hierarchical configuration (YAML → env → CLI)
    from config.config_loader import load_config, get_alpha  # Local import to avoid circulars

    cli_args = {k: v for k, v in vars(args).items() if v is not None}
    config = load_config(cli_args=cli_args)

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

    print(f"\nExperiment Configuration:")
    print(f"  Scheduler: {config.scheduler.algorithm}")
    print(f"  Simulation time: {config.experiment.simulation_time:.2f}s")
    print(f"  System load (ρ): {config.scheduler.system_load}")
    print(f"  Base service time (L_0): {config.workload.service_time.base_service_time}")
    print(f"  Alpha (α): {alpha}")
    print(f"  Arrival model: Fixed-rate")

    # Calculate arrival rate from system_load
    expected_service_time = config.workload.service_time.base_service_time
    arrival_rate = config.scheduler.system_load / expected_service_time

    print(f"  Calculated arrival rate (λ): {arrival_rate:.3f} req/sec")
    print(
        f"  Expected jobs: "
        f"~{int(arrival_rate * config.experiment.simulation_time)}"
    )

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

    # Create scheduler
    scheduler = create_scheduler(config)

    # Initialize LLM handler (LLM-only mode)
    llm_handler = init_llm_handler(config, alpha)

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
    save_cache_if_needed(llm_handler)

    # Save job configuration for reproducibility across runs/schedulers
    save_job_config_if_needed(
        config=config,
        completed_jobs=completed_jobs,
        arrival_rate=arrival_rate,
        alpha=alpha,
        pre_generated_jobs=pre_generated_jobs,
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


__all__ = ["run_emotion_aware_experiment"]

