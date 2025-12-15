import os
import sys
from datetime import datetime

import numpy as np

from core.emotion import EmotionConfig
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
    Run a complete affect-aware scheduling experiment.
    """
    # Load hierarchical configuration (YAML → env → CLI)
    from config.config_loader import load_config, get_affect_weight_params

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
        print(f"Affect-Aware LLM Scheduling Simulator ({mode_display} Mode)")
        print("=" * 80)
        print(f"Log file: {log_path}")

        # Create configurations using loaded config
        emotion_config = EmotionConfig(
            arousal_noise_std=config.workload.emotion.arousal_noise_std
        )

        # Get experiment parameters
        mode = config.experiment.mode
        num_jobs = config.experiment.num_jobs
        simulation_duration = config.experiment.simulation_duration

        print(f"\nExperiment Configuration:")
        print(f"  Scheduler: {config.scheduler.algorithm}")
        if config.scheduler.algorithm in ("AW-SSJF", "Weight-Only"):
            affect_cfg = config.scheduler.affect_weight
            print(f"  Affect Weight: w_max={affect_cfg.w_max}, p={affect_cfg.p}, q={affect_cfg.q}")
        print(f"  Experiment mode: {mode}")

        if mode == "fixed_jobs":
            print(f"  Number of jobs: {num_jobs}")
        else:  # time_window
            print(f"  Simulation duration: {simulation_duration}s")
            print(f"  Trace size: {num_jobs} jobs (pre-generated)")

        print(f"  System load (ρ): {config.scheduler.system_load}")
        print(f"  Base service time (L_0): {config.workload.service_time.base_service_time}")

        # Calculate arrival rate from system_load
        expected_service_time = config.workload.service_time.base_service_time
        arrival_rate = config.scheduler.system_load / expected_service_time

        print(f"  Calculated arrival rate (λ): {arrival_rate:.3f} req/sec")

        # MMPP configuration
        mmpp_config = config.workload.mmpp
        if mmpp_config.enabled:
            print(f"\n  MMPP (burst traffic) enabled:")
            print(f"    lambda_high: {mmpp_config.lambda_high}")
            print(f"    lambda_low: {mmpp_config.lambda_low}")
            print(f"    alpha: {mmpp_config.alpha} (mean burst duration: {1/mmpp_config.alpha:.1f}s)")
            print(f"    beta: {mmpp_config.beta} (mean normal duration: {1/mmpp_config.beta:.1f}s)")
            print(f"    burst intensity: {mmpp_config.lambda_high/mmpp_config.lambda_low:.1f}x")

        # Set random seed if provided (for reproducible job generation)
        if config.experiment.random_seed is not None:
            np.random.seed(config.experiment.random_seed)
            import random

            random.seed(config.experiment.random_seed)
            print(f"  Random seed: {config.experiment.random_seed}")

        # Generate or load job trace
        from workload.task_generator import generate_job_trace, create_jobs_from_trace

        # Get default service time from length predictor config
        default_service_time = config.length_predictor.default_service_time

        # === Initialize Length Predictor (for early service time prediction) ===
        early_prompt_generator = None
        length_estimator = None

        predictor_disabled = getattr(config.length_predictor, 'disabled', False)

        if predictor_disabled:
            print("\nLength predictor disabled via flag; skipping initialization.")
        elif config.length_predictor.enabled:
            print(f"\nInitializing BERT Bucket Predictor...")
            print(f"  Model path: {config.length_predictor.model_path}")
            print(f"  Bin edges: {config.length_predictor.bin_edges_path}")

            from predictor.length_estimator import create_length_estimator

            length_estimator = create_length_estimator({
                'enabled': config.length_predictor.enabled,
                'model_path': config.length_predictor.model_path,
                'bin_edges_path': config.length_predictor.bin_edges_path,
                'model_name': config.length_predictor.model_name,
                'device': config.length_predictor.device,
                'per_token_latency': config.length_predictor.per_token_latency,
                'const_latency': config.length_predictor.const_latency,
                'default_service_time': config.length_predictor.default_service_time,
            })

            if length_estimator.is_available():
                print(f"  ✓ BERT bucket predictor loaded successfully")
            else:
                print(f"  ⚠ Predictor not available, using default service time")

        if mode == "fixed_jobs":
            # ------------------------------------------------------------------
            # Fixed-jobs mode: keep the original JobConfigManager behaviour.
            # ------------------------------------------------------------------
            pre_generated_jobs, _use_saved, force_new = load_pre_generated_jobs(
                config=config,
                emotion_config=emotion_config,
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
                    default_service_time=default_service_time,
                    enable_emotion=config.workload.emotion.enable_emotion_aware,
                    random_seed=None,  # Seed already set globally
                    use_stratified_sampling=config.workload.emotion.use_stratified_sampling,
                    # MMPP parameters
                    mmpp_enabled=mmpp_config.enabled,
                    mmpp_lambda_high=mmpp_config.lambda_high,
                    mmpp_lambda_low=mmpp_config.lambda_low,
                    mmpp_alpha=mmpp_config.alpha,
                    mmpp_beta=mmpp_config.beta,
                )

                # Convert trace dicts to Job objects for the fixed-jobs runner
                pre_generated_jobs = create_jobs_from_trace(
                    job_trace, emotion_config, max_jobs=num_jobs
                )

                print(f"  ✓ Generated trace with {len(job_trace)} jobs")

        else:  # mode == "time_window"
            # ------------------------------------------------------------------
            # Time-window mode:
            # Use a persistent JSON trace so that all schedulers see the same
            # arrival pattern.
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

            # Metadata used to validate cache correctness across runs
            trace_size = int(num_jobs * 2)  # ensure enough arrivals for the window
            trace_metadata = {
                "num_jobs": num_jobs,
                "trace_size": trace_size,
                "simulation_duration": simulation_duration,
                "arrival_rate": arrival_rate,
                "system_load": config.scheduler.system_load,
                "base_service_time": config.workload.service_time.base_service_time,
                "default_service_time": default_service_time,
                "enable_emotion": config.workload.emotion.enable_emotion_aware,
                "use_stratified_sampling": config.workload.emotion.use_stratified_sampling,
                "random_seed": config.experiment.random_seed,
            }

            def _load_time_window_trace(path):
                with open(path, "r") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "trace" in data:
                    return data.get("trace", []), data.get("metadata", {})
                return data, {}

            def _trace_matches(cached_meta, expected_meta):
                if not cached_meta:
                    return False
                # If this is a unified trace (for load sweep experiments),
                # skip validation of load-dependent fields
                if cached_meta.get('unified_trace', False):
                    skip_keys = {'arrival_rate', 'system_load'}
                    for key, val in expected_meta.items():
                        if key in skip_keys:
                            continue
                        if cached_meta.get(key) != val:
                            return False
                    return True
                # Standard validation: all fields must match
                for key, val in expected_meta.items():
                    if cached_meta.get(key) != val:
                        return False
                return True

            def _save_time_window_trace(path, trace, metadata):
                with open(path, "w") as f:
                    json.dump({"metadata": metadata, "trace": trace}, f, indent=2)

            cache_valid = False

            if os.path.exists(trace_file):
                cached_trace, cached_meta = _load_time_window_trace(trace_file)
                cache_valid = _trace_matches(cached_meta, trace_metadata)

                if cache_valid:
                    job_trace = cached_trace
                    trace_metadata = cached_meta
                    loaded_from_cache = True
                    if config.output.verbose:
                        print(f"\nLoading time-window job trace from: {trace_file}")
                    print(f"  ✓ Loaded trace with {len(job_trace)} jobs")
                else:
                    print(
                        "\nCached time-window trace does not match current config; regenerating..."
                    )
                    if config.output.verbose and cached_meta:
                        for key in trace_metadata:
                            if cached_meta.get(key) != trace_metadata[key]:
                                print(
                                    f"    {key}: cached={cached_meta.get(key)} "
                                    f"current={trace_metadata[key]}"
                                )

            if job_trace is None:
                # No trace yet or cache invalid: generate and save one
                print(f"\nGenerating job trace (time_window mode)...")

                job_trace = generate_job_trace(
                    num_jobs=trace_size,
                    arrival_rate=arrival_rate,
                    emotion_config=emotion_config,
                    default_service_time=default_service_time,
                    enable_emotion=config.workload.emotion.enable_emotion_aware,
                    random_seed=None,  # Seed already set globally
                    use_stratified_sampling=config.workload.emotion.use_stratified_sampling,
                    # MMPP parameters
                    mmpp_enabled=mmpp_config.enabled,
                    mmpp_lambda_high=mmpp_config.lambda_high,
                    mmpp_lambda_low=mmpp_config.lambda_low,
                    mmpp_alpha=mmpp_config.alpha,
                    mmpp_beta=mmpp_config.beta,
                )

                _save_time_window_trace(trace_file, job_trace, trace_metadata)

                print(
                    f"  ✓ Generated trace with {len(job_trace)} jobs "
                    f"and saved to {trace_file}"
                )

        # Create scheduler
        scheduler = create_scheduler(config)

        # Create adaptive k controller if enabled (Exp-4 Online Control)
        adaptive_k_controller = None
        if getattr(config.scheduler, 'adaptive_k', False):
            from core.adaptive_k_controller import create_controller_from_config
            adaptive_k_controller = create_controller_from_config(config.scheduler)
            if adaptive_k_controller is not None:
                print(f"  ✓ Adaptive k controller enabled (k: {config.scheduler.adaptive_k_min}-{config.scheduler.adaptive_k_max})")

        # Initialize LLM handler (LLM-only mode)
        llm_handler = init_llm_handler(config)

        # === Create EarlyPromptGenerator if predictor is available ===
        if length_estimator is not None and llm_handler is not None:
            from predictor.early_prompt_generator import EarlyPromptGenerator

            early_prompt_generator = EarlyPromptGenerator(
                dataset_loader=llm_handler.dataset_loader,
                prompt_builder=llm_handler.prompt_builder,
                length_estimator=length_estimator,
                default_service_time=default_service_time,
            )
            print(f"  ✓ Early prompt generator initialized")

            # Check if trace needs predictions
            if job_trace is not None:
                # Check if trace already has predicted_service_time
                has_predictions = any('predicted_service_time' in entry for entry in job_trace)
                if not has_predictions and early_prompt_generator.is_prediction_available():
                    print(f"\nEnriching trace with predictions...")
                    for i, entry in enumerate(job_trace):
                        # Create temp job for prediction
                        class TempJob:
                            def __init__(self, emotion, arousal, valence, job_id, conv_idx):
                                self.emotion_label = emotion
                                self.arousal = arousal
                                self.valence = valence
                                self.job_id = job_id
                                self.conversation_index = conv_idx

                            def get_emotion_label(self):
                                return self.emotion_label

                            def get_arousal(self):
                                return self.arousal

                        temp_job = TempJob(
                            entry['emotion'],
                            entry['arousal'],
                            entry['valence'],
                            entry['job_id'],
                            entry.get('conversation_index')
                        )
                        prompt, predicted_time, conv_idx = early_prompt_generator.generate_prompt_and_predict(temp_job)
                        entry['predicted_service_time'] = float(predicted_time)
                        entry['conversation_index'] = conv_idx

                        if (i + 1) % 100 == 0:
                            print(f"    Processed {i + 1}/{len(job_trace)} jobs")

                    print(f"  ✓ Added predictions to {len(job_trace)} jobs")

                    # Save updated trace (for time_window mode)
                    if mode == "time_window" and 'trace_file' in locals():
                        with open(trace_file, "w") as f:
                            json.dump(
                                {"metadata": trace_metadata, "trace": job_trace},
                                f,
                                indent=2,
                            )
                        print(f"  ✓ Saved updated trace with predictions")

        # Determine predictor mode (A2 defense experiment)
        if config.length_predictor.use_oracle:
            predictor_mode = 'oracle'
            print(f"  Predictor mode: ORACLE (using actual service times)")
        elif config.length_predictor.disabled:
            predictor_mode = 'disabled'
            print(f"  Predictor mode: DISABLED (using default service time)")
        else:
            predictor_mode = 'predicted'

        # Run scheduling based on mode
        print(f"\nRunning scheduling ({mode} mode)...")

        if mode == "fixed_jobs":
            completed_jobs, run_metrics = run_scheduling_loop(
                scheduler=scheduler,
                jobs=pre_generated_jobs,
                verbose=config.output.verbose,
                llm_handler=llm_handler,
                llm_skip_on_error=config.llm.error_handling.skip_on_error,
                early_prompt_generator=early_prompt_generator,
                adaptive_k_controller=adaptive_k_controller,
                predictor_mode=predictor_mode,
                default_service_time=default_service_time,
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
                early_prompt_generator=early_prompt_generator,
                adaptive_k_controller=adaptive_k_controller,
                predictor_mode=predictor_mode,
                default_service_time=default_service_time,
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

        # Save adaptive k controller trajectory if enabled
        if adaptive_k_controller is not None:
            trajectory_path = os.path.join(output_dir, "adaptive_k_trajectory.json")
            adaptive_k_controller.save_trajectory(trajectory_path)
            print(f"\nAdaptive k trajectory saved to: {trajectory_path}")

            # Print summary statistics
            stats = adaptive_k_controller.get_statistics()
            print(f"  k transitions: {stats['num_k_transitions']}")
            print(f"  k values used: {stats['k_values_used']}")
            print(f"  final k: {stats['final_k']}")
            print(f"  avg queue length: {stats['avg_queue_length']:.2f}")

    finally:
        # Restore stdout and close log file
        sys.stdout = original_stdout
        log_file.close()
        print(f"Log saved to: {log_path}")


__all__ = ["run_emotion_aware_experiment"]
