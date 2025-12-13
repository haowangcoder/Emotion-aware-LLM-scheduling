"""
LLM Runtime Module for Scheduler and Inference Handler Creation.

This module provides factory functions for creating:
- Scheduler instances based on configuration
- LLM inference handlers for response generation
"""

import os
import sys


def create_scheduler(config):
    """
    Create scheduler instance based on configuration.

    Supported algorithms:
        - FCFS: First-Come-First-Serve (baseline)
        - SJF/SSJF: Shortest-Job-First (uses predicted service time)
        - AW-SSJF: Affect-Weighted SSJF (main algorithm)
        - Weight-Only: Pure affect-based priority (ablation baseline)
    """
    from core.scheduler_base import FCFSScheduler, SJFScheduler
    from core.aw_ssjf_scheduler import AWSSJFScheduler
    from core.weight_only_scheduler import WeightOnlyScheduler
    from core.affect_weight_v2 import WeightMode, WeightConfig

    algorithm = config.scheduler.algorithm
    print(f"\nCreating {algorithm} scheduler...")

    if algorithm == "FCFS":
        scheduler = FCFSScheduler()

    elif algorithm in ("SJF", "SSJF"):
        scheduler = SJFScheduler()

    elif algorithm == "AW-SSJF":
        # Main algorithm: Affect-Weighted SSJF
        affect_cfg = config.scheduler.affect_weight

        # Determine whether to use v2 weight config or legacy parameters
        weight_config = None
        weight_preset = getattr(affect_cfg, "weight_preset", None)
        weight_mode = getattr(affect_cfg, "weight_mode", None)
        mode_value = str(weight_mode).lower() if weight_mode is not None else None

        if weight_preset is None and mode_value:
            try:
                mode_enum = WeightMode(mode_value)
            except ValueError:
                # Treat unknown modes as preset names (e.g., depression_first_soft)
                weight_preset = mode_value
            else:
                weight_config = WeightConfig(
                    mode=mode_enum,
                    w_max=affect_cfg.w_max,
                    p=affect_cfg.p,
                    q=affect_cfg.q,
                    r=getattr(affect_cfg, "r", 1.0),
                    k_v=getattr(affect_cfg, "k_v", 5.0),
                    k_a=getattr(affect_cfg, "k_a", 5.0),
                    tau_v=getattr(affect_cfg, "tau_v", 0.0),
                    tau_a=getattr(affect_cfg, "tau_a", 0.0),
                    tau_h=getattr(affect_cfg, "tau_h", 0.0),
                    gamma_dep=getattr(affect_cfg, "gamma_dep", 1.0),
                    gamma_panic=getattr(affect_cfg, "gamma_panic", 0.3),
                )

        weight_exp = getattr(config.scheduler, 'weight_exponent', 1.0)
        scheduler = AWSSJFScheduler(
            w_max=affect_cfg.w_max,
            p=affect_cfg.p,
            q=affect_cfg.q,
            use_confidence=affect_cfg.use_confidence,
            starvation_threshold=config.scheduler.starvation_prevention.threshold,
            starvation_coefficient=config.scheduler.starvation_prevention.coefficient,
            weight_config=weight_config,
            weight_preset=weight_preset,
            use_robust_scoring=getattr(config.scheduler, 'use_robust_scoring', False),
            use_conservative_prediction=getattr(config.scheduler, 'use_conservative_prediction', False),
            conservative_margin=getattr(config.scheduler, 'conservative_margin', 1.3),
            weight_exponent=weight_exp,
        )
        if weight_exp != 1.0:
            print(f"  Weight exponent: k={weight_exp} (w^{weight_exp} amplification)")
        if getattr(config.scheduler, 'use_robust_scoring', False):
            print(f"  Robust scoring: ENABLED (log transform)")
        if getattr(config.scheduler, 'use_conservative_prediction', False):
            margin = getattr(config.scheduler, 'conservative_margin', 1.3)
            print(f"  Conservative prediction: ENABLED (margin={margin:.0%})")
        if weight_preset:
            print(f"  weight_preset={weight_preset} (v2 weights)")
        elif weight_config:
            print(
                f"  weight_mode={weight_config.mode.value} (v2 weights), "
                f"w_max={weight_config.w_max}, p={weight_config.p}, q={weight_config.q}"
            )
        else:
            print(f"  w_max={affect_cfg.w_max}, p={affect_cfg.p}, q={affect_cfg.q} (legacy)")

    elif algorithm == "Weight-Only":
        # Ablation baseline: Pure affect-based priority (with v2.0 weight support)
        affect_cfg = config.scheduler.affect_weight

        # Determine whether to use v2 weight config or legacy parameters
        weight_config = None
        weight_preset = getattr(affect_cfg, "weight_preset", None)
        weight_mode = getattr(affect_cfg, "weight_mode", None)
        mode_value = str(weight_mode).lower() if weight_mode is not None else None

        if weight_preset is None and mode_value:
            try:
                mode_enum = WeightMode(mode_value)
            except ValueError:
                # Treat unknown modes as preset names (e.g., depression_first_soft)
                weight_preset = mode_value
            else:
                weight_config = WeightConfig(
                    mode=mode_enum,
                    w_max=affect_cfg.w_max,
                    p=affect_cfg.p,
                    q=affect_cfg.q,
                    r=getattr(affect_cfg, "r", 1.0),
                    k_v=getattr(affect_cfg, "k_v", 5.0),
                    k_a=getattr(affect_cfg, "k_a", 5.0),
                    tau_v=getattr(affect_cfg, "tau_v", 0.0),
                    tau_a=getattr(affect_cfg, "tau_a", 0.0),
                    tau_h=getattr(affect_cfg, "tau_h", 0.0),
                    gamma_dep=getattr(affect_cfg, "gamma_dep", 1.0),
                    gamma_panic=getattr(affect_cfg, "gamma_panic", 0.3),
                )

        scheduler = WeightOnlyScheduler(
            w_max=affect_cfg.w_max,
            p=affect_cfg.p,
            q=affect_cfg.q,
            use_confidence=affect_cfg.use_confidence,
            starvation_threshold=config.scheduler.starvation_prevention.threshold,
            weight_config=weight_config,
            weight_preset=weight_preset,
        )
        if weight_preset:
            print(f"  weight_preset={weight_preset} (v2 weights)")
        elif weight_config:
            print(
                f"  weight_mode={weight_config.mode.value} (v2 weights), "
                f"w_max={weight_config.w_max}, p={weight_config.p}, q={weight_config.q}"
            )
        else:
            print(f"  w_max={affect_cfg.w_max}, p={affect_cfg.p}, q={affect_cfg.q} (legacy)")

    else:
        raise ValueError(
            f"Unknown scheduler algorithm: {algorithm}. "
            f"Supported: FCFS, SJF, SSJF, AW-SSJF, Weight-Only"
        )

    return scheduler


def init_llm_handler(config):
    """
    Initialize LLM inference handler from configuration.

    Args:
        config: Configuration object
    """
    print(f"\nInitializing LLM Inference Handler...")
    print(f"  Model: {config.llm.model.name}")
    print(f"  Dataset: {config.dataset.emotion_dataset_path}")
    print(
        f"  Cache: {'Enabled' if config.llm.cache.use_response_cache else 'Disabled'}"
    )

    try:
        from llm.inference_handler import LLMInferenceHandler

        llm_handler = LLMInferenceHandler(
            model_name=config.llm.model.name,
            dataset_path=config.dataset.emotion_dataset_path,
            cache_path=os.path.join(
                config.llm.cache.cache_dir, config.llm.cache.cache_file
            ),
            use_cache=config.llm.cache.use_response_cache,
            force_regenerate=config.llm.cache.force_regenerate,
            device_map=config.llm.model.device_map,
            dtype=config.llm.model.dtype,
            load_in_8bit=config.llm.model.load_in_8bit,
            include_system_prompt=config.llm.prompt.include_system_prompt,
            include_emotion_hint=config.llm.prompt.include_emotion_hint,
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

        return llm_handler

    except Exception as e:  # pragma: no cover - direct process exit
        print(f"ERROR: Failed to initialize LLM handler: {e}")
        print("LLM-only mode is enabled; no fallback is available.")
        print("Please fix the LLM setup (model, dataset, or device settings) and retry.")
        sys.exit(1)


def save_cache_if_needed(llm_handler) -> None:
    """
    Save LLM response cache if handler is available.
    """
    if llm_handler is None:
        return

    print(f"\nSaving LLM response cache...")
    llm_handler.save_cache()
    cache_stats = llm_handler.get_cache_stats()
    print(
        f"  Cache stats: {cache_stats['num_entries']} entries, "
        f"{cache_stats['hit_rate']:.1%} hit rate"
    )


__all__ = ["create_scheduler", "init_llm_handler", "save_cache_if_needed"]
