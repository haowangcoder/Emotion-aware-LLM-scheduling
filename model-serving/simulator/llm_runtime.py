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

    algorithm = config.scheduler.algorithm
    print(f"\nCreating {algorithm} scheduler...")

    if algorithm == "FCFS":
        scheduler = FCFSScheduler()

    elif algorithm in ("SJF", "SSJF"):
        scheduler = SJFScheduler()

    elif algorithm == "AW-SSJF":
        # Main algorithm: Affect-Weighted SSJF
        affect_cfg = config.scheduler.affect_weight
        scheduler = AWSSJFScheduler(
            w_max=affect_cfg.w_max,
            p=affect_cfg.p,
            q=affect_cfg.q,
            use_confidence=affect_cfg.use_confidence,
            starvation_threshold=config.scheduler.starvation_prevention.threshold,
            starvation_coefficient=config.scheduler.starvation_prevention.coefficient,
        )
        print(f"  w_max={affect_cfg.w_max}, p={affect_cfg.p}, q={affect_cfg.q}")

    elif algorithm == "Weight-Only":
        # Ablation baseline: Pure affect-based priority
        affect_cfg = config.scheduler.affect_weight
        scheduler = WeightOnlyScheduler(
            w_max=affect_cfg.w_max,
            p=affect_cfg.p,
            q=affect_cfg.q,
            use_confidence=affect_cfg.use_confidence,
            starvation_threshold=config.scheduler.starvation_prevention.threshold,
        )
        print(f"  w_max={affect_cfg.w_max}, p={affect_cfg.p}, q={affect_cfg.q}")

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
