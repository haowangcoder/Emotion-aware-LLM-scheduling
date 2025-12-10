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
        - SJF: Shortest-Job-First (uses predicted service time)
        - AW-SSJF: Affect-Weighted SSJF (main algorithm)
        - Weight-Only: Pure affect-based priority (ablation baseline)
        - SSJF-Emotion: Legacy emotion-aware scheduler (deprecated)
        - SSJF-Valence: Legacy valence-weighted scheduler (deprecated)
        - SSJF-Combined: Legacy W/L scheduler (deprecated)
    """
    from core.scheduler_base import FCFSScheduler, SJFScheduler
    from core.aw_ssjf_scheduler import AWSSJFScheduler
    from core.weight_only_scheduler import WeightOnlyScheduler
    # Legacy schedulers (deprecated)
    from core.ssjf_emotion import (
        SSJFEmotionScheduler,
        SSJFValenceScheduler,
        SSJFCombinedScheduler,
    )

    algorithm = config.scheduler.algorithm
    print(f"\nCreating {algorithm} scheduler...")

    if algorithm == "FCFS":
        scheduler = FCFSScheduler()

    elif algorithm == "SJF" or algorithm == "SSJF":
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

    # ========== Legacy Schedulers (Deprecated) ==========
    elif algorithm == "SSJF-Emotion":
        scheduler = SSJFEmotionScheduler(
            starvation_threshold=config.scheduler.starvation_prevention.threshold,
            starvation_coefficient=config.scheduler.starvation_prevention.coefficient,
        )
        print("  [DEPRECATED] Using legacy SSJF-Emotion scheduler")

    elif algorithm == "SSJF-Valence":
        # Check if valence_priority config exists (backward compatibility)
        if hasattr(config.scheduler, 'valence_priority'):
            valence_cfg = config.scheduler.valence_priority
            beta = valence_cfg.beta
            min_positive_weight = valence_cfg.min_positive_weight
            class_weights = getattr(valence_cfg, "class_weights", None)
        else:
            # Use affect_weight config as fallback
            beta = config.scheduler.affect_weight.w_max - 1.0
            min_positive_weight = 0.1
            class_weights = None

        scheduler = SSJFValenceScheduler(
            beta=beta,
            starvation_threshold=config.scheduler.starvation_prevention.threshold,
            valence_weight_map=class_weights if class_weights else None,
            min_positive_weight=min_positive_weight,
        )
        print("  [DEPRECATED] Using legacy SSJF-Valence scheduler")

    elif algorithm == "SSJF-Combined":
        # Check if valence_priority config exists (backward compatibility)
        if hasattr(config.scheduler, 'valence_priority'):
            beta = config.scheduler.valence_priority.beta
        else:
            beta = config.scheduler.affect_weight.w_max - 1.0

        # Check if alpha config exists (backward compatibility)
        if hasattr(config.workload.service_time, 'alpha'):
            alpha = config.workload.service_time.alpha
        else:
            alpha = 0.5  # Default fallback

        scheduler = SSJFCombinedScheduler(
            beta=beta,
            alpha=alpha,
            base_service_time=config.workload.service_time.base_service_time,
            starvation_threshold=config.scheduler.starvation_prevention.threshold,
            starvation_coefficient=config.scheduler.starvation_prevention.coefficient,
        )
        print("  [DEPRECATED] Using legacy SSJF-Combined scheduler")

    else:
        raise ValueError(f"Unknown scheduler algorithm: {algorithm}")

    return scheduler


def init_llm_handler(config, alpha=None):
    """
    Initialize LLM inference handler from configuration.

    Args:
        config: Configuration object
        alpha: Alpha parameter (optional, for backward compatibility)
    """
    print(f"\nInitializing LLM Inference Handler...")
    print(f"  Model: {config.llm.model.name}")
    print(f"  Dataset: {config.dataset.emotion_dataset_path}")
    print(
        f"  Cache: {'Enabled' if config.llm.cache.use_response_cache else 'Disabled'}"
    )

    try:
        from llm.inference_handler import LLMInferenceHandler

        # Get alpha from config if not provided
        if alpha is None:
            # Try to get alpha from workload config (backward compatibility)
            alpha = getattr(config.workload.service_time, 'alpha', 0.5)

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
            include_emotion_hint=config.llm.prompt.include_emotion_hint,
            enable_emotion_length_control=False,  # Disabled in new architecture
            base_response_length=64,  # Not used when length control is disabled
            alpha=alpha,  # Kept for backward compatibility
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
