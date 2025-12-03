import os
from typing import List, Optional, Tuple

from core.job import Job
from core.job_config_manager import JobConfigManager
from workload.service_time_mapper import ServiceTimeConfig
from workload.task_generator import create_emotion_aware_jobs
from core.emotion import EmotionConfig


def build_job_config_manager(config) -> Tuple[JobConfigManager, str]:
    """
    Build JobConfigManager and its underlying file path from config.
    """
    job_config_path = config.llm.cache.full_job_config_path
    job_config_manager = JobConfigManager(job_config_path)
    return job_config_manager, job_config_path


def load_pre_generated_jobs(
    config,
    emotion_config: EmotionConfig,
    service_config: ServiceTimeConfig,
    arrival_rate: float,
) -> Tuple[Optional[List[Job]], bool, bool]:
    """
    Load pre-generated jobs from saved job configuration if enabled.

    Returns (pre_generated_jobs, use_saved, force_new).
    """
    job_config_manager, job_config_path = build_job_config_manager(config)

    pre_generated_jobs: Optional[List[Job]] = None
    use_saved = config.llm.cache.use_saved_job_config
    force_new = config.llm.cache.force_new_job_config

    print(f"\nJob Configuration:")
    print(f"  Use saved config: {use_saved}")
    print(f"  Force new config: {force_new}")
    print(f"  Config file: {job_config_path}")

    if use_saved and not force_new:
        print(f"  Attempting to load existing job configuration...")
        config_data = job_config_manager.load_job_configs()

        if config_data is not None:
            job_configs = config_data.get("jobs", [])

            print(f"  Generating jobs from loaded configuration...")
            pre_generated_jobs = create_emotion_aware_jobs(
                num_jobs=len(job_configs),
                arrival_rate=arrival_rate,
                emotion_config=emotion_config,
                service_time_config=service_config,
                enable_emotion=config.workload.emotion.enable_emotion_aware,
                job_configs=job_configs,
                random_seed=None,  # Don't reset seed here, already set above
            )
            print(f"  ✓ Loaded {len(pre_generated_jobs)} jobs from saved configuration")
        else:
            print(f"  No existing configuration found, will generate new jobs")
    else:
        if force_new:
            print(f"  Force new config enabled, will generate new jobs")
        else:
            print(f"  Saved config disabled, will generate new jobs")

    return pre_generated_jobs, use_saved, force_new


def save_job_config_if_needed(
    config,
    completed_jobs: List[Job],
    arrival_rate: float,
    alpha: float,
    pre_generated_jobs: Optional[List[Job]],
    force_new: bool,
) -> None:
    """
    Save job configurations for reproducibility if conditions are met.
    """
    job_config_manager, job_config_path = build_job_config_manager(config)

    if pre_generated_jobs is None or force_new:
        print(f"\nSaving job configuration for future reproducibility...")

        job_config_metadata = {
            "experiment_mode": "fixed_jobs",
            "num_jobs": len(completed_jobs),
            "arrival_rate": arrival_rate,
            "system_load": config.scheduler.system_load,
            "random_seed": config.experiment.random_seed,
            "base_service_time": config.workload.service_time.base_service_time,
            "alpha": alpha,
            "emotion_correlation": config.workload.service_time.emotion_correlation,
            "enable_emotion_aware": config.workload.emotion.enable_emotion_aware,
            "model_name": config.llm.model.name,
        }

        job_config_manager.save_job_configs(
            jobs=completed_jobs,
            metadata=job_config_metadata,
        )
        print(
            f"  ✓ Saved {len(completed_jobs)} job configurations to {job_config_path}"
        )
        print(
            "  Note: Different schedulers can now use this config for fair comparison"
        )
    else:
        print(f"\nSkipping job config save (loaded from existing configuration)")


__all__ = [
    "build_job_config_manager",
    "load_pre_generated_jobs",
    "save_job_config_if_needed",
]

