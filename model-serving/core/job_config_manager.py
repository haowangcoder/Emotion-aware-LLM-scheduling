"""
Job Configuration Manager for reproducible experiments.

This module provides functionality to save and load job configurations
to ensure that different scheduler runs use identical workloads for fair comparison.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path


class JobConfigManager:
    """Manages saving and loading of job configurations for reproducible experiments."""

    def __init__(self, config_file: str):
        """
        Initialize the JobConfigManager.

        Args:
            config_file: Path to the job configuration file
        """
        self.config_file = config_file
        self._ensure_directory()

    def _ensure_directory(self):
        """Ensure the directory for config file exists."""
        directory = os.path.dirname(self.config_file)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def save_job_configs(
        self,
        jobs: List[Any],
        metadata: Dict[str, Any]
    ) -> None:
        """
        Save job configurations to file.

        Args:
            jobs: List of job objects
            metadata: Metadata about the experiment (num_jobs, random_seed, etc.)
        """
        config_data = {
            "metadata": {
                **metadata,
                "created_at": datetime.now().isoformat(),
                "version": "1.0"
            },
            "jobs": []
        }

        for job in jobs:
            job_config = {
                "job_id": job.job_id,
                "emotion": getattr(job, 'emotion_label', None),
                "arousal": getattr(job, 'arousal', None),
                "conversation_index": getattr(job, 'conversation_index', None),
                "arrival_time": job.arrival_time,
                "service_time": job.execution_duration,
            }

            # Add optional fields if they exist
            if hasattr(job, 'prompt_hash'):
                job_config["prompt_hash"] = job.prompt_hash

            config_data["jobs"].append(job_config)

        # Save to file with pretty printing
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)

        print(f"[JobConfigManager] Saved {len(jobs)} job configurations to {self.config_file}")

    def load_job_configs(self) -> Optional[Dict[str, Any]]:
        """
        Load job configurations from file.

        Returns:
            Dictionary containing metadata and job configurations, or None if file doesn't exist
        """
        if not os.path.exists(self.config_file):
            print(f"[JobConfigManager] No existing job config found at {self.config_file}")
            return None

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            print(f"[JobConfigManager] Loaded {len(config_data['jobs'])} job configurations from {self.config_file}")
            print(f"[JobConfigManager] Config metadata: {config_data['metadata']}")

            return config_data
        except Exception as e:
            print(f"[JobConfigManager] Error loading job config: {e}")
            return None

    def config_exists(self) -> bool:
        """Check if a job configuration file exists."""
        return os.path.exists(self.config_file)

    def delete_config(self) -> bool:
        """
        Delete the job configuration file.

        Returns:
            True if file was deleted, False if it didn't exist
        """
        if os.path.exists(self.config_file):
            os.remove(self.config_file)
            print(f"[JobConfigManager] Deleted job config file: {self.config_file}")
            return True
        return False

    def validate_config(self, config_data: Dict[str, Any], expected_num_jobs: int) -> bool:
        """
        Validate that the loaded configuration matches expected parameters.

        Args:
            config_data: Loaded configuration data
            expected_num_jobs: Expected number of jobs

        Returns:
            True if configuration is valid, False otherwise
        """
        if not config_data:
            return False

        if len(config_data['jobs']) != expected_num_jobs:
            print(f"[JobConfigManager] Warning: Config has {len(config_data['jobs'])} jobs, "
                  f"but {expected_num_jobs} expected")
            return False

        # Check that all jobs have required fields
        required_fields = ['job_id', 'emotion', 'arrival_time', 'service_time']
        for job in config_data['jobs']:
            for field in required_fields:
                if field not in job:
                    print(f"[JobConfigManager] Error: Job {job.get('job_id', '?')} missing field: {field}")
                    return False

        return True

    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Get metadata from the configuration file without loading all job data.

        Returns:
            Metadata dictionary or None if file doesn't exist
        """
        if not os.path.exists(self.config_file):
            return None

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            return config_data.get('metadata')
        except Exception as e:
            print(f"[JobConfigManager] Error reading metadata: {e}")
            return None


def create_job_config_manager(config_file: Optional[str] = None) -> JobConfigManager:
    """
    Factory function to create a JobConfigManager instance.
    Explicitly require a provided config_file so that the system
    never falls back to hardcoded results/cache paths.
    """
    if config_file is None:
        raise ValueError("JobConfigManager requires an explicit config_file path")

    return JobConfigManager(config_file)
