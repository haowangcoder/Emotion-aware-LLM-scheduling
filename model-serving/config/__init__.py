"""
Configuration package for emotion-aware LLM scheduling.

Use the YAML-based configuration loader:

    from config.config_loader import load_config, get_alpha

The legacy Python config modules (base.py, llm_config.py, scheduler_config.py,
workload_config.py) have been removed.
"""

from .config_loader import load_config, get_alpha  # Convenience re-export

__all__ = ["load_config", "get_alpha"]
