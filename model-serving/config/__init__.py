"""
Configuration module for emotion-aware LLM scheduling.

This module consolidates all configuration parameters into logical groups:
- base: General settings, paths, and experiment configuration
- llm_config: LLM model, generation, and caching parameters
- scheduler_config: Scheduling algorithm and system load settings
- workload_config: Task generation and emotion-aware workload parameters
"""

# Import all configuration parameters for convenient access
from .base import *
from .llm_config import *
from .scheduler_config import *
from .workload_config import *
