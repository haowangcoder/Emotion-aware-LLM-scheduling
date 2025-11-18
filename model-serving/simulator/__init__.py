"""
Simulator package for emotion-aware LLM scheduling.

Provides convenient entry points:
- main(): CLI entry
- run_emotion_aware_experiment(): high-level experiment orchestration
- run_scheduling_loop(): core scheduling loop
"""

from .cli import main
from .experiment import run_emotion_aware_experiment
from .loop import run_scheduling_loop

__all__ = ["main", "run_emotion_aware_experiment", "run_scheduling_loop"]

