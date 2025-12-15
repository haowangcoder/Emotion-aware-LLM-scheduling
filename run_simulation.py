#!/usr/bin/env python3
"""
Unified Entry Point for Emotion-aware LLM Scheduling Experiments

This script provides a clean entry point to run the emotion-aware LLM scheduling
simulator from the project root directory.

- Jobs arrive continuously at rate λ = system_load / E[S]
- Simulation runs for a fixed duration (`simulation_duration`)
- Only jobs completed within the window are counted (jobs finishing after the cutoff are excluded)

Usage:
    python run_simulation.py --mode time_window --scheduler FCFS --simulation_duration 200 --system_load 0.6
    python run_simulation.py --mode time_window --scheduler SJF --simulation_duration 300 --system_load 0.8
    python run_simulation.py --help

For detailed documentation, see: model-serving/simulator/
"""

import sys
import os

# Add model-serving directory to Python path
model_serving_dir = os.path.join(os.path.dirname(__file__), 'model-serving')
sys.path.insert(0, model_serving_dir)

# Import and run the simulator
if __name__ == '__main__':
    from simulator import main
    main()
