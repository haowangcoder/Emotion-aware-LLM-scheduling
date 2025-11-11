#!/usr/bin/env python3
"""
Unified Entry Point for Emotion-aware LLM Scheduling Experiments

This script provides a clean entry point to run the emotion-aware LLM scheduling
simulator from the project root directory.

Usage:
    python run_simulation.py --scheduler FCFS --num_jobs 100 --system_load 0.6
    python run_simulation.py --scheduler SSJF-Emotion --num_jobs 200 --alpha 0.7
    python run_simulation.py --help

For detailed documentation, see: model-serving/simulator.py
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
