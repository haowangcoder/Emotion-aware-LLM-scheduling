"""
Queueing Theory Visualization

Plotting functions for theory vs simulation comparison.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Color scheme
COLORS = {
    'FCFS': '#1f77b4',      # Blue
    'SSJF-Emotion': '#ff7f0e',  # Orange
    'SSJF-Combined': '#2ca02c',  # Green
    'theory': '#333333',     # Dark gray
    'simulation': '#666666', # Medium gray
    'low': '#2ecc71',        # Green
    'medium': '#f39c12',     # Yellow
    'high': '#e74c3c',       # Red
}


def save_figure(fig, output_path: Union[str, Path], formats: List[str] = ['pdf', 'png']):
    """Save figure in multiple formats."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        fig.savefig(f"{output_path}.{fmt}", dpi=300, bbox_inches='tight')


def plot_theory_vs_simulation_load_sweep(
    simulation_results: Dict[float, dict],
    output_path: Union[str, Path],
    scheduler: str = 'FCFS',
    metric: str = 'W_q',
    formats: List[str] = ['pdf', 'png']
):
    """
    Plot theory vs simulation curves across load levels.

    Uses actual theory values from validation_results (computed with real E[S]).

    Args:
        simulation_results: {load: validation_result} from validate_load_sweep
        output_path: Base path for output (without extension)
        scheduler: 'FCFS' or 'SSJF-Emotion'
        metric: 'W_q' (waiting time) or 'W' (response time)
        formats: Output formats
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract both theory and simulation values from validation_results
    # This ensures we use actual parameters (E[S], λ) from the simulation data
    sim_loads = sorted(simulation_results.keys())
    theory_values = []
    sim_values = []

    for load in sim_loads:
        result = simulation_results[load]
        if scheduler == 'FCFS':
            # Theory computed with actual E[S] from data
            theory_values.append(result['theory'][metric])
            sim_values.append(result['simulation'][metric])
        else:  # SSJF-Emotion
            if metric == 'W_q':
                theory_values.append(result['overall']['theory_W_q_avg'])
                sim_values.append(result['overall']['sim_W_q_avg'])
            else:
                theory_values.append(result['overall'].get('theory_W_avg', np.nan))
                sim_values.append(result['overall'].get('sim_W_avg', np.nan))

    # Plot using same x-axis for both (actual load levels)
    ax.plot(sim_loads, theory_values, 'o-',
            label='Theory (M/G/1)', color=COLORS['theory'],
            linewidth=2, markersize=8)
    ax.plot(sim_loads, sim_values, 's--',
            label='Simulation', color=COLORS.get(scheduler, COLORS['simulation']),
            linewidth=2, markersize=8)

    # Labels
    ax.set_xlabel('System Load (ρ)', fontsize=12, fontweight='bold')
    ylabel = 'Average Waiting Time (s)' if metric == 'W_q' else 'Average Response Time (s)'
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(f'{scheduler}: Theory vs Simulation', fontsize=14, fontweight='bold')

    ax.legend(loc='upper left', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    save_figure(fig, output_path, formats)
    plt.close()


def plot_perclass_theory_vs_simulation(
    simulation_results: Dict[float, dict],
    output_path: Union[str, Path],
    load: float,
    metric: str = 'W_q',
    formats: List[str] = ['pdf', 'png']
):
    """
    Bar plot comparing theory vs simulation for each arousal class at a given load.

    Uses actual theory values from validation_results (computed with real E[S]).

    Args:
        simulation_results: Validation results from validate_load_sweep
        output_path: Base path for output
        load: Load level to plot
        metric: 'W_q' or 'W'
        formats: Output formats
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    classes = ['low', 'medium', 'high']
    x = np.arange(len(classes))
    width = 0.35

    # Get result for this load level
    result = simulation_results.get(load, {})

    # Get theory values from validation_results (computed with actual E[S])
    theory_metrics = result.get('theory', {})
    theory_values = [
        theory_metrics.get(c, {}).get(metric, np.nan)
        for c in classes
    ]

    # Get simulation values
    sim_metrics = result.get('simulation', {})
    sim_values = [
        sim_metrics.get(c, {}).get(metric, np.nan)
        for c in classes
    ]

    # Plot bars
    bars1 = ax.bar(x - width/2, theory_values, width, label='Theory',
                   color=[COLORS[c] for c in classes], alpha=0.7)
    bars2 = ax.bar(x + width/2, sim_values, width, label='Simulation',
                   color=[COLORS[c] for c in classes], alpha=1.0,
                   edgecolor='black', linewidth=1.5)

    # Add error annotations
    for i, (theory, sim) in enumerate(zip(theory_values, sim_values)):
        if not np.isnan(theory) and not np.isnan(sim) and theory > 0:
            error_pct = abs(theory - sim) / theory * 100
            y_pos = max(theory, sim) + 0.3
            ax.text(i, y_pos, f'{error_pct:.1f}%\nerror',
                    ha='center', fontsize=9, color='gray')

    # Labels
    ax.set_xlabel('Arousal Class', fontsize=12, fontweight='bold')
    ylabel = 'Waiting Time (s)' if metric == 'W_q' else 'Response Time (s)'
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(f'SSJF-Emotion Per-Class: Theory vs Simulation (ρ={load})',
                 fontsize=14, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in classes], fontsize=11)

    # Custom legend
    theory_patch = mpatches.Patch(facecolor='gray', alpha=0.7, label='Theory')
    sim_patch = mpatches.Patch(facecolor='gray', edgecolor='black',
                               linewidth=1.5, label='Simulation')
    ax.legend(handles=[theory_patch, sim_patch], loc='upper left', fontsize=11)

    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    save_figure(fig, output_path, formats)
    plt.close()


def plot_error_heatmap(
    load_levels: List[float],
    schedulers: List[str],
    error_matrix: np.ndarray,
    output_path: Union[str, Path],
    formats: List[str] = ['pdf', 'png']
):
    """
    Heatmap showing relative error (%) across loads and schedulers.

    Args:
        load_levels: List of load levels
        schedulers: List of scheduler names
        error_matrix: Shape (len(schedulers), len(load_levels)) with error percentages
        output_path: Base path for output
        formats: Output formats
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    # Create heatmap
    im = ax.imshow(error_matrix, cmap='RdYlGn_r', aspect='auto',
                   vmin=0, vmax=50)

    # Labels
    ax.set_xticks(np.arange(len(load_levels)))
    ax.set_yticks(np.arange(len(schedulers)))
    ax.set_xticklabels([f'ρ={l:.1f}' for l in load_levels], fontsize=10)
    ax.set_yticklabels(schedulers, fontsize=10)

    # Annotate cells
    for i in range(len(schedulers)):
        for j in range(len(load_levels)):
            value = error_matrix[i, j]
            text_color = 'white' if value > 25 else 'black'
            ax.text(j, i, f'{value:.1f}%',
                    ha='center', va='center', color=text_color, fontsize=10)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Relative Error (%)', fontsize=11)

    ax.set_title('Theory vs Simulation Error Across Loads',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('System Load', fontsize=12, fontweight='bold')
    ax.set_ylabel('Scheduler', fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_figure(fig, output_path, formats)
    plt.close()


def plot_priority_effect(
    validation_results: Dict[str, Dict[float, dict]],
    output_path: Union[str, Path],
    load: float = 0.7,
    formats: List[str] = ['pdf', 'png']
):
    """
    Plot showing how priority affects waiting time for different classes.

    Compares FCFS (equal treatment) vs SSJF-Emotion (priority by arousal).
    Uses actual theory values from validation_results.

    Args:
        validation_results: {scheduler: {load: result}} from validate_load_sweep
        output_path: Base path for output
        load: Load level to show
        formats: Output formats
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    classes = ['low', 'medium', 'high']
    x = np.arange(len(classes))
    width = 0.35

    # FCFS: same W_q for all classes (from theory)
    fcfs_result = validation_results.get('FCFS', {}).get(load, {})
    fcfs_wq = fcfs_result.get('theory', {}).get('W_q', np.nan)
    fcfs_values = [fcfs_wq] * len(classes)

    # SSJF-Emotion: different W_q by class (from theory)
    ssjf_result = validation_results.get('SSJF-Emotion', {}).get(load, {})
    ssjf_theory = ssjf_result.get('theory', {})
    ssjf_values = [
        ssjf_theory.get(c, {}).get('W_q', np.nan)
        for c in classes
    ]

    # Plot
    ax.bar(x - width/2, fcfs_values, width, label='FCFS',
           color=COLORS['FCFS'], alpha=0.8)
    ax.bar(x + width/2, ssjf_values, width, label='SSJF-Emotion',
           color=COLORS['SSJF-Emotion'], alpha=0.8)

    # Add arrows showing the effect
    for i, (fcfs, ssjf) in enumerate(zip(fcfs_values, ssjf_values)):
        if ssjf < fcfs:
            # Improvement (down arrow)
            ax.annotate('', xy=(i + width/2, ssjf), xytext=(i + width/2, fcfs),
                        arrowprops=dict(arrowstyle='->', color='green', lw=2))
        else:
            # Degradation (up arrow)
            ax.annotate('', xy=(i + width/2, ssjf), xytext=(i + width/2, fcfs),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # Labels
    ax.set_xlabel('Arousal Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Theoretical Waiting Time (s)', fontsize=12, fontweight='bold')
    ax.set_title(f'Priority Effect: FCFS vs SSJF-Emotion (ρ={load})',
                 fontsize=14, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f'{c.capitalize()}\n(Priority {i+1})' for i, c in enumerate(classes)])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    save_figure(fig, output_path, formats)
    plt.close()


def plot_load_sensitivity(
    validation_results: Dict[str, Dict[float, dict]],
    output_path: Union[str, Path],
    scheduler: str = 'FCFS',
    formats: List[str] = ['pdf', 'png']
):
    """
    Plot showing waiting time sensitivity to load changes.

    Demonstrates the 1/(1-ρ) relationship near ρ=1.
    Uses actual ρ values from validation_results.

    Args:
        validation_results: {scheduler: {load: result}} from validate_load_sweep
        output_path: Base path for output
        scheduler: Scheduler to plot
        formats: Output formats
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Extract data from validation_results
    scheduler_results = validation_results.get(scheduler, {})
    sorted_loads = sorted(scheduler_results.keys())

    # Extract actual rho values and W_q from validation
    actual_rhos = []
    wq_values = []

    for load in sorted_loads:
        result = scheduler_results[load]
        if scheduler == 'FCFS':
            actual_rhos.append(result['theory']['rho'])
            wq_values.append(result['theory']['W_q'])
        else:
            # For SSJF-Emotion, use overall metrics
            # rho is in individual class results, use first one's rho_cumulative
            theory = result.get('theory', {})
            if 'high' in theory:
                actual_rhos.append(theory['high']['rho_cumulative'])
            else:
                actual_rhos.append(np.nan)
            wq_values.append(result['overall']['theory_W_q_avg'])

    loads = np.array(actual_rhos)
    wq_values = np.array(wq_values)

    # Left plot: W_q vs ρ
    ax1.plot(loads, wq_values, 'o-', color=COLORS.get(scheduler, 'blue'),
             linewidth=2, markersize=8)
    ax1.set_xlabel('System Load (ρ)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Waiting Time W_q (s)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{scheduler}: Waiting Time vs Load', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)

    # Right plot: W_q vs 1/(1-ρ) - should be linear
    sensitivity = 1 / (1 - loads)
    ax2.plot(sensitivity, wq_values, 'o-', color=COLORS.get(scheduler, 'blue'),
             linewidth=2, markersize=8)
    ax2.set_xlabel('1/(1-ρ)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Waiting Time W_q (s)', fontsize=12, fontweight='bold')
    ax2.set_title('Linear Relationship Test', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_figure(fig, output_path, formats)
    plt.close()


def generate_all_validation_plots(
    validation_results: Dict[str, Dict[float, dict]],
    output_dir: Union[str, Path],
    formats: List[str] = ['pdf', 'png']
):
    """
    Generate all validation plots.

    All plots now use validation_results which contains theory values
    computed with actual E[S] from simulation data.

    Args:
        validation_results: {scheduler: {load: result}}
        output_dir: Directory to save plots
        formats: Output formats
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sweep plots
    for scheduler in validation_results:
        plot_theory_vs_simulation_load_sweep(
            validation_results[scheduler],
            output_dir / f'load_sweep_{scheduler.lower().replace("-", "_")}',
            scheduler=scheduler,
            formats=formats
        )

    # Per-class comparison (for middle load)
    loads = list(validation_results.get('SSJF-Emotion', {}).keys())
    if loads:
        mid_load = loads[len(loads) // 2]
        plot_perclass_theory_vs_simulation(
            validation_results.get('SSJF-Emotion', {}),
            output_dir / f'perclass_load{mid_load}',
            load=mid_load,
            formats=formats
        )

    # Error heatmap
    schedulers = list(validation_results.keys())
    if loads and schedulers:
        error_matrix = np.zeros((len(schedulers), len(loads)))
        for i, scheduler in enumerate(schedulers):
            for j, load in enumerate(sorted(loads)):
                result = validation_results[scheduler].get(load, {})
                if scheduler == 'FCFS':
                    error = result.get('errors', {}).get('rel_error_W_q', np.nan)
                else:
                    theory_wq = result.get('overall', {}).get('theory_W_q_avg', 0)
                    sim_wq = result.get('overall', {}).get('sim_W_q_avg', 0)
                    error = abs(theory_wq - sim_wq) / theory_wq * 100 if theory_wq > 0 else np.nan
                error_matrix[i, j] = error

        plot_error_heatmap(
            sorted(loads), schedulers, error_matrix,
            output_dir / 'error_heatmap',
            formats=formats
        )

    # Priority effect plot
    if loads:
        plot_priority_effect(
            validation_results,
            output_dir / 'priority_effect',
            load=loads[len(loads) // 2],
            formats=formats
        )

    # Load sensitivity
    plot_load_sensitivity(
        validation_results,
        output_dir / 'load_sensitivity',
        scheduler='FCFS',
        formats=formats
    )

    print(f"All plots saved to {output_dir}")
