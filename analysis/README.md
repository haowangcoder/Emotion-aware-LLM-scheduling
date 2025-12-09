# Analysis Module

This module contains experiment result aggregation, visualization, and theoretical validation functionality.

## Directory Structure

```
analysis/
├── README.md                   # This document
│
├── aggregate_results.py        # Multi-seed result aggregation
├── plot_emotion_results.py     # Legacy plotting (deprecated)
├── plot_publication.py         # Publication-quality plot entry point
│
├── plotting/                   # General plotting module
│   ├── __init__.py
│   ├── main.py                 # Main entry: generate_all_publication_plots()
│   ├── constants.py            # Colors, markers, ordering
│   ├── data_loader.py          # Data loading functions
│   ├── utils.py                # Utilities (style, save)
│   └── plots/                  # Plot types
│       ├── forest.py           # Forest plot (mean + CI)
│       ├── distribution.py     # ECDF/CCDF distribution
│       ├── pareto.py           # Pareto frontier (efficiency-fairness)
│       ├── slopegraph.py       # Group analysis (arousal/valence)
│       ├── calibration.py      # Prediction calibration
│       ├── heatmap.py          # Parameter sweep heatmaps
│       └── ablation.py         # Ablation/robustness experiments
│
└── queueing_theory/            # M/G/1 queueing theory validation
    ├── __init__.py
    ├── mg1_formulas.py         # Pollaczek-Khinchin formula
    ├── multiclass_priority.py  # Kleinrock multi-class priority formula
    ├── service_time_analysis.py # Service time distribution analysis
    ├── load_sweep.py           # Load sweep theory predictions
    ├── error_analysis.py       # Error analysis (transient removal, etc.)
    ├── validation.py           # Theory vs simulation validation
    └── plotting.py             # Validation result visualization
```

## Module Descriptions

### 1. Multi-seed Aggregation (`aggregate_results.py`)

Aggregates results from multiple random seeds and computes statistics.

```bash
uv run python analysis/aggregate_results.py \
    --results_dir results/multi_seed_runs \
    --output_file results/multi_seed_runs/aggregated_results.json
```

**Output**: `aggregated_results.json` with mean, std, 95% confidence intervals.

### 2. Publication Plotting (`plotting/`)

Generates publication-quality figures.

```bash
uv run python analysis/plot_publication.py \
    --aggregated_results results/multi_seed_runs/aggregated_results.json \
    --output_dir results/publication_plots
```

**Main Plots**:
- `forest_*.png` - Scheduler comparison (mean + 95% CI)
- `pareto_*.png` - Efficiency-fairness Pareto frontier
- `slopegraph_*.png` - Per-emotion-class waiting time comparison
- `ecdf_*.png` / `ccdf_*.png` - Waiting time distributions

### 3. Queueing Theory Validation (`queueing_theory/`)

Validates simulation results against M/G/1 queueing theory predictions.

```bash
./scripts/run_queueing_validation.sh
```

**Main Features**:
- P-K formula validation for FCFS scheduler
- Kleinrock multi-class priority formula for SSJF-Emotion
- Load sweep (ρ = 0.5 ~ 0.9)
- Error analysis (transient removal, finite sample correction)

**Output**: `results/queueing_validation/validation_results/`

### 4. Legacy Plotting (`plot_emotion_results.py`)

> **DEPRECATED**: Use `plotting/` module instead.

Kept for backward compatibility with older scripts.

## Experiment-to-Module Mapping

| Experiment | Data Location | Analysis Script | Plotting Module |
|------------|---------------|-----------------|-----------------|
| Multi-seed | `results/multi_seed_runs/` | `aggregate_results.py` | `plotting/plots/forest.py` |
| Param sweep | `results/param_sweep/` | - | `plotting/plots/heatmap.py` |
| Starvation | `results/starvation_sweep/` | - | `plotting/plots/pareto.py` |
| Shuffle ablation | `results/shuffle_experiment/` | - | `plotting/plots/ablation.py` |
| Robustness | `results/robustness_experiment/` | - | `plotting/plots/ablation.py` |
| Queueing validation | `results/queueing_validation/` | `queueing_theory/validation.py` | `queueing_theory/plotting.py` |

## Usage Examples

### Generate All Publication Plots

```python
from analysis.plotting import generate_all_publication_plots

generate_all_publication_plots(
    aggregated_results_path='results/multi_seed_runs/aggregated_results.json',
    output_dir='results/publication_plots'
)
```

### Run Queueing Theory Validation

```python
from analysis.queueing_theory import validate_load_sweep, generate_validation_report

results = validate_load_sweep(
    results_dir='results/queueing_validation',
    load_levels=[0.5, 0.6, 0.7, 0.8, 0.9],
    schedulers=['FCFS', 'SSJF-Emotion']
)

report = generate_validation_report(results, output_path='validation_report.md')
```

## Dependencies

- `pandas`, `numpy` - Data processing
- `matplotlib`, `seaborn` - Visualization
- `scipy` - Statistical analysis
