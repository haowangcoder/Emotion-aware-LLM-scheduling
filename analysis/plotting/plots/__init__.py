"""
Plot generation subpackage.

Contains specialized modules for each plot category:
- forest: Mean + CI comparison plots
- distribution: ECDF/CCDF plots
- pareto: Fairness-efficiency tradeoff plots
- slopegraph: Per-group analysis plots
- calibration: Prediction validation plots
- heatmap: 2D parameter sweep heatmaps
- ablation: Ablation study plots (shuffle experiments)
"""

from .forest import generate_forest_plots, plot_forest_metric, plot_forest_delta_vs_baseline
from .distribution import generate_distribution_plots, plot_ecdf, plot_ccdf_log
from .pareto import (
    generate_pareto_plots,
    plot_pareto_tradeoff,
    plot_starvation_pareto,
    generate_starvation_sweep_plots
)
from .slopegraph import (
    generate_group_analysis_plots,
    plot_slopegraph_arousal,
    plot_slopegraph_valence,
    plot_dumbbell_arousal
)
from .calibration import (
    generate_calibration_plots,
    plot_calibration_scatter,
    plot_calibration_binned,
    plot_error_distribution
)
from .heatmap import (
    plot_param_heatmap,
    plot_param_dual_line,
    generate_param_sweep_plots
)
from .ablation import (
    plot_shuffle_comparison,
    plot_shuffle_delta,
    generate_shuffle_experiment_plots,
    plot_robustness_heatmap,
    plot_robustness_grouped_bar,
    plot_robustness_delta,
    generate_robustness_experiment_plots,
)

__all__ = [
    # Forest
    'generate_forest_plots',
    'plot_forest_metric',
    'plot_forest_delta_vs_baseline',
    # Distribution
    'generate_distribution_plots',
    'plot_ecdf',
    'plot_ccdf_log',
    # Pareto
    'generate_pareto_plots',
    'plot_pareto_tradeoff',
    'plot_starvation_pareto',
    'generate_starvation_sweep_plots',
    # Slopegraph
    'generate_group_analysis_plots',
    'plot_slopegraph_arousal',
    'plot_slopegraph_valence',
    'plot_dumbbell_arousal',
    # Calibration
    'generate_calibration_plots',
    'plot_calibration_scatter',
    'plot_calibration_binned',
    'plot_error_distribution',
    # Heatmap
    'plot_param_heatmap',
    'plot_param_dual_line',
    'generate_param_sweep_plots',
    # Ablation
    'plot_shuffle_comparison',
    'plot_shuffle_delta',
    'generate_shuffle_experiment_plots',
    # Robustness
    'plot_robustness_heatmap',
    'plot_robustness_grouped_bar',
    'plot_robustness_delta',
    'generate_robustness_experiment_plots',
]
