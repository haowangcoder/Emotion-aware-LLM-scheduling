import argparse

from .experiment import run_emotion_aware_experiment


def build_parser() -> argparse.ArgumentParser:
    """
    Build argument parser for the simulator CLI.
    """
    parser = argparse.ArgumentParser(
        description="Emotion-aware LLM Scheduling Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core arguments
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        choices=["FCFS", "SSJF-Emotion"],
        help="Scheduling algorithm (overrides config.scheduler.algorithm)",
    )
    parser.add_argument(
        "--num_jobs",
        type=int,
        default=None,
        help="Number of jobs to run (overrides config.experiment.num_jobs)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["fixed_jobs", "time_window"],
        help="Experiment mode (overrides config.experiment.mode)",
    )
    parser.add_argument(
        "--simulation_duration",
        type=float,
        default=None,
        help="Simulation duration for time_window mode (overrides config.experiment.simulation_duration)",
    )

    # System configuration
    parser.add_argument(
        "--system_load",
        type=float,
        default=None,
        help="Target system load (rho, overrides config.scheduler.system_load)",
    )
    parser.add_argument(
        "--base_service_time",
        type=float,
        default=None,
        help="Base service time L_0 (overrides config.workload.service_time.base_service_time)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Alpha parameter for arousal-service time mapping (overrides config.workload.service_time.alpha)",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=None,
        help="Correlation strength rho (overrides config.workload.service_time.rho)",
    )

    # Job generation
    emotion_group = parser.add_mutually_exclusive_group()
    emotion_group.add_argument(
        "--enable_emotion",
        dest="enable_emotion",
        action="store_true",
        default=None,
        help=(
            "Enable emotion-aware features "
            "(overrides config.workload.emotion.enable_emotion_aware)"
        ),
    )
    emotion_group.add_argument(
        "--disable_emotion",
        dest="enable_emotion",
        action="store_false",
        help=(
            "Disable emotion-aware features "
            "(overrides config.workload.emotion.enable_emotion_aware)"
        ),
    )
    parser.add_argument(
        "--arousal_noise",
        type=float,
        default=None,
        help=(
            "Standard deviation of arousal noise "
            "(overrides config.workload.emotion.arousal_noise_std)"
        ),
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help=(
            "Random seed for reproducibility "
            "(overrides config.experiment.random_seed)"
        ),
    )

    # Scheduler configuration
    parser.add_argument(
        "--starvation_threshold",
        type=float,
        default=None,
        help=(
            "Absolute starvation threshold for SSJF "
            "(overrides config.scheduler.starvation_prevention.threshold)"
        ),
    )
    parser.add_argument(
        "--starvation_coefficient",
        type=float,
        default=None,
        help=(
            "Relative starvation coefficient for SSJF "
            "(overrides config.scheduler.starvation_prevention.coefficient)"
        ),
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save results (overrides config.output.results_dir)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=None,
        help="Print detailed scheduling progress (overrides config.output.verbose)",
    )

    # LLM Inference Configuration (LLM-only)
    # Note: These are loaded from config by default
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="HuggingFace model identifier (overrides config)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to EmpatheticDialogues dataset (overrides config)",
    )
    parser.add_argument(
        "--use_cache",
        action="store_true",
        default=None,
        help="Enable response caching (overrides config)",
    )
    parser.add_argument(
        "--force_regenerate",
        action="store_true",
        default=None,
        help="Force regenerate responses (overrides config)",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        help="Device mapping for model (overrides config)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Data type for model weights (overrides config)",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        default=None,
        help="Use 8-bit quantization (overrides config)",
    )

    return parser


def main() -> None:
    """
    CLI entry point.
    """
    parser = build_parser()
    args = parser.parse_args()
    run_emotion_aware_experiment(args)


__all__ = ["build_parser", "main"]

