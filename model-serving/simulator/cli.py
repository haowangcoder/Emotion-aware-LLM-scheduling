import argparse

from .experiment import run_emotion_aware_experiment


def build_parser() -> argparse.ArgumentParser:
    """
    Build argument parser for the simulator CLI.
    """
    parser = argparse.ArgumentParser(
        description="Affect-Aware LLM Scheduling Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core arguments
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        choices=["FCFS", "SJF", "SSJF", "AW-SSJF", "Weight-Only"],
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

    # Affect weight configuration (AW-SSJF parameters)
    parser.add_argument(
        "--w_max",
        type=float,
        default=None,
        help="Maximum affect weight (overrides config.scheduler.affect_weight.w_max). Range: [1.2, 3.0]",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=None,
        help="Negative valence exponent (overrides config.scheduler.affect_weight.p). Default: 1.0",
    )
    parser.add_argument(
        "--q",
        type=float,
        default=None,
        help="Low arousal exponent (overrides config.scheduler.affect_weight.q). Default: 1.0",
    )
    parser.add_argument(
        "--use_confidence",
        action="store_true",
        default=None,
        help="Use emotion confidence for weight discounting (overrides config.scheduler.affect_weight.use_confidence)",
    )
    parser.add_argument(
        "--no_confidence",
        dest="use_confidence",
        action="store_false",
        help="Disable emotion confidence discounting",
    )

    # v2.0 Weight parameters
    parser.add_argument(
        "--weight_exponent",
        type=float,
        default=None,
        help="Exponent k for w^k in scoring formula Score=S/w^k (overrides config.scheduler.weight_exponent)",
    )
    parser.add_argument(
        "--gamma_panic",
        type=float,
        default=None,
        help="Panic channel weight for DUAL_CHANNEL mode (overrides config.scheduler.affect_weight.gamma_panic)",
    )
    parser.add_argument(
        "--gamma_dep",
        type=float,
        default=None,
        help="Depression channel weight (overrides config.scheduler.affect_weight.gamma_dep)",
    )
    parser.add_argument(
        "--weight_mode",
        type=str,
        default=None,
        choices=["hard", "soft", "dual"],
        help="Weight computation mode (overrides config.scheduler.affect_weight.weight_mode)",
    )

    # Adaptive k (Online Control) parameters
    parser.add_argument(
        "--adaptive_k",
        action="store_true",
        default=None,
        help="Enable adaptive k control based on queue length (Exp-4 Online Control)",
    )
    parser.add_argument(
        "--adaptive_k_min",
        type=float,
        default=None,
        help="Minimum k value for adaptive control (overrides config.scheduler.adaptive_k_min)",
    )
    parser.add_argument(
        "--adaptive_k_max",
        type=float,
        default=None,
        help="Maximum k value for adaptive control (overrides config.scheduler.adaptive_k_max)",
    )
    parser.add_argument(
        "--adaptive_k_high_threshold",
        type=int,
        default=None,
        help="Queue length threshold to increase k (overrides config.scheduler.adaptive_k_high_threshold)",
    )
    parser.add_argument(
        "--adaptive_k_low_threshold",
        type=int,
        default=None,
        help="Queue length threshold to decrease k (overrides config.scheduler.adaptive_k_low_threshold)",
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
            "Absolute starvation threshold for schedulers "
            "(overrides config.scheduler.starvation_prevention.threshold)"
        ),
    )
    parser.add_argument(
        "--starvation_coefficient",
        type=float,
        default=None,
        help=(
            "Relative starvation coefficient for schedulers "
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

    # Predictor mode (for A2 defense experiment)
    predictor_group = parser.add_mutually_exclusive_group()
    predictor_group.add_argument(
        "--use_oracle_service_time",
        action="store_true",
        default=None,
        help="Use actual (oracle) service time instead of predicted (for A2 experiment)",
    )
    predictor_group.add_argument(
        "--disable_predictor",
        action="store_true",
        default=None,
        help="Disable BERT predictor, use default service time (for A2 experiment)",
    )

    # Job config caching (mutually exclusive)
    job_config_group = parser.add_mutually_exclusive_group()
    job_config_group.add_argument(
        "--force_new_job_config",
        action="store_true",
        default=None,
        help="Force generate new job configurations (overrides config.llm.cache.force_new_job_config)",
    )
    job_config_group.add_argument(
        "--use_saved_job_config",
        action="store_true",
        default=None,
        help="Use saved job configurations if available (overrides config.llm.cache.use_saved_job_config)",
    )

    # MMPP (burst traffic) parameters
    parser.add_argument(
        "--mmpp_enabled",
        action="store_true",
        default=None,
        help="Enable MMPP (Markov Modulated Poisson Process) for bursty traffic",
    )
    parser.add_argument(
        "--mmpp_lambda_high",
        type=float,
        default=None,
        help="MMPP high state arrival rate (burst period)",
    )
    parser.add_argument(
        "--mmpp_lambda_low",
        type=float,
        default=None,
        help="MMPP low state arrival rate (normal period)",
    )
    parser.add_argument(
        "--mmpp_alpha",
        type=float,
        default=None,
        help="MMPP HIGH->LOW transition rate (mean burst duration = 1/alpha)",
    )
    parser.add_argument(
        "--mmpp_beta",
        type=float,
        default=None,
        help="MMPP LOW->HIGH transition rate (mean normal duration = 1/beta)",
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
