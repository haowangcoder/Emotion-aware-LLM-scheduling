"""
Configuration package for Affect-Aware LLM Scheduling.

Use the YAML-based configuration loader:

    from config.config_loader import load_config

New in AW-SSJF Refactor:
    - AffectWeightConfig: Depression-First weight parameters
    - LengthPredictorConfig: BERT-based length prediction settings
    - get_affect_weight_params(): Get affect weight parameters
    - get_length_predictor_config(): Get length predictor configuration
"""

from .config_loader import (
    load_config,
    get_affect_weight_params,
    get_length_predictor_config,
    Config,
    AffectWeightConfig,
    LengthPredictorConfig,
)

__all__ = [
    "load_config",
    "get_affect_weight_params",
    "get_length_predictor_config",
    "Config",
    "AffectWeightConfig",
    "LengthPredictorConfig",
]
