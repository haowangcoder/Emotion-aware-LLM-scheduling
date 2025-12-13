"""
Configuration Loader for Affect-Aware LLM Scheduling System.

Provides YAML-based hierarchical configuration with environment variable
and CLI argument override support.

New in AW-SSJF Refactor:
    - AffectWeightConfig: Depression-First weight parameters
    - LengthPredictorConfig: BERT-based length prediction settings
    - Removed: alpha, emotion_correlation, ValencePriorityConfig
"""
import os
import yaml
import warnings
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, fields


# ============================================================================
# Configuration Data Classes
# ============================================================================

@dataclass
class ServiceTimeConfig:
    """Service time configuration (simplified)."""
    base_service_time: float = 2.0
    min_service_time: float = 0.1


@dataclass
class ArrivalConfig:
    """Task arrival configuration."""
    base_arrival_rate: float = 2.0


@dataclass
class EmotionConfig:
    """Emotion parameters configuration."""
    arousal_noise_std: float = 0.0
    valence_noise_std: float = 0.0
    enable_emotion_aware: bool = True
    use_stratified_sampling: bool = True
    quadrant_distribution: str = 'uniform'  # 'uniform' or dict


@dataclass
class WorkloadConfig:
    """Workload generation configuration."""
    service_time: ServiceTimeConfig = field(default_factory=ServiceTimeConfig)
    arrival: ArrivalConfig = field(default_factory=ArrivalConfig)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)


@dataclass
class ModelConfig:
    """LLM model configuration."""
    name: str = 'meta-llama/Meta-Llama-3-8B-Instruct'
    device_map: str = 'auto'
    dtype: str = 'auto'
    trust_remote_code: bool = True
    load_in_8bit: bool = False


@dataclass
class GenerationConfig:
    """LLM generation parameters."""
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = False
    repetition_penalty: float = 1.1


@dataclass
class PromptConfig:
    """Prompt configuration (simplified)."""
    include_system_prompt: bool = True
    include_emotion_hint: bool = False
    max_conversation_turns: int = 2


@dataclass
class CacheConfig:
    """Response caching configuration."""
    use_response_cache: bool = True
    cache_dir: str = 'results/cache'
    cache_file: str = 'responses.json'
    force_regenerate: bool = False
    use_saved_job_config: bool = True
    job_config_file: str = 'job_configs.json'
    force_new_job_config: bool = False


@dataclass
class ErrorHandlingConfig:
    """Error handling configuration."""
    max_retries: int = 2
    enable_cpu_fallback: bool = True
    skip_on_error: bool = True


@dataclass
class LLMConfig:
    """LLM inference configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    error_handling: ErrorHandlingConfig = field(default_factory=ErrorHandlingConfig)


@dataclass
class StarvationPreventionConfig:
    """Starvation prevention configuration."""
    threshold: float = float('inf')
    coefficient: float = 3.0


@dataclass
class AffectWeightConfig:
    """
    Affect weight configuration for Depression-First scheduling.

    The affect weight is computed as:
        w = 1 + (w_max - 1) * c * u

    where:
        - u = n^p * ell^q is the urgency (depression intensity)
        - n = max(0, -valence) is the unpleasant intensity
        - ell = max(0, -arousal) is the low arousal intensity
        - c is the emotion recognition confidence
    """
    weight_mode: str = 'hard'   # 'hard', 'soft', 'dual', or preset name
    weight_preset: Optional[str] = None  # Optional explicit preset name
    w_max: float = 2.0  # Maximum weight, recommended [1.2, 3.0]
    p: float = 1.0      # Exponent for negative valence
    q: float = 1.0      # Exponent for low arousal
    use_confidence: bool = True  # Apply confidence discount
    # Soft gating parameters (v2)
    k_v: float = 5.0
    k_a: float = 5.0
    tau_v: float = 0.0
    tau_a: float = 0.0
    # Dual-channel parameters (v2)
    r: float = 1.0
    tau_h: float = 0.0
    gamma_dep: float = 1.0
    gamma_panic: float = 0.3


@dataclass
class SchedulerConfig:
    """Scheduler configuration."""
    algorithm: str = 'FCFS'  # FCFS, SSJF, SJF, Weight-Only, AW-SSJF
    system_load: float = 0.6
    use_robust_scoring: bool = False  # Use log(S+1)/w instead of S/w for prediction error tolerance
    use_conservative_prediction: bool = False  # Multiply predicted S by margin to reduce underestimation
    conservative_margin: float = 1.3  # Safety margin multiplier (1.3 = 30% increase)
    weight_exponent: float = 1.0  # Exponent k for w^k: 1=standard, 2=squared (amplifies weight influence)
    starvation_prevention: StarvationPreventionConfig = field(default_factory=StarvationPreventionConfig)
    affect_weight: AffectWeightConfig = field(default_factory=AffectWeightConfig)


@dataclass
class LengthPredictorConfig:
    """
    BERT bucket predictor configuration.

    Uses classification into bins with expected value method:
        T_mean = sum(q_i * m_i)
        S = const_latency + T_mean * per_token_latency
    """
    enabled: bool = False  # Disabled by default (requires trained model)
    model_path: str = 'predictor/models/bert_bucket'  # HuggingFace model directory
    bin_edges_path: str = 'predictor/models/bin_edges.npy'  # Bin edges file
    model_name: str = 'distilbert-base-uncased'
    device: str = 'cuda'
    num_bins: int = 5  # Number of classification bins
    per_token_latency: float = 0.02  # c_1: Latency per generated token (seconds)
    const_latency: float = 0.1       # c_0: Constant latency overhead (seconds)
    default_service_time: float = 2.0  # Fallback when predictor unavailable


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    emotion_dataset_path: str = './dataset'


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    mode: str = 'fixed_jobs'  # 'fixed_jobs' or 'time_window'
    num_jobs: int = 100
    simulation_duration: float = 300.0
    random_seed: Optional[int] = None
    fairness_metric: str = 'waiting_time'
    calculate_fairness: bool = True


@dataclass
class OutputConfig:
    """Output and logging configuration."""
    results_dir: str = 'results/llm_runs/'
    verbose: bool = False


@dataclass
class Config:
    """Root configuration object."""
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    length_predictor: LengthPredictorConfig = field(default_factory=LengthPredictorConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


# ============================================================================
# Configuration Loader
# ============================================================================

class ConfigLoader:
    """
    Loads and manages hierarchical configuration with override support.

    Configuration precedence (lowest to highest):
    1. Default values in dataclasses
    2. default.yaml file
    3. Environment variables
    4. CLI arguments (passed explicitly)
    """

    DEFAULT_CONFIG_PATH = Path(__file__).parent / 'default.yaml'

    @staticmethod
    def load_yaml(config_path: Optional[Path] = None) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if config_path is None:
            config_path = ConfigLoader.DEFAULT_CONFIG_PATH

        if not config_path.exists():
            warnings.warn(f"Config file not found: {config_path}. Using defaults.")
            return {}

        with open(config_path, 'r') as f:
            try:
                config_dict = yaml.safe_load(f) or {}
                return config_dict
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML config: {e}")

    @staticmethod
    def _update_from_dict(obj: Any, data: Dict[str, Any], prefix: str = "") -> None:
        """Recursively update dataclass from dictionary."""
        if not hasattr(obj, '__dataclass_fields__'):
            return

        for field_info in fields(obj):
            field_name = field_info.name
            field_value = data.get(field_name)

            if field_value is None:
                continue

            current_value = getattr(obj, field_name)

            # If nested dataclass, recurse
            if hasattr(current_value, '__dataclass_fields__'):
                if isinstance(field_value, dict):
                    ConfigLoader._update_from_dict(current_value, field_value,
                                                   f"{prefix}.{field_name}" if prefix else field_name)
            else:
                # Handle special case for .inf in YAML
                if isinstance(field_value, str) and field_value == '.inf':
                    field_value = float('inf')
                setattr(obj, field_name, field_value)

    @staticmethod
    def _apply_env_overrides(config: Config) -> None:
        """Apply environment variable overrides to configuration."""
        env_mappings = [
            # Workload
            ('WORKLOAD_SERVICE_TIME_BASE_SERVICE_TIME', lambda c, v: setattr(c.workload.service_time, 'base_service_time', float(v))),
            ('WORKLOAD_SERVICE_TIME_MIN_SERVICE_TIME', lambda c, v: setattr(c.workload.service_time, 'min_service_time', float(v))),
            ('WORKLOAD_ARRIVAL_BASE_ARRIVAL_RATE', lambda c, v: setattr(c.workload.arrival, 'base_arrival_rate', float(v))),
            ('WORKLOAD_EMOTION_AROUSAL_NOISE_STD', lambda c, v: setattr(c.workload.emotion, 'arousal_noise_std', float(v))),
            ('WORKLOAD_EMOTION_ENABLE_EMOTION_AWARE', lambda c, v: setattr(c.workload.emotion, 'enable_emotion_aware', v.lower() == 'true')),

            # LLM Model
            ('LLM_MODEL_NAME', lambda c, v: setattr(c.llm.model, 'name', v)),
            ('LLM_MODEL_DEVICE_MAP', lambda c, v: setattr(c.llm.model, 'device_map', v)),
            ('LLM_MODEL_DTYPE', lambda c, v: setattr(c.llm.model, 'dtype', v)),
            ('LLM_MODEL_LOAD_IN_8BIT', lambda c, v: setattr(c.llm.model, 'load_in_8bit', v.lower() == 'true')),

            # LLM Generation
            ('LLM_GENERATION_MAX_NEW_TOKENS', lambda c, v: setattr(c.llm.generation, 'max_new_tokens', int(v))),
            ('LLM_GENERATION_TEMPERATURE', lambda c, v: setattr(c.llm.generation, 'temperature', float(v))),

            # Scheduler
            ('SCHEDULER_ALGORITHM', lambda c, v: setattr(c.scheduler, 'algorithm', v)),
            ('SCHEDULER_SYSTEM_LOAD', lambda c, v: setattr(c.scheduler, 'system_load', float(v))),
            ('SCHEDULER_STARVATION_PREVENTION_THRESHOLD', lambda c, v: setattr(c.scheduler.starvation_prevention, 'threshold', float(v))),
            ('SCHEDULER_STARVATION_PREVENTION_COEFFICIENT', lambda c, v: setattr(c.scheduler.starvation_prevention, 'coefficient', float(v))),

            # Affect Weight
            ('SCHEDULER_AFFECT_WEIGHT_W_MAX', lambda c, v: setattr(c.scheduler.affect_weight, 'w_max', float(v))),
            ('SCHEDULER_AFFECT_WEIGHT_P', lambda c, v: setattr(c.scheduler.affect_weight, 'p', float(v))),
            ('SCHEDULER_AFFECT_WEIGHT_Q', lambda c, v: setattr(c.scheduler.affect_weight, 'q', float(v))),
            ('SCHEDULER_AFFECT_WEIGHT_USE_CONFIDENCE', lambda c, v: setattr(c.scheduler.affect_weight, 'use_confidence', v.lower() == 'true')),

            # Length Predictor
            ('LENGTH_PREDICTOR_ENABLED', lambda c, v: setattr(c.length_predictor, 'enabled', v.lower() == 'true')),
            ('LENGTH_PREDICTOR_MODEL_PATH', lambda c, v: setattr(c.length_predictor, 'model_path', v)),
            ('LENGTH_PREDICTOR_BIN_EDGES_PATH', lambda c, v: setattr(c.length_predictor, 'bin_edges_path', v)),
            ('LENGTH_PREDICTOR_DEVICE', lambda c, v: setattr(c.length_predictor, 'device', v)),
            ('LENGTH_PREDICTOR_DEFAULT_SERVICE_TIME', lambda c, v: setattr(c.length_predictor, 'default_service_time', float(v))),

            # Dataset
            ('DATASET_EMOTION_DATASET_PATH', lambda c, v: setattr(c.dataset, 'emotion_dataset_path', v)),

            # Experiment
            ('EXPERIMENT_NUM_JOBS', lambda c, v: setattr(c.experiment, 'num_jobs', int(v))),
            ('EXPERIMENT_RANDOM_SEED', lambda c, v: setattr(c.experiment, 'random_seed', int(v) if v.lower() != 'null' else None)),

            # Output
            ('OUTPUT_RESULTS_DIR', lambda c, v: setattr(c.output, 'results_dir', v)),
            ('OUTPUT_VERBOSE', lambda c, v: setattr(c.output, 'verbose', v.lower() == 'true')),
        ]

        for env_var, setter in env_mappings:
            value = os.environ.get(env_var)
            if value is not None:
                try:
                    setter(config, value)
                except Exception as e:
                    warnings.warn(f"Failed to apply env override {env_var}={value}: {e}")

    @staticmethod
    def _apply_cli_overrides(config: Config, cli_args: Optional[Dict[str, Any]] = None) -> None:
        """Apply CLI argument overrides to configuration."""
        if cli_args is None:
            return

        # Map CLI argument names to config attributes
        cli_mappings = {
            'scheduler': lambda c, v: setattr(c.scheduler, 'algorithm', v),
            'num_jobs': lambda c, v: setattr(c.experiment, 'num_jobs', v),
            'mode': lambda c, v: setattr(c.experiment, 'mode', v),
            'simulation_duration': lambda c, v: setattr(c.experiment, 'simulation_duration', v),
            'system_load': lambda c, v: setattr(c.scheduler, 'system_load', v),
            'base_service_time': lambda c, v: setattr(c.workload.service_time, 'base_service_time', v),
            'enable_emotion': lambda c, v: setattr(c.workload.emotion, 'enable_emotion_aware', v),
            'arousal_noise': lambda c, v: setattr(c.workload.emotion, 'arousal_noise_std', v),
            'random_seed': lambda c, v: setattr(c.experiment, 'random_seed', v),
            'starvation_threshold': lambda c, v: setattr(c.scheduler.starvation_prevention, 'threshold', v),
            'starvation_coefficient': lambda c, v: setattr(c.scheduler.starvation_prevention, 'coefficient', v),
            'use_robust_scoring': lambda c, v: setattr(c.scheduler, 'use_robust_scoring', v),
            'use_conservative_prediction': lambda c, v: setattr(c.scheduler, 'use_conservative_prediction', v),
            'conservative_margin': lambda c, v: setattr(c.scheduler, 'conservative_margin', v),
            'weight_exponent': lambda c, v: setattr(c.scheduler, 'weight_exponent', v),

            # Affect weight parameters
            'w_max': lambda c, v: setattr(c.scheduler.affect_weight, 'w_max', v),
            'p': lambda c, v: setattr(c.scheduler.affect_weight, 'p', v),
            'q': lambda c, v: setattr(c.scheduler.affect_weight, 'q', v),
            'use_confidence': lambda c, v: setattr(c.scheduler.affect_weight, 'use_confidence', v),

            # Length predictor
            'enable_predictor': lambda c, v: setattr(c.length_predictor, 'enabled', v),
            'predictor_model_path': lambda c, v: setattr(c.length_predictor, 'model_path', v),
            'predictor_bin_edges_path': lambda c, v: setattr(c.length_predictor, 'bin_edges_path', v),
            'default_service_time': lambda c, v: setattr(c.length_predictor, 'default_service_time', v),

            # Output and misc
            'output_dir': lambda c, v: setattr(c.output, 'results_dir', v),
            'verbose': lambda c, v: setattr(c.output, 'verbose', v),
            'model_name': lambda c, v: setattr(c.llm.model, 'name', v),
            'dataset_path': lambda c, v: setattr(c.dataset, 'emotion_dataset_path', v),
            'use_cache': lambda c, v: setattr(c.llm.cache, 'use_response_cache', v),
            'force_regenerate': lambda c, v: setattr(c.llm.cache, 'force_regenerate', v),
            'device_map': lambda c, v: setattr(c.llm.model, 'device_map', v),
            'dtype': lambda c, v: setattr(c.llm.model, 'dtype', v),
            'load_in_8bit': lambda c, v: setattr(c.llm.model, 'load_in_8bit', v),

            # Job config caching
            'force_new_job_config': lambda c, v: setattr(c.llm.cache, 'force_new_job_config', v),
            'use_saved_job_config': lambda c, v: setattr(c.llm.cache, 'use_saved_job_config', v),
        }

        for arg_name, setter in cli_mappings.items():
            if arg_name in cli_args and cli_args[arg_name] is not None:
                try:
                    setter(config, cli_args[arg_name])
                except Exception as e:
                    warnings.warn(f"Failed to apply CLI override {arg_name}={cli_args[arg_name]}: {e}")

    @staticmethod
    def _validate_config(config: Config) -> None:
        """Validate configuration and issue warnings for potential issues."""
        # Check system load
        if not (0 < config.scheduler.system_load < 1):
            warnings.warn(f"System load {config.scheduler.system_load} should be in range (0, 1)")

        # Check base service time
        if config.workload.service_time.base_service_time <= 0:
            raise ValueError("base_service_time must be positive")

        # Check affect weight parameters
        if config.scheduler.affect_weight.w_max < 1.0:
            warnings.warn(f"w_max={config.scheduler.affect_weight.w_max} should be >= 1.0")

        # Info output
        if config.output.verbose:
            print(f"[Config] Scheduler: {config.scheduler.algorithm}")
            print(f"         Affect Weight: w_max={config.scheduler.affect_weight.w_max}, "
                  f"p={config.scheduler.affect_weight.p}, q={config.scheduler.affect_weight.q}")
            print(f"         Length Predictor: {'enabled' if config.length_predictor.enabled else 'disabled'}")

    @classmethod
    def load(cls, config_path: Optional[Path] = None, cli_args: Optional[Dict[str, Any]] = None) -> Config:
        """
        Load configuration with override support.

        Args:
            config_path: Path to YAML config file (default: default.yaml)
            cli_args: Dictionary of CLI arguments to override config

        Returns:
            Config object with all overrides applied
        """
        # 1. Start with default dataclass values
        config = Config()

        # 2. Load and apply YAML config
        yaml_data = cls.load_yaml(config_path)
        cls._update_from_dict(config, yaml_data)

        # 3. Apply environment variable overrides
        cls._apply_env_overrides(config)

        # 4. Apply CLI argument overrides
        cls._apply_cli_overrides(config, cli_args)

        # 5. Validate configuration
        cls._validate_config(config)

        return config

    @staticmethod
    def to_dict(config: Config) -> Dict[str, Any]:
        """Convert config object to dictionary."""
        def dataclass_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {
                    field.name: dataclass_to_dict(getattr(obj, field.name))
                    for field in fields(obj)
                }
            return obj

        return dataclass_to_dict(config)


# ============================================================================
# Convenience Functions
# ============================================================================

def load_config(config_path: Optional[Path] = None, cli_args: Optional[Dict[str, Any]] = None) -> Config:
    """Convenience function to load configuration."""
    config = ConfigLoader.load(config_path, cli_args)

    # Patch cache paths to follow output_dir
    output_dir = config.output.results_dir
    cache_base = os.path.join(output_dir, "cache")

    config.llm.cache.cache_dir = cache_base
    config.llm.cache.full_cache_path = os.path.join(cache_base, config.llm.cache.cache_file)
    config.llm.cache.full_job_config_path = os.path.join(cache_base, config.llm.cache.job_config_file)

    os.makedirs(cache_base, exist_ok=True)
    return config


def get_affect_weight_params(config: Config) -> Dict[str, Any]:
    """Get affect weight parameters as dictionary."""
    return {
        'w_max': config.scheduler.affect_weight.w_max,
        'p': config.scheduler.affect_weight.p,
        'q': config.scheduler.affect_weight.q,
        'use_confidence': config.scheduler.affect_weight.use_confidence,
    }


def get_length_predictor_config(config: Config) -> Dict[str, Any]:
    """Get length predictor configuration as dictionary."""
    return {
        'enabled': config.length_predictor.enabled,
        'model_path': config.length_predictor.model_path,
        'bin_edges_path': config.length_predictor.bin_edges_path,
        'model_name': config.length_predictor.model_name,
        'device': config.length_predictor.device,
        'num_bins': config.length_predictor.num_bins,
        'per_token_latency': config.length_predictor.per_token_latency,
        'const_latency': config.length_predictor.const_latency,
        'default_service_time': config.length_predictor.default_service_time,
    }
