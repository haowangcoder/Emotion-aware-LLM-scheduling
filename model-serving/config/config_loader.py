"""
Configuration Loader for Emotion-aware LLM Scheduling System.

Provides YAML-based hierarchical configuration with environment variable
and CLI argument override support.
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
    """Service time mapping configuration."""
    base_service_time: float = 2.0
    alpha: float = 0.5
    rho: float = 1.0
    min_service_time: float = 0.1
    mapping_function: str = 'linear'


@dataclass
class ArrivalConfig:
    """Task arrival configuration."""
    base_arrival_rate: float = 2.0


@dataclass
class EmotionConfig:
    """Emotion parameters configuration."""
    arousal_noise_std: float = 0.0
    enable_emotion_aware: bool = True


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
class EmotionLengthControlConfig:
    """Emotion-aware response length control."""
    enabled: bool = True
    base_response_length: int = 100
    # Note: alpha is synchronized from workload.service_time.alpha


@dataclass
class PromptConfig:
    """Prompt configuration."""
    include_system_prompt: bool = True
    include_emotion_hint: bool = False
    max_conversation_turns: int = 2
    emotion_length_control: EmotionLengthControlConfig = field(default_factory=EmotionLengthControlConfig)


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
class SchedulerConfig:
    """Scheduler configuration."""
    algorithm: str = 'FCFS'
    system_load: float = 0.6
    starvation_prevention: StarvationPreventionConfig = field(default_factory=StarvationPreventionConfig)


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    emotion_dataset_path: str = './dataset'


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    num_jobs: int = 100
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
            ('WORKLOAD_SERVICE_TIME_ALPHA', lambda c, v: setattr(c.workload.service_time, 'alpha', float(v))),
            ('WORKLOAD_SERVICE_TIME_RHO', lambda c, v: setattr(c.workload.service_time, 'rho', float(v))),
            ('WORKLOAD_SERVICE_TIME_MIN_SERVICE_TIME', lambda c, v: setattr(c.workload.service_time, 'min_service_time', float(v))),
            ('WORKLOAD_SERVICE_TIME_MAPPING_FUNCTION', lambda c, v: setattr(c.workload.service_time, 'mapping_function', v)),
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
            ('LLM_GENERATION_TOP_P', lambda c, v: setattr(c.llm.generation, 'top_p', float(v))),
            ('LLM_GENERATION_DO_SAMPLE', lambda c, v: setattr(c.llm.generation, 'do_sample', v.lower() == 'true')),
            ('LLM_GENERATION_REPETITION_PENALTY', lambda c, v: setattr(c.llm.generation, 'repetition_penalty', float(v))),

            # LLM Prompt
            ('LLM_PROMPT_INCLUDE_SYSTEM_PROMPT', lambda c, v: setattr(c.llm.prompt, 'include_system_prompt', v.lower() == 'true')),
            ('LLM_PROMPT_INCLUDE_EMOTION_HINT', lambda c, v: setattr(c.llm.prompt, 'include_emotion_hint', v.lower() == 'true')),
            ('LLM_PROMPT_MAX_CONVERSATION_TURNS', lambda c, v: setattr(c.llm.prompt, 'max_conversation_turns', int(v))),
            ('LLM_PROMPT_EMOTION_LENGTH_CONTROL_ENABLED', lambda c, v: setattr(c.llm.prompt.emotion_length_control, 'enabled', v.lower() == 'true')),
            ('LLM_PROMPT_EMOTION_LENGTH_CONTROL_BASE_RESPONSE_LENGTH', lambda c, v: setattr(c.llm.prompt.emotion_length_control, 'base_response_length', int(v))),

            # LLM Cache
            ('LLM_CACHE_USE_RESPONSE_CACHE', lambda c, v: setattr(c.llm.cache, 'use_response_cache', v.lower() == 'true')),
            ('LLM_CACHE_CACHE_DIR', lambda c, v: setattr(c.llm.cache, 'cache_dir', v)),
            ('LLM_CACHE_FORCE_REGENERATE', lambda c, v: setattr(c.llm.cache, 'force_regenerate', v.lower() == 'true')),

            # Scheduler
            ('SCHEDULER_ALGORITHM', lambda c, v: setattr(c.scheduler, 'algorithm', v)),
            ('SCHEDULER_SYSTEM_LOAD', lambda c, v: setattr(c.scheduler, 'system_load', float(v))),
            ('SCHEDULER_STARVATION_PREVENTION_THRESHOLD', lambda c, v: setattr(c.scheduler.starvation_prevention, 'threshold', float(v))),
            ('SCHEDULER_STARVATION_PREVENTION_COEFFICIENT', lambda c, v: setattr(c.scheduler.starvation_prevention, 'coefficient', float(v))),

            # Dataset
            ('DATASET_EMOTION_DATASET_PATH', lambda c, v: setattr(c.dataset, 'emotion_dataset_path', v)),

            # Experiment
            ('EXPERIMENT_NUM_JOBS', lambda c, v: setattr(c.experiment, 'num_jobs', int(v))),
            ('EXPERIMENT_RANDOM_SEED', lambda c, v: setattr(c.experiment, 'random_seed', int(v) if v.lower() != 'null' else None)),

            # Output
            ('OUTPUT_RESULTS_DIR', lambda c, v: setattr(c.output, 'results_dir', v)),
            ('OUTPUT_VERBOSE', lambda c, v: setattr(c.output, 'verbose', v.lower() == 'true')),

            # Backward compatibility with old env var names
            ('ALPHA_ENV', lambda c, v: setattr(c.workload.service_time, 'alpha', float(v))),
            ('BASE_SERVICE_TIME_ENV', lambda c, v: setattr(c.workload.service_time, 'base_service_time', float(v))),
            ('RHO_ENV', lambda c, v: setattr(c.workload.service_time, 'rho', float(v))),
            ('LLM_MODEL_NAME_ENV', lambda c, v: setattr(c.llm.model, 'name', v)),
            ('LLM_DEVICE_MAP_ENV', lambda c, v: setattr(c.llm.model, 'device_map', v)),
            ('EMOTION_DATASET_PATH_ENV', lambda c, v: setattr(c.dataset, 'emotion_dataset_path', v)),
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
            'system_load': lambda c, v: setattr(c.scheduler, 'system_load', v),
            'base_service_time': lambda c, v: setattr(c.workload.service_time, 'base_service_time', v),
            'alpha': lambda c, v: setattr(c.workload.service_time, 'alpha', v),
            'rho': lambda c, v: setattr(c.workload.service_time, 'rho', v),
            'enable_emotion': lambda c, v: setattr(c.workload.emotion, 'enable_emotion_aware', v),
            'arousal_noise': lambda c, v: setattr(c.workload.emotion, 'arousal_noise_std', v),
            'random_seed': lambda c, v: setattr(c.experiment, 'random_seed', v),
            'starvation_threshold': lambda c, v: setattr(c.scheduler.starvation_prevention, 'threshold', v),
            'starvation_coefficient': lambda c, v: setattr(c.scheduler.starvation_prevention, 'coefficient', v),
            'output_dir': lambda c, v: setattr(c.output, 'results_dir', v),
            'verbose': lambda c, v: setattr(c.output, 'verbose', v),
            'model_name': lambda c, v: setattr(c.llm.model, 'name', v),
            'dataset_path': lambda c, v: setattr(c.dataset, 'emotion_dataset_path', v),
            'use_cache': lambda c, v: setattr(c.llm.cache, 'use_response_cache', v),
            'force_regenerate': lambda c, v: setattr(c.llm.cache, 'force_regenerate', v),
            'device_map': lambda c, v: setattr(c.llm.model, 'device_map', v),
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
        # Check alpha synchronization
        alpha = config.workload.service_time.alpha
        if not (0 < alpha <= 2.0):
            warnings.warn(f"Alpha value {alpha} is outside typical range (0, 2.0]")

        # Check system load
        if not (0 < config.scheduler.system_load < 1):
            warnings.warn(f"System load {config.scheduler.system_load} should be in range (0, 1)")

        # Check base service time
        if config.workload.service_time.base_service_time <= 0:
            raise ValueError("base_service_time must be positive")

        # Info: Alpha is synchronized
        if config.output.verbose:
            print(f"[Config] Alpha parameter synchronized: {alpha}")
            print(f"         - Service time mapping: S_i = {config.workload.service_time.base_service_time} * (1 + {alpha} * a_i)")
            print(f"         - Response length: L_i = {config.llm.prompt.emotion_length_control.base_response_length} * (1 + {alpha} * a_i)")

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
    return ConfigLoader.load(config_path, cli_args)


def get_alpha(config: Config) -> float:
    """
    Get the synchronized alpha parameter.

    This is used for both service time mapping and LLM response length control.
    """
    return config.workload.service_time.alpha
