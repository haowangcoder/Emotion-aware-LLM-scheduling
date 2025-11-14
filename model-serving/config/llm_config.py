"""
LLM Inference Configuration for Real Model Integration.

Contains model loading, generation parameters, prompt configuration,
caching, and error handling settings.
"""
import os


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Model Configuration
LLM_MODEL_NAME = 'meta-llama/Meta-Llama-3-8B-Instruct'  # HuggingFace model identifier
LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME_ENV', LLM_MODEL_NAME)

# Model Loading Parameters
LLM_DEVICE_MAP = 'auto'  # 'auto', 'cuda', 'cpu', or specific GPU mapping
LLM_DTYPE = 'auto'  # 'auto', 'float16', 'bfloat16', 'float32'
LLM_TRUST_REMOTE_CODE = True  # Allow custom model code (needed for some models)
LLM_LOAD_IN_8BIT = False  # Use 8-bit quantization for memory efficiency

# Environment variable overrides
LLM_DEVICE_MAP = os.environ.get('LLM_DEVICE_MAP_ENV', LLM_DEVICE_MAP)
LLM_DTYPE = os.environ.get('LLM_DTYPE_ENV', LLM_DTYPE)
LLM_LOAD_IN_8BIT = os.environ.get('LLM_LOAD_IN_8BIT_ENV', 'False').lower() == 'true'


# ============================================================================
# GENERATION PARAMETERS
# ============================================================================

LLM_MAX_NEW_TOKENS = 64  # Maximum tokens to generate per response
LLM_TEMPERATURE = 0.7  # Sampling temperature (0.0 = greedy, 1.0 = more random)
LLM_TOP_P = 0.9  # Nucleus sampling threshold
LLM_DO_SAMPLE = True  # Use sampling instead of greedy decoding
LLM_REPETITION_PENALTY = 1.1  # Penalty for repeating tokens

# Environment variable overrides
LLM_MAX_NEW_TOKENS = int(os.environ.get('LLM_MAX_NEW_TOKENS_ENV', LLM_MAX_NEW_TOKENS))
LLM_TEMPERATURE = float(os.environ.get('LLM_TEMPERATURE_ENV', LLM_TEMPERATURE))
LLM_TOP_P = float(os.environ.get('LLM_TOP_P_ENV', LLM_TOP_P))
LLM_DO_SAMPLE = os.environ.get('LLM_DO_SAMPLE_ENV', 'True').lower() == 'true'
LLM_REPETITION_PENALTY = float(os.environ.get('LLM_REPETITION_PENALTY_ENV', LLM_REPETITION_PENALTY))


# ============================================================================
# PROMPT CONFIGURATION
# ============================================================================

LLM_INCLUDE_SYSTEM_PROMPT = True  # Include system instruction in prompt
LLM_INCLUDE_EMOTION_HINT = False  # Explicitly mention user's emotion in prompt
LLM_MAX_CONVERSATION_TURNS = 2  # Max turns of conversation history to include

# Environment variable overrides
LLM_INCLUDE_SYSTEM_PROMPT = os.environ.get('LLM_INCLUDE_SYSTEM_PROMPT_ENV', 'True').lower() == 'true'
LLM_INCLUDE_EMOTION_HINT = os.environ.get('LLM_INCLUDE_EMOTION_HINT_ENV', 'False').lower() == 'true'
LLM_MAX_CONVERSATION_TURNS = int(os.environ.get('LLM_MAX_CONVERSATION_TURNS_ENV', LLM_MAX_CONVERSATION_TURNS))


# ============================================================================
# CACHING CONFIGURATION
# ============================================================================

USE_RESPONSE_CACHE = True  # Enable response caching for reproducibility
RESPONSE_CACHE_DIR = 'results/cache'  # Directory to store cache files
RESPONSE_CACHE_PATH = os.path.join(RESPONSE_CACHE_DIR, 'responses.json')  # Cache file path
FORCE_REGENERATE = False  # Force regenerate even if cached (ignores cache)

# Job Configuration Cache for reproducible experiments
JOB_CONFIG_CACHE_FILE = os.path.join(RESPONSE_CACHE_DIR, 'job_configs.json')  # Job config file path
USE_SAVED_JOB_CONFIG = True  # Use saved job configurations if available
FORCE_NEW_JOB_CONFIG = False  # Force generate new job configurations (ignores saved)

# Environment variable overrides
USE_RESPONSE_CACHE = os.environ.get('USE_RESPONSE_CACHE_ENV', 'True').lower() == 'true'
RESPONSE_CACHE_DIR = os.environ.get('RESPONSE_CACHE_DIR_ENV', RESPONSE_CACHE_DIR)
RESPONSE_CACHE_PATH = os.environ.get('RESPONSE_CACHE_PATH_ENV', RESPONSE_CACHE_PATH)
FORCE_REGENERATE = os.environ.get('FORCE_REGENERATE_ENV', 'False').lower() == 'true'

JOB_CONFIG_CACHE_FILE = os.environ.get('JOB_CONFIG_CACHE_FILE_ENV', JOB_CONFIG_CACHE_FILE)
USE_SAVED_JOB_CONFIG = os.environ.get('USE_SAVED_JOB_CONFIG_ENV', 'True').lower() == 'true'
FORCE_NEW_JOB_CONFIG = os.environ.get('FORCE_NEW_JOB_CONFIG_ENV', 'False').lower() == 'true'


# ============================================================================
# ERROR HANDLING CONFIGURATION
# ============================================================================

LLM_MAX_RETRIES = 2  # Maximum retries on generation failure
LLM_ENABLE_CPU_FALLBACK = True  # Fallback to CPU if GPU OOM
LLM_SKIP_ON_ERROR = True  # Skip job on persistent errors (vs fail entire experiment)

# Environment variable overrides
LLM_MAX_RETRIES = int(os.environ.get('LLM_MAX_RETRIES_ENV', LLM_MAX_RETRIES))
LLM_ENABLE_CPU_FALLBACK = os.environ.get('LLM_ENABLE_CPU_FALLBACK_ENV', 'True').lower() == 'true'
LLM_SKIP_ON_ERROR = os.environ.get('LLM_SKIP_ON_ERROR_ENV', 'True').lower() == 'true'
