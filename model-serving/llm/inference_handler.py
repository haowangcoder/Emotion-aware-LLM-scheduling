"""
LLM Inference Handler

Orchestrates all LLM inference components:
- LLMEngine: Model loading and generation
- EmpatheticDialoguesLoader: Dataset loading
- PromptBuilder: Prompt construction
- ResponseCache: Response caching

This module provides a high-level interface for the simulator to:
1. Initialize LLM components
2. Execute jobs with real model inference
3. Handle errors and fallbacks
4. Manage caching for reproducibility
"""

import time
import logging
from typing import Dict, Optional, Any

from llm.engine import LLMEngine
from llm.dataset_loader import EmpatheticDialoguesLoader
from llm.prompt_builder import PromptBuilder
from llm.response_cache import ResponseCache
from core.job import Job

logger = logging.getLogger(__name__)


class LLMInferenceHandler:
    """
    High-level handler for LLM inference in scheduling runs.

    Manages all LLM components and provides a simple interface
    for executing jobs with real model inference.
    """

    def __init__(
        self,
        model_name: str = 'meta-llama/Meta-Llama-3-8B-Instruct',
        dataset_path: str = './dataset',
        cache_path: Optional[str] = 'results/cache/responses.json',
        use_cache: bool = True,
        force_regenerate: bool = False,
        device_map: str = 'auto',
        dtype: str = 'auto',
        load_in_8bit: bool = False,
        # Prompt configuration
        include_emotion_hint: bool = False,
        enable_emotion_length_control: bool = True,
        base_response_length: int = 100,
        alpha: float = 0.5,
        max_conversation_turns: int = 2,
        # Generation parameters
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = False,
        repetition_penalty: float = 1.1,
        # Error handling
        max_retries: int = 2
    ):
        """
        Initialize LLM inference handler.

        Args:
            model_name: HuggingFace model identifier
            dataset_path: Path to EmpatheticDialogues dataset
            cache_path: Path to response cache file
            use_cache: Enable response caching
            force_regenerate: Force regenerate even if cached
            device_map: Device mapping for model
            dtype: Data type for model weights
            load_in_8bit: Use 8-bit quantization
            include_emotion_hint: Include emotion hints in prompts
            enable_emotion_length_control: Enable emotion-aware response length control
            base_response_length: Base response length L_0 in tokens
            alpha: Scaling factor α for arousal impact (synchronized with service time mapping)
            max_conversation_turns: Maximum conversation history turns to include
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            do_sample: Use sampling instead of greedy decoding
            repetition_penalty: Penalty for repeating tokens
            max_retries: Maximum retries on generation failure
        """
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.cache_path = cache_path
        self.use_cache = use_cache and not force_regenerate
        self.force_regenerate = force_regenerate
        self.max_retries = max_retries
        self.max_conversation_turns = max_conversation_turns

        # Initialize components
        logger.info("Initializing LLM Inference Handler...")

        # 1. Initialize LLM Engine
        logger.info(f"Loading LLM model: {model_name}")
        self.llm_engine = LLMEngine()
        success = self.llm_engine.load_model(
            model_name=model_name,
            device_map=device_map,
            dtype=dtype,
            load_in_8bit=load_in_8bit
        )

        if not success:
            raise RuntimeError(f"Failed to load LLM model: {model_name}")

        # 2. Initialize Dataset Loader
        logger.info(f"Loading EmpatheticDialogues dataset from: {dataset_path}")
        self.dataset_loader = EmpatheticDialoguesLoader(dataset_dir=dataset_path)
        success = self.dataset_loader.load(splits=["train", "valid"])

        if not success:
            raise RuntimeError(f"Failed to load dataset from: {dataset_path}")

        # 3. Initialize Prompt Builder
        self.prompt_builder = PromptBuilder(
            include_emotion_hint=include_emotion_hint,
            enable_emotion_length_control=enable_emotion_length_control,
            base_response_length=base_response_length,
            alpha=alpha
        )

        # 4. Initialize Response Cache
        if self.use_cache:
            logger.info(f"Initializing response cache: {cache_path}")
            self.response_cache = ResponseCache(cache_file=cache_path)
        else:
            logger.info("Response caching disabled")
            self.response_cache = None

        # Generation parameters
        self.gen_params = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": do_sample,
            "repetition_penalty": repetition_penalty
        }

        logger.info("LLM Inference Handler initialized successfully")

    def execute_job(self, job: Job, max_retries: Optional[int] = None) -> bool:
        """
        Execute a job using real LLM inference.

        Performs actual model generation and measures execution time.
        Updates the job object in-place with:
        - response_text: Generated response
        - actual_execution_duration: Real inference time
        - output_token_length: Number of tokens generated
        - cached: Whether response was from cache
        - error_msg: Error message if failed
        - fallback_used: Whether CPU fallback was used

        Args:
            job: Job object to execute
            max_retries: Maximum retry attempts on failure (uses instance default if None)

        Returns:
            bool: True if successful, False if failed
        """
        # Use instance max_retries if not specified
        if max_retries is None:
            max_retries = self.max_retries

        # Get conversation context from dataset based on emotion
        emotion = job.get_emotion_label()

        if not emotion:
            logger.error(f"Job {job.get_job_id()} has no emotion label")
            job.set_error_msg("No emotion label")
            return False

        # Get arousal value for emotion-aware response length control
        arousal = job.get_arousal()

        # Get user context from dataset
        # Use conversation_index if available (for reproducibility)
        conversation_index = getattr(job, 'conversation_index', None)

        user_context, selected_index = self.dataset_loader.get_user_context_by_emotion(
            emotion=emotion.lower(),
            max_turns=self.max_conversation_turns,
            conversation_index=conversation_index
        )

        # Store the conversation index in the job for later saving
        if job.conversation_index is None and selected_index >= 0:
            job.conversation_index = selected_index

        if not user_context:
            logger.warning(f"No conversation found for emotion: {emotion}, using fallback")
            user_context = f"I'm feeling {emotion} right now."

        # Build prompt with arousal for emotion-aware response length
        prompt = self.prompt_builder.build_prompt(
            user_context=user_context,
            emotion=emotion,
            arousal=arousal
        )

        # Store conversation context in job
        job.set_conversation_context(prompt)

        # Check cache first (if enabled and not forcing regeneration)
        if self.use_cache and self.response_cache and not self.force_regenerate:
            cached_result = self.response_cache.get(prompt, self.model_name)

            if cached_result:
                logger.debug(f"Job {job.get_job_id()}: Using cached response")

                # Populate job with cached data
                job.set_response_text(cached_result["response_text"])
                job.set_actual_execution_duration(cached_result["execution_time"])
                job.set_output_token_length(cached_result["output_token_length"])
                job.set_cached(True)
                job.set_fallback_used(cached_result.get("fallback_used", False))
                job.set_model_name(self.model_name)

                # Also update execution_duration for scheduling clock
                job.set_execution_duration(cached_result["execution_time"])

                return True

        # Generate response with retries
        for attempt in range(max_retries + 1):
            try:
                logger.debug(f"Job {job.get_job_id()}: Generating response (attempt {attempt + 1}/{max_retries + 1})")

                result = self.llm_engine.generate(
                    prompt=prompt,
                    **self.gen_params
                )

                if result["error"]:
                    logger.warning(f"Job {job.get_job_id()}: Generation error: {result['error']}")

                    if attempt < max_retries:
                        logger.info(f"Retrying job {job.get_job_id()}...")
                        time.sleep(0.5)  # Brief delay before retry
                        continue
                    else:
                        job.set_error_msg(result["error"])
                        return False

                # Success! Populate job with results
                job.set_response_text(result["response_text"])
                job.set_actual_execution_duration(result["execution_time"])
                job.set_output_token_length(result["output_token_length"])
                job.set_cached(False)
                job.set_fallback_used(result["fallback_used"])
                job.set_model_name(self.model_name)

                # Update execution_duration for scheduling clock
                job.set_execution_duration(result["execution_time"])

                # Cache the result (if caching enabled)
                if self.use_cache and self.response_cache:
                    self.response_cache.set(
                        prompt=prompt,
                        response_text=result["response_text"],
                        execution_time=result["execution_time"],
                        output_token_length=result["output_token_length"],
                        model_name=self.model_name,
                        generation_params=self.gen_params,
                        error=None,
                        fallback_used=result["fallback_used"]
                    )

                logger.debug(f"Job {job.get_job_id()}: Generated {result['output_token_length']} tokens in {result['execution_time']:.3f}s")

                return True

            except Exception as e:
                logger.error(f"Job {job.get_job_id()}: Unexpected error: {e}")

                if attempt < max_retries:
                    logger.info(f"Retrying job {job.get_job_id()}...")
                    time.sleep(0.5)
                    continue
                else:
                    job.set_error_msg(str(e))
                    return False

        return False

    def save_cache(self) -> bool:
        """Save response cache to disk."""
        if self.response_cache:
            return self.response_cache.save_to_disk()
        return True

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.response_cache:
            return self.response_cache.get_stats()
        return {}

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return self.llm_engine.get_model_info()
