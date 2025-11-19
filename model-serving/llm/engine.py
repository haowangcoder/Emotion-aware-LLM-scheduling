"""
LLM Engine Module for Real Model Inference

This module provides a singleton LLMEngine class that handles:
- Loading HuggingFace models (AutoModel + AutoTokenizer)
- Generating responses with time measurement
- Error handling (CUDA OOM, context overflow, etc.)
- GPU/CPU device management
"""

import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)
from typing import Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMEngine:
    """
    Singleton class for LLM model loading and inference.

    Designed for efficient model management with:
    - Lazy loading (model loaded on first use)
    - Device auto-detection (GPU if available, else CPU)
    - Error recovery (OOM fallback to CPU)
    - Time measurement for realistic scheduling runs
    """

    _instance = None
    _model = None
    _tokenizer = None
    _model_name = None
    _device = None
    _loaded = False

    def __new__(cls):
        """Singleton pattern: only one instance exists."""
        if cls._instance is None:
            cls._instance = super(LLMEngine, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize (only first time)."""
        pass

    def load_model(
        self,
        model_name: str,
        device_map: str = "auto",
        dtype: str = "auto",
        trust_remote_code: bool = True,
        load_in_8bit: bool = False
    ) -> bool:
        """
        Load HuggingFace model and tokenizer.

        Args:
            model_name: Model identifier (e.g., "meta-llama/Meta-Llama-3-8B-Instruct")
            device_map: Device placement ("auto", "cuda", "cpu")
            dtype: Data type ("auto", "float16", "bfloat16", "float32")
            trust_remote_code: Allow custom model code
            load_in_8bit: Use 8-bit quantization for memory efficiency

        Returns:
            bool: True if loaded successfully, False otherwise
        """
        if self._loaded and self._model_name == model_name:
            logger.info(f"Model {model_name} already loaded.")
            return True

        try:
            logger.info(f"Loading model: {model_name}")
            logger.info(f"Device map: {device_map}, dtype: {dtype}")

            # Check CUDA availability
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"CUDA available: {gpu_name} ({gpu_memory:.2f} GB)")
            else:
                logger.warning("CUDA not available, will use CPU (slower)")

            # Load tokenizer
            logger.info("Loading tokenizer...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code
            )

            # Ensure tokenizer has pad token
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
                logger.info(f"Set pad_token to eos_token: {self._tokenizer.eos_token}")

            # Prepare model loading kwargs
            model_kwargs = {
                "trust_remote_code": trust_remote_code,
            }

            # Handle dtype
            if dtype == "auto":
                model_kwargs["torch_dtype"] = "auto"
            elif dtype == "float16":
                model_kwargs["torch_dtype"] = torch.float16
            elif dtype == "bfloat16":
                model_kwargs["torch_dtype"] = torch.bfloat16
            elif dtype == "float32":
                model_kwargs["torch_dtype"] = torch.float32

            # Handle device mapping
            if device_map == "auto":
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = device_map

            # Handle 8-bit quantization
            if load_in_8bit:
                model_kwargs["load_in_8bit"] = True
                logger.info("Using 8-bit quantization")

            # Load model
            logger.info("Loading model weights (this may take a minute)...")
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )

            # Set to evaluation mode
            self._model.eval()

            # Store configuration
            self._model_name = model_name
            self._device = next(self._model.parameters()).device
            self._loaded = True

            logger.info(f"Model loaded successfully on device: {self._device}")
            logger.info(f"Model config max_position_embeddings: {self._model.config.max_position_embeddings}")

            return True

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM during model loading: {e}")
            logger.info("Attempting to load on CPU...")
            return self._load_on_cpu(model_name, trust_remote_code)

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _load_on_cpu(self, model_name: str, trust_remote_code: bool) -> bool:
        """Fallback: load model on CPU."""
        try:
            torch.cuda.empty_cache()
            logger.info("Loading model on CPU...")

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=trust_remote_code
            )
            self._model.eval()

            self._model_name = model_name
            self._device = torch.device("cpu")
            self._loaded = True

            logger.info("Model loaded successfully on CPU")
            return True

        except Exception as e:
            logger.error(f"Failed to load model on CPU: {e}")
            return False

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        repetition_penalty: float = 1.1
    ) -> Dict[str, any]:
        """
        Generate response from prompt and measure execution time.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold
            do_sample: Use sampling instead of greedy decoding
            repetition_penalty: Penalty for repeating tokens

        Returns:
            Dictionary with:
                - response_text: Generated response (cleaned)
                - execution_time: Time taken in seconds
                - output_token_length: Number of tokens generated
                - error: Error message if failed, None otherwise
                - fallback_used: Whether CPU fallback was used
        """
        if not self._loaded:
            return {
                "response_text": "",
                "execution_time": 0.0,
                "output_token_length": 0,
                "error": "Model not loaded",
                "fallback_used": False
            }

        try:
            # Start timing
            start_time = time.time()

            # Tokenize input
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self._model.config.max_position_embeddings - max_new_tokens
            )

            # Move to device
            input_ids = inputs["input_ids"].to(self._device)
            attention_mask = inputs["attention_mask"].to(self._device)

            input_length = input_ids.shape[1]

            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id
                )

            # Decode output (exclude input prompt)
            generated_tokens = outputs[0][input_length:]
            response_text = self._tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )

            # Clean response
            response_text = self._clean_response(response_text)

            # Calculate time
            execution_time = time.time() - start_time
            output_token_length = len(generated_tokens)

            logger.debug(f"Generated {output_token_length} tokens in {execution_time:.3f}s")

            return {
                "response_text": response_text,
                "execution_time": execution_time,
                "output_token_length": output_token_length,
                "error": None,
                "fallback_used": False
            }

        except torch.cuda.OutOfMemoryError as e:
            logger.warning(f"CUDA OOM during generation: {e}")
            logger.info("Attempting CPU fallback for this generation...")
            return self._generate_on_cpu(
                prompt, max_new_tokens, temperature, top_p, do_sample, repetition_penalty
            )

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                "response_text": "",
                "execution_time": 0.0,
                "output_token_length": 0,
                "error": str(e),
                "fallback_used": False
            }

    def _generate_on_cpu(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        repetition_penalty: float
    ) -> Dict[str, any]:
        """Fallback: generate on CPU if GPU OOM."""
        try:
            # Move model to CPU temporarily
            original_device = self._device
            self._model.to("cpu")
            self._device = torch.device("cpu")
            torch.cuda.empty_cache()

            # Try generation on CPU
            start_time = time.time()

            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self._model.config.max_position_embeddings - max_new_tokens
            )

            input_ids = inputs["input_ids"].to("cpu")
            attention_mask = inputs["attention_mask"].to("cpu")
            input_length = input_ids.shape[1]

            with torch.no_grad():
                outputs = self._model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id
                )

            generated_tokens = outputs[0][input_length:]
            response_text = self._tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )
            response_text = self._clean_response(response_text)

            execution_time = time.time() - start_time
            output_token_length = len(generated_tokens)

            logger.info(f"CPU fallback successful: {output_token_length} tokens in {execution_time:.3f}s")

            # Try to move back to original device (may fail if still OOM)
            try:
                if str(original_device) != "cpu":
                    self._model.to(original_device)
                    self._device = original_device
                    torch.cuda.empty_cache()
            except:
                logger.warning("Could not move model back to GPU, staying on CPU")

            return {
                "response_text": response_text,
                "execution_time": execution_time,
                "output_token_length": output_token_length,
                "error": None,
                "fallback_used": True
            }

        except Exception as e:
            logger.error(f"CPU fallback also failed: {e}")
            return {
                "response_text": "",
                "execution_time": 0.0,
                "output_token_length": 0,
                "error": f"CPU fallback failed: {str(e)}",
                "fallback_used": True
            }

    def _clean_response(self, text: str) -> str:
        """
        Clean generated response text.

        Removes:
        - Leading/trailing whitespace
        - Common role markers (Assistant:, User:, etc.)
        - Extra newlines
        """
        text = text.strip()

        # Remove common role markers at start
        role_markers = ["Assistant:", "助理:", "AI:", "Bot:"]
        for marker in role_markers:
            if text.startswith(marker):
                text = text[len(marker):].strip()

        # Remove multiple consecutive newlines
        while "\n\n\n" in text:
            text = text.replace("\n\n\n", "\n\n")

        return text

    def get_model_info(self) -> Dict[str, any]:
        """Get information about loaded model."""
        if not self._loaded:
            return {"loaded": False}

        return {
            "loaded": True,
            "model_name": self._model_name,
            "device": str(self._device),
            "max_position_embeddings": self._model.config.max_position_embeddings,
            "vocab_size": self._model.config.vocab_size
        }

    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self._loaded
