"""
Response Cache for LLM Inference

Provides caching mechanism to:
- Avoid redundant LLM generation for same prompts
- Enable reproducible experiments across different schedulers
- Speed up iterative experimentation

The cache uses SHA256 hashing of prompts as keys and stores:
- Full prompt text (for reproducibility and analysis)
- Generated response text
- Execution time
- Output token length
- Model name
- Metadata (timestamp, generation parameters)
"""

import json
import hashlib
import os
from typing import Dict, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ResponseCache:
    """
    Cache for storing and retrieving LLM responses.

    Uses prompt hash as key to enable fast lookups.
    """

    def __init__(self, cache_file: Optional[str] = None):
        """
        Initialize response cache.

        Args:
            cache_file: Path to cache file (JSON format)
        """
        # Initialize cache and stats FIRST (always required)
        self.cache = {}  # In-memory cache: prompt_hash -> response_data
        self.stats = {
            "hits": 0,
            "misses": 0,
            "saves": 0
        }
        
        # Handle cache file path
        if cache_file:
            if os.path.isabs(cache_file) or "/" in cache_file:
                self.cache_file = cache_file   # already full path
            else:
                self.cache_file = cache_file
        else:
            self.cache_file = None

        # Load existing cache if file exists
        if self.cache_file and os.path.exists(self.cache_file):
            self.load_from_disk(self.cache_file)

    def _hash_prompt(self, prompt: str, model_name: Optional[str] = None) -> str:
        """
        Generate hash key for prompt.

        Includes model name in hash to support multi-model caching.

        Args:
            prompt: The prompt text
            model_name: Optional model name to include in hash

        Returns:
            str: SHA256 hash (first 16 characters for readability)
        """
        # Combine prompt and model name
        key_str = prompt
        if model_name:
            key_str = f"{model_name}::{prompt}"

        # Generate SHA256 hash
        hash_obj = hashlib.sha256(key_str.encode('utf-8'))
        return hash_obj.hexdigest()[:16]  # Use first 16 chars for readability

    def get(
        self,
        prompt: str,
        model_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached response for prompt.

        Args:
            prompt: The prompt text
            model_name: Model name used for generation

        Returns:
            Dictionary with cached data if found, None otherwise:
                - prompt: Full prompt text
                - response_text: Generated response
                - execution_time: Time taken (seconds)
                - output_token_length: Number of tokens generated
                - model_name: Model used
                - timestamp: When cached
                - generation_params: Parameters used for generation
        """
        prompt_hash = self._hash_prompt(prompt, model_name)

        if prompt_hash in self.cache:
            self.stats["hits"] += 1
            logger.debug(f"Cache HIT for prompt hash: {prompt_hash}")
            return self.cache[prompt_hash]
        else:
            self.stats["misses"] += 1
            logger.debug(f"Cache MISS for prompt hash: {prompt_hash}")
            return None

    def set(
        self,
        prompt: str,
        response_text: str,
        execution_time: float,
        output_token_length: int,
        model_name: str,
        generation_params: Optional[Dict] = None,
        error: Optional[str] = None,
        fallback_used: bool = False
    ) -> str:
        """
        Store response in cache.

        Args:
            prompt: The prompt text
            response_text: Generated response
            execution_time: Time taken (seconds)
            output_token_length: Number of tokens generated
            model_name: Model used
            generation_params: Generation parameters (temperature, max_tokens, etc.)
            error: Error message if generation failed
            fallback_used: Whether CPU fallback was used

        Returns:
            str: Prompt hash (cache key)
        """
        prompt_hash = self._hash_prompt(prompt, model_name)

        cache_entry = {
            "response_text": response_text,
            "execution_time": execution_time,
            "output_token_length": output_token_length,
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "generation_params": generation_params or {},
            "error": error,
            "fallback_used": fallback_used,
            "prompt": prompt  # Store full prompt for reproducibility and analysis
        }

        self.cache[prompt_hash] = cache_entry
        self.stats["saves"] += 1

        logger.debug(f"Cached response for prompt hash: {prompt_hash}")

        return prompt_hash

    def has(self, prompt: str, model_name: Optional[str] = None) -> bool:
        """Check if prompt is cached."""
        prompt_hash = self._hash_prompt(prompt, model_name)
        return prompt_hash in self.cache

    def save_to_disk(self, filepath: Optional[str] = None) -> bool:
        """
        Persist cache to disk as JSON.

        Args:
            filepath: Path to save cache (uses self.cache_file if None)

        Returns:
            bool: True if successful, False otherwise
        """
        filepath = filepath or self.cache_file

        if filepath is None:
            logger.warning("No cache file specified, cannot save")
            return False

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # Prepare data to save
            cache_data = {
                "cache": self.cache,
                "stats": self.stats,
                "metadata": {
                    "saved_at": datetime.now().isoformat(),
                    "num_entries": len(self.cache)
                }
            }

            # Write to file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Cache saved to {filepath} ({len(self.cache)} entries)")
            return True

        except Exception as e:
            logger.error(f"Failed to save cache to {filepath}: {e}")
            return False

    def load_from_disk(self, filepath: Optional[str] = None) -> bool:
        """
        Load cache from disk.

        Args:
            filepath: Path to load cache from (uses self.cache_file if None)

        Returns:
            bool: True if successful, False otherwise
        """
        filepath = filepath or self.cache_file

        if filepath is None or not os.path.exists(filepath):
            logger.warning(f"Cache file not found: {filepath}")
            return False

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            self.cache = cache_data.get("cache", {})
            loaded_stats = cache_data.get("stats", {})
            metadata = cache_data.get("metadata", {})

            # Merge stats (preserve current hits/misses, add to saves)
            self.stats["saves"] += loaded_stats.get("saves", 0)

            logger.info(f"Cache loaded from {filepath}")
            logger.info(f"Loaded {len(self.cache)} entries")
            if metadata:
                logger.info(f"Cache metadata: {metadata}")

            return True

        except Exception as e:
            logger.error(f"Failed to load cache from {filepath}: {e}")
            return False

    def clear(self):
        """Clear all cached entries."""
        self.cache = {}
        self.stats = {"hits": 0, "misses": 0, "saves": 0}
        logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0

        return {
            "num_entries": len(self.cache),
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "saves": self.stats["saves"],
            "total_requests": total_requests,
            "hit_rate": hit_rate
        }

    def get_size_bytes(self) -> int:
        """Estimate cache size in bytes (approximate)."""
        # Convert to JSON and measure size
        cache_str = json.dumps(self.cache)
        return len(cache_str.encode('utf-8'))

    def remove(self, prompt: str, model_name: Optional[str] = None) -> bool:
        """
        Remove entry from cache.

        Args:
            prompt: The prompt text
            model_name: Model name

        Returns:
            bool: True if entry was removed, False if not found
        """
        prompt_hash = self._hash_prompt(prompt, model_name)

        if prompt_hash in self.cache:
            del self.cache[prompt_hash]
            logger.debug(f"Removed cache entry: {prompt_hash}")
            return True
        else:
            logger.debug(f"Cache entry not found: {prompt_hash}")
            return False
