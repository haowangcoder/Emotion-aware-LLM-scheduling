"""
Unified Length Estimation Interface for Service Time Prediction.

This module provides a unified interface for predicting LLM output length
and service time using BERT bucket classification with expected value method.

The predictor:
1. Classifies input into token count bins
2. Computes expected token count: T_mean = sum(q_i * m_i)
3. Converts to service time: S = c_0 + T_mean * c_1
"""

from typing import List, Optional, Dict
import os
import warnings


class LengthEstimator:
    """
    Unified interface for length and service time prediction.

    Uses BERT bucket predictor with expected value method for accurate
    service time estimation.

    Args:
        model_path: Path to HuggingFace model directory
        bin_edges_path: Path to bin_edges.npy file
        model_name: HuggingFace model name (default: 'distilbert-base-uncased')
        device: Device for inference ('cuda' or 'cpu')
        per_token_latency: Latency per generated token (c_1)
        const_latency: Constant latency overhead (c_0)
        default_service_time: Fallback service time when predictor unavailable

    Example:
        >>> estimator = LengthEstimator(
        ...     model_path='models/bert_bucket',
        ...     bin_edges_path='models/bin_edges.npy'
        ... )
        >>> service_time = estimator.predict("What is machine learning?")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        bin_edges_path: Optional[str] = None,
        model_name: str = 'distilbert-base-uncased',
        device: str = 'cuda',
        per_token_latency: float = 0.02,
        const_latency: float = 0.1,
        default_service_time: float = 2.0
    ):
        self.default_service_time = default_service_time
        self.per_token_latency = per_token_latency
        self.const_latency = const_latency
        self.predictor = None
        self._model_path = model_path
        self._bin_edges_path = bin_edges_path

        # Resolve paths
        resolved_model_path = self._resolve_path(model_path)
        resolved_bin_edges_path = self._resolve_path(bin_edges_path)

        # Initialize BERT predictor if valid paths found
        if resolved_model_path and resolved_bin_edges_path:
            try:
                from predictor.bert_predictor import BertPredictor
                self.predictor = BertPredictor(
                    model_path=resolved_model_path,
                    bin_edges_path=resolved_bin_edges_path,
                    model_name=model_name,
                    device=device,
                    per_token_latency=per_token_latency,
                    const_latency=const_latency
                )
            except Exception as e:
                warnings.warn(
                    f"Failed to initialize BERT predictor: {e}. "
                    f"Using default service time: {default_service_time}"
                )
                self.predictor = None
        elif model_path and not resolved_bin_edges_path:
            warnings.warn(
                f"bin_edges_path not found or not provided. "
                f"Using default service time: {default_service_time}"
            )

        # Store resolved paths for introspection
        self._model_path = resolved_model_path
        self._bin_edges_path = resolved_bin_edges_path

    def _resolve_path(self, path: Optional[str]) -> Optional[str]:
        """
        Resolve a path, checking multiple locations.

        Args:
            path: Path to resolve (can be relative or absolute)

        Returns:
            Resolved absolute path, or None if not found
        """
        if not path:
            return None

        # Try as-is first
        if os.path.isabs(path):
            return path if os.path.exists(path) else None

        # Try relative to current working directory
        if os.path.exists(path):
            return os.path.abspath(path)

        # Try relative to model-serving package root
        package_root = os.path.dirname(os.path.dirname(__file__))
        candidate = os.path.join(package_root, path)
        if os.path.exists(candidate):
            return candidate

        return None

    def predict(self, prompt: str) -> float:
        """
        Predict service time for a single prompt.

        Uses BERT predictor if available, otherwise returns default.

        Args:
            prompt: Input prompt text

        Returns:
            Predicted service time in seconds
        """
        if self.predictor is not None:
            return self.predictor.predict_service_time(prompt)
        return self.default_service_time

    def predict_tokens(self, prompt: str) -> float:
        """
        Predict output token count for a single prompt.

        Args:
            prompt: Input prompt text

        Returns:
            Predicted token count (or default estimate)
        """
        if self.predictor is not None:
            return self.predictor.predict_tokens(prompt)
        # Estimate tokens from default service time
        return (self.default_service_time - self.const_latency) / self.per_token_latency

    def predict_batch(self, prompts: List[str]) -> List[float]:
        """
        Batch prediction of service times for multiple prompts.

        Args:
            prompts: List of input prompt texts

        Returns:
            List of predicted service times in seconds
        """
        if not prompts:
            return []

        if self.predictor is not None:
            return self.predictor.predict_batch(prompts)
        return [self.default_service_time] * len(prompts)

    def predict_tokens_batch(self, prompts: List[str]) -> List[float]:
        """
        Batch prediction of token counts for multiple prompts.

        Args:
            prompts: List of input prompt texts

        Returns:
            List of predicted token counts
        """
        if not prompts:
            return []

        if self.predictor is not None:
            return self.predictor.predict_tokens_batch(prompts)
        # Estimate tokens from default service time
        default_tokens = (self.default_service_time - self.const_latency) / self.per_token_latency
        return [default_tokens] * len(prompts)

    def is_available(self) -> bool:
        """
        Check if BERT predictor is available.

        Returns:
            True if BERT predictor is loaded and ready
        """
        return self.predictor is not None

    def get_info(self) -> Dict[str, any]:
        """
        Get information about the estimator configuration.

        Returns:
            Dictionary with configuration details
        """
        info = {
            'predictor_available': self.is_available(),
            'model_path': self._model_path,
            'bin_edges_path': self._bin_edges_path,
            'default_service_time': self.default_service_time,
            'per_token_latency': self.per_token_latency,
            'const_latency': self.const_latency,
        }
        # Add predictor-specific info if available
        if self.predictor is not None:
            info.update(self.predictor.get_info())
        return info


def create_length_estimator(config: dict) -> LengthEstimator:
    """
    Factory function to create LengthEstimator from configuration.

    Args:
        config: Dictionary with length_predictor configuration:
            - enabled: bool, whether to use BERT predictor
            - model_path: str, path to HuggingFace model directory
            - bin_edges_path: str, path to bin_edges.npy
            - model_name: str, HuggingFace model name
            - device: str, 'cuda' or 'cpu'
            - per_token_latency: float (c_1)
            - const_latency: float (c_0)
            - default_service_time: float

    Returns:
        Configured LengthEstimator instance
    """
    if not config.get('enabled', False):
        return LengthEstimator(
            model_path=None,
            bin_edges_path=None,
            default_service_time=config.get('default_service_time', 2.0),
            per_token_latency=config.get('per_token_latency', 0.02),
            const_latency=config.get('const_latency', 0.1)
        )

    return LengthEstimator(
        model_path=config.get('model_path'),
        bin_edges_path=config.get('bin_edges_path'),
        model_name=config.get('model_name', 'distilbert-base-uncased'),
        device=config.get('device', 'cuda'),
        per_token_latency=config.get('per_token_latency', 0.02),
        const_latency=config.get('const_latency', 0.1),
        default_service_time=config.get('default_service_time', 2.0)
    )
