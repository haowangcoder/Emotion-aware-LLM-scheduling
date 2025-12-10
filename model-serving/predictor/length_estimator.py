"""
Unified Length Estimation Interface for Service Time Prediction.

This module provides a unified interface for predicting LLM output length
and service time, supporting multiple prediction modes:
    1. BERT model prediction (recommended)
    2. Pre-computed values from CSV/cache
    3. Default fallback values

The length estimator abstracts away the underlying prediction method,
allowing the scheduler to work with any prediction source.
"""

from typing import List, Optional, Dict
import os


class LengthEstimator:
    """
    Unified interface for length and service time prediction.

    Supports multiple prediction modes:
        - BERT model: Uses trained BERT regression model
        - Default: Returns constant default service time

    Args:
        model_path: Path to trained BERT model weights (optional)
        model_name: HuggingFace model name (default: 'bert-base-uncased')
        device: Device for inference ('cuda' or 'cpu')
        per_token_latency: Latency per generated token in seconds
        const_latency: Constant latency overhead in seconds
        default_service_time: Fallback service time when predictor unavailable

    Example:
        >>> estimator = LengthEstimator(model_path='models/bert_regression.pth')
        >>> service_time = estimator.predict("What is machine learning?")
        >>> print(f"Predicted service time: {service_time:.2f}s")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = 'bert-base-uncased',
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

        # Resolve relative model path in a robust way:
        # 1) As given (relative to current working directory)
        # 2) Relative to the package root (model-serving/)
        resolved_path = None
        if model_path:
            if os.path.isabs(model_path):
                resolved_path = model_path if os.path.exists(model_path) else None
            else:
                # Try relative to current working directory first
                if os.path.exists(model_path):
                    resolved_path = model_path
                else:
                    # Then try relative to the model-serving package root
                    package_root = os.path.dirname(os.path.dirname(__file__))
                    candidate = os.path.join(package_root, model_path)
                    if os.path.exists(candidate):
                        resolved_path = candidate

        # Initialize BERT predictor if a valid model path was found
        if resolved_path:
            try:
                from predictor.bert_predictor import BertPredictor
                self.predictor = BertPredictor(
                    model_path=resolved_path,
                    model_name=model_name,
                    device=device,
                    per_token_latency=per_token_latency,
                    const_latency=const_latency
                )
            except Exception as e:
                import warnings
                warnings.warn(
                    f"Failed to initialize BERT predictor: {e}. "
                    f"Using default service time: {default_service_time}"
                )
                self.predictor = None

        # Store the resolved path (if any) for introspection
        self._model_path = resolved_path

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
        return {
            'predictor_available': self.is_available(),
            'model_path': self._model_path,
            'default_service_time': self.default_service_time,
            'per_token_latency': self.per_token_latency,
            'const_latency': self.const_latency,
        }


def create_length_estimator(config: dict) -> LengthEstimator:
    """
    Factory function to create LengthEstimator from configuration.

    Args:
        config: Dictionary with length_predictor configuration:
            - enabled: bool, whether to use BERT predictor
            - model_path: str, path to model weights
            - model_name: str, HuggingFace model name
            - device: str, 'cuda' or 'cpu'
            - per_token_latency: float
            - const_latency: float
            - default_service_time: float

    Returns:
        Configured LengthEstimator instance
    """
    if not config.get('enabled', False):
        return LengthEstimator(
            model_path=None,
            default_service_time=config.get('default_service_time', 2.0),
            per_token_latency=config.get('per_token_latency', 0.02),
            const_latency=config.get('const_latency', 0.1)
        )

    return LengthEstimator(
        model_path=config.get('model_path'),
        model_name=config.get('model_name', 'bert-base-uncased'),
        device=config.get('device', 'cuda'),
        per_token_latency=config.get('per_token_latency', 0.02),
        const_latency=config.get('const_latency', 0.1),
        default_service_time=config.get('default_service_time', 2.0)
    )
