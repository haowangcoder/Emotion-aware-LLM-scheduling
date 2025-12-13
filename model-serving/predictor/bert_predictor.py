"""
BERT Bucket Predictor for LLM Output Token Length.

Uses classification into bins with expected value method to predict
the number of output tokens. The expected token count is computed as:

    T_mean = sum(q_i * m_i)

where q_i is the softmax probability for bin i, and m_i is the bin midpoint.

Service time is then computed as:
    S = const_latency + T_mean * per_token_latency
"""

from typing import List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BertPredictor:
    """
    BERT-based bucket predictor using expected value method.

    Instead of direct regression, this model:
    1. Classifies input into token count bins
    2. Computes expected token count from probability distribution
    3. Converts to service time using linear formula

    Args:
        model_path: Path to HuggingFace model directory
        bin_edges_path: Path to bin_edges.npy file
        device: Device for inference ('cuda' or 'cpu')
        per_token_latency: Latency per generated token (c_1)
        const_latency: Constant latency overhead (c_0)
        model_name: HuggingFace model name for tokenizer

    Example:
        >>> predictor = BertPredictor(
        ...     model_path='models/bert_bucket',
        ...     bin_edges_path='models/bin_edges.npy'
        ... )
        >>> tokens = predictor.predict_tokens("What is machine learning?")
        >>> service_time = predictor.predict_service_time("What is machine learning?")
    """

    def __init__(
        self,
        model_path: str,
        bin_edges_path: str,
        device: str = 'cuda',
        per_token_latency: float = 0.02,
        const_latency: float = 0.1,
        model_name: str = 'distilbert-base-uncased'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.per_token_latency = per_token_latency
        self.const_latency = const_latency

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load bin edges and compute midpoints
        self.bin_edges = np.load(bin_edges_path)
        self.num_bins = len(self.bin_edges) - 1

        # Compute bin midpoints: m_i = (e_i + e_{i+1}) / 2
        self.bin_midpoints = torch.tensor([
            (self.bin_edges[i] + self.bin_edges[i + 1]) / 2
            for i in range(self.num_bins)
        ], dtype=torch.float32, device=self.device)

        # Load classification model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=self.num_bins
        ).to(self.device)
        self.model.eval()

    def predict_distribution(self, prompt: str) -> torch.Tensor:
        """
        Predict probability distribution over bins.

        Args:
            prompt: Input prompt text

        Returns:
            Softmax probabilities [num_bins]
        """
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=-1)

        return probs.squeeze(0)  # [num_bins]

    def predict_tokens(self, prompt: str) -> float:
        """
        Predict expected token count using probability-weighted midpoints.

        T_mean = sum(q_i * m_i)

        Args:
            prompt: Input prompt text

        Returns:
            Expected number of output tokens
        """
        probs = self.predict_distribution(prompt)  # [num_bins]
        expected_tokens = torch.dot(probs, self.bin_midpoints).item()
        return max(0.0, expected_tokens)

    def predict_service_time(self, prompt: str) -> float:
        """
        Predict service time from expected token count.

        S = c_0 + T_mean * c_1

        Args:
            prompt: Input prompt text

        Returns:
            Predicted service time in seconds
        """
        expected_tokens = self.predict_tokens(prompt)
        return self.const_latency + expected_tokens * self.per_token_latency

    def predict_batch(self, prompts: List[str]) -> List[float]:
        """
        Batch prediction of service times.

        Args:
            prompts: List of input prompt texts

        Returns:
            List of predicted service times in seconds
        """
        if not prompts:
            return []
        return [self.predict_service_time(p) for p in prompts]

    def predict_tokens_batch(self, prompts: List[str]) -> List[float]:
        """
        Batch prediction of token counts.

        Args:
            prompts: List of input prompt texts

        Returns:
            List of predicted token counts
        """
        if not prompts:
            return []
        return [self.predict_tokens(p) for p in prompts]

    def predict_bin(self, prompt: str) -> int:
        """
        Predict most likely bin (for evaluation/debugging).

        Args:
            prompt: Input prompt text

        Returns:
            Predicted bin index (argmax)
        """
        probs = self.predict_distribution(prompt)
        return probs.argmax().item()

    def get_bin_range(self, bin_idx: int) -> tuple:
        """
        Get token range for a specific bin.

        Args:
            bin_idx: Bin index

        Returns:
            Tuple of (low, high) token counts
        """
        low = int(self.bin_edges[bin_idx])
        high = int(self.bin_edges[bin_idx + 1])
        return (low, high)

    def get_info(self) -> dict:
        """
        Get predictor configuration info.

        Returns:
            Dictionary with configuration details
        """
        return {
            'num_bins': self.num_bins,
            'bin_edges': self.bin_edges.tolist(),
            'bin_midpoints': self.bin_midpoints.cpu().tolist(),
            'per_token_latency': self.per_token_latency,
            'const_latency': self.const_latency,
            'device': str(self.device),
        }
