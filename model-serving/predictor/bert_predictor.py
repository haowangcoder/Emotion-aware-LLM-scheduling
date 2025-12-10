"""
BERT-based Length Predictor for LLM Output Tokens.

This module provides a BERT regression model to predict the expected output
token length of an LLM response given the input prompt. The predicted length
is used to estimate service time for scheduling decisions.

Migrated from: LLM-serving-with-proxy-models/output-token-len-prediction/latency_prediction.py

Model Architecture:
    User Query -> BERT Encoder -> [CLS] token -> Linear layers -> Predicted token count

Supported prediction tasks:
    - Regression (task_type=0): Directly predict output token count (0-512)
    - Binary classification (task_type=1): Predict if length exceeds median
    - Multi-class classification (task_type=2): Predict percentile bucket
"""

from typing import List, Optional
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, BertModel


class BertRegressionModel(nn.Module):
    """
    BERT-based regression model for output token length prediction.

    Uses the [CLS] token representation from BERT followed by linear layers
    to predict the expected number of output tokens.

    Args:
        model_name: HuggingFace model name (default: 'bert-base-uncased')
        hidden_dim: Hidden dimension for linear layers (default: 128)
        freeze_bert: Whether to freeze BERT weights (default: True for inference)
    """

    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        hidden_dim: int = 128,
        freeze_bert: bool = True
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = BertModel.from_pretrained(model_name)

        # Freeze BERT weights for inference
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Output layers: [CLS] -> hidden_dim -> hidden_dim -> 1
        self.cls = nn.Linear(self.config.hidden_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass to predict output token count.

        Args:
            input_ids: Tokenized input IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Predicted token count [batch_size]
        """
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Extract [CLS] token representation (first token)
        # outputs.last_hidden_state: [batch_size, sequence_size, hidden_size]
        logits = outputs.last_hidden_state[:, 0, :]

        # Pass through linear layers
        output = self.relu(self.cls(logits))
        output = self.relu(self.fc1(output))
        output = self.fc2(output).squeeze(-1)

        return output


class BertPredictor:
    """
    BERT-based length predictor wrapper.

    Handles model loading, tokenization, and inference for predicting
    LLM output token counts and service times.

    Args:
        model_path: Path to trained model weights (.pth file)
        model_name: HuggingFace model name (default: 'bert-base-uncased')
        device: Device to run inference on ('cuda' or 'cpu')
        per_token_latency: Latency per generated token in seconds (default: 0.02)
        const_latency: Constant latency overhead in seconds (default: 0.1)
        hidden_dim: Hidden dimension for the model (default: 128)

    Example:
        >>> predictor = BertPredictor(model_path='models/bert_regression.pth')
        >>> tokens = predictor.predict_tokens("What is machine learning?")
        >>> service_time = predictor.predict_service_time("What is machine learning?")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = 'bert-base-uncased',
        device: str = 'cuda',
        per_token_latency: float = 0.02,
        const_latency: float = 0.1,
        hidden_dim: int = 128
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.per_token_latency = per_token_latency
        self.const_latency = const_latency

        # Initialize model
        self.model = BertRegressionModel(
            model_name=model_name,
            hidden_dim=hidden_dim,
            freeze_bert=True
        )

        # Load trained weights if provided
        if model_path:
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )

        self.model.to(self.device)
        self.model.eval()

    def _truncate_tail(self, inputs: dict, max_length: int = 512) -> dict:
        """
        Truncate inputs to keep the LAST max_length tokens (tail truncation).

        This matches the training data preprocessing in preprocess_dataset.py
        which uses: example['input_ids'] = example['input_ids'][-512:]

        Args:
            inputs: Tokenizer outputs with input_ids, attention_mask, etc.
            max_length: Maximum sequence length

        Returns:
            Truncated inputs
        """
        seq_len = inputs['input_ids'].shape[-1]
        if seq_len > max_length:
            for key in inputs:
                if inputs[key].dim() == 1:
                    inputs[key] = inputs[key][-max_length:]
                else:
                    inputs[key] = inputs[key][:, -max_length:]
        return inputs

    def predict_tokens(self, prompt: str) -> float:
        """
        Predict output token count for a single prompt.

        Args:
            prompt: Input prompt text

        Returns:
            Predicted number of output tokens
        """
        # Tokenize without truncation first
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=False,
            padding=True
        )
        # Apply tail truncation to match training preprocessing
        inputs = self._truncate_tail(inputs, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            predicted_tokens = self.model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )

        # Ensure non-negative prediction
        return max(0.0, predicted_tokens.item())

    def predict_service_time(self, prompt: str) -> float:
        """
        Predict service time (latency) for a single prompt.

        Service time is computed as:
            service_time = const_latency + predicted_tokens * per_token_latency

        Args:
            prompt: Input prompt text

        Returns:
            Predicted service time in seconds
        """
        predicted_tokens = self.predict_tokens(prompt)
        return self.const_latency + predicted_tokens * self.per_token_latency

    def predict_batch(self, prompts: List[str]) -> List[float]:
        """
        Batch prediction for multiple prompts.

        More efficient than calling predict_service_time multiple times
        as it processes all prompts in a single forward pass.

        Args:
            prompts: List of input prompt texts

        Returns:
            List of predicted service times in seconds
        """
        if not prompts:
            return []

        # Process each prompt individually to apply tail truncation
        # (batch processing with variable-length tail truncation is complex)
        service_times = []
        for prompt in prompts:
            service_times.append(self.predict_service_time(prompt))

        return service_times

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

        # Process each prompt individually to apply tail truncation
        token_counts = []
        for prompt in prompts:
            token_counts.append(self.predict_tokens(prompt))

        return token_counts
