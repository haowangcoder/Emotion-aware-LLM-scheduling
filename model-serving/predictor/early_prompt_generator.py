"""
Early Prompt Generator for BERT-based Service Time Prediction.

This module generates prompts at job arrival time (before scheduling) to enable
BERT-based service time prediction for informed scheduling decisions.

The key insight is that BERT prediction requires the actual prompt text,
but prompts are normally only generated during LLM execution (after scheduling).
By generating prompts early, we can use BERT predictions for scheduling.
"""

from typing import Optional, Tuple, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from llm.dataset_loader import EmpatheticDialoguesLoader
    from llm.prompt_builder import PromptBuilder
    from predictor.length_estimator import LengthEstimator
    from core.job import Job

logger = logging.getLogger(__name__)


class EarlyPromptGenerator:
    """
    Generates prompts early (at job arrival) for service time prediction.

    This enables BERT-based prediction before scheduling decisions are made,
    allowing schedulers to use accurate service time estimates.

    Attributes:
        dataset_loader: EmpatheticDialoguesLoader for getting user contexts
        prompt_builder: PromptBuilder for constructing prompts
        length_estimator: LengthEstimator (with BERT predictor) for prediction
        default_service_time: Fallback when prediction unavailable
    """

    def __init__(
        self,
        dataset_loader: "EmpatheticDialoguesLoader",
        prompt_builder: "PromptBuilder",
        length_estimator: Optional["LengthEstimator"] = None,
        default_service_time: float = 2.0,
    ):
        """
        Initialize EarlyPromptGenerator.

        Args:
            dataset_loader: Loader for EmpatheticDialogues dataset
            prompt_builder: Builder for constructing prompts
            length_estimator: Estimator with BERT predictor (optional)
            default_service_time: Default service time when predictor unavailable
        """
        self.dataset_loader = dataset_loader
        self.prompt_builder = prompt_builder
        self.length_estimator = length_estimator
        self.default_service_time = default_service_time

        # Track statistics
        self.stats = {
            "total_predictions": 0,
            "bert_predictions": 0,
            "fallback_predictions": 0,
        }

    def generate_prompt_and_predict(
        self,
        job: "Job",
        max_conversation_turns: int = 2,
    ) -> Tuple[str, float, int]:
        """
        Generate prompt for a job and predict service time.

        This method:
        1. Gets user context from dataset based on job's emotion
        2. Builds the complete prompt
        3. Calls BERT predictor (if available) to estimate service time

        Args:
            job: Job object with emotion_label, arousal, etc.
            max_conversation_turns: Max conversation history turns

        Returns:
            Tuple of (prompt, predicted_service_time, conversation_index)
        """
        # Get emotion info from job
        emotion = job.get_emotion_label() if hasattr(job, 'get_emotion_label') else getattr(job, 'emotion_label', 'neutral')
        arousal = job.get_arousal() if hasattr(job, 'get_arousal') else getattr(job, 'arousal', 0.0)

        # Check if job already has a conversation_index (for reproducibility)
        conversation_index = getattr(job, 'conversation_index', None)

        # Get user context from dataset
        user_context, selected_index = self.dataset_loader.get_user_context_by_emotion(
            emotion=emotion.lower() if emotion else 'neutral',
            max_turns=max_conversation_turns,
            conversation_index=conversation_index
        )

        # Fallback if no context found
        if not user_context:
            logger.debug(f"Job {job.job_id}: No context found for emotion '{emotion}', using fallback")
            user_context = f"I'm feeling {emotion} right now."
            selected_index = -1

        # Build prompt
        prompt = self.prompt_builder.build_prompt(
            user_context=user_context,
            emotion=emotion,
            arousal=arousal
        )

        # Predict service time
        self.stats["total_predictions"] += 1

        if self.length_estimator is not None and self.length_estimator.is_available():
            predicted_service_time = self.length_estimator.predict(prompt)
            self.stats["bert_predictions"] += 1
            logger.debug(
                f"Job {job.job_id}: BERT predicted {predicted_service_time:.3f}s "
                f"for emotion '{emotion}'"
            )
        else:
            # Fallback to default
            predicted_service_time = self.default_service_time
            self.stats["fallback_predictions"] += 1
            logger.debug(
                f"Job {job.job_id}: Using default service time {predicted_service_time:.3f}s "
                f"(BERT not available)"
            )

        return prompt, predicted_service_time, selected_index

    def generate_prompt_only(
        self,
        job: "Job",
        max_conversation_turns: int = 2,
    ) -> Tuple[str, int]:
        """
        Generate prompt without prediction (for trace preprocessing).

        Args:
            job: Job object with emotion_label, arousal, etc.
            max_conversation_turns: Max conversation history turns

        Returns:
            Tuple of (prompt, conversation_index)
        """
        emotion = job.get_emotion_label() if hasattr(job, 'get_emotion_label') else getattr(job, 'emotion_label', 'neutral')
        arousal = job.get_arousal() if hasattr(job, 'get_arousal') else getattr(job, 'arousal', 0.0)
        conversation_index = getattr(job, 'conversation_index', None)

        user_context, selected_index = self.dataset_loader.get_user_context_by_emotion(
            emotion=emotion.lower() if emotion else 'neutral',
            max_turns=max_conversation_turns,
            conversation_index=conversation_index
        )

        if not user_context:
            user_context = f"I'm feeling {emotion} right now."
            selected_index = -1

        prompt = self.prompt_builder.build_prompt(
            user_context=user_context,
            emotion=emotion,
            arousal=arousal
        )

        return prompt, selected_index

    def predict_service_time(self, prompt: str) -> float:
        """
        Predict service time for a given prompt.

        Args:
            prompt: The complete prompt text

        Returns:
            Predicted service time in seconds
        """
        if self.length_estimator is not None and self.length_estimator.is_available():
            return self.length_estimator.predict(prompt)
        return self.default_service_time

    def is_prediction_available(self) -> bool:
        """Check if BERT prediction is available."""
        return self.length_estimator is not None and self.length_estimator.is_available()

    def get_stats(self) -> dict:
        """Get prediction statistics."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset prediction statistics."""
        self.stats = {
            "total_predictions": 0,
            "bert_predictions": 0,
            "fallback_predictions": 0,
        }


def create_early_prompt_generator(
    dataset_loader: "EmpatheticDialoguesLoader",
    prompt_builder: "PromptBuilder",
    length_estimator: Optional["LengthEstimator"] = None,
    default_service_time: float = 2.0,
) -> EarlyPromptGenerator:
    """
    Factory function to create EarlyPromptGenerator.

    Args:
        dataset_loader: EmpatheticDialoguesLoader instance
        prompt_builder: PromptBuilder instance
        length_estimator: Optional LengthEstimator with BERT predictor
        default_service_time: Default service time fallback

    Returns:
        Configured EarlyPromptGenerator instance
    """
    return EarlyPromptGenerator(
        dataset_loader=dataset_loader,
        prompt_builder=prompt_builder,
        length_estimator=length_estimator,
        default_service_time=default_service_time,
    )
