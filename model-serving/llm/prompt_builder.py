"""
Prompt Builder for Empathetic Dialogue Generation

Constructs prompts for LLM generation based on:
- Emotion labels
- Conversation context from EmpatheticDialogues
- System instructions for empathetic responses

The prompt format follows chat model conventions:
System: <role and instructions>
User: <emotional expression>
Assistant: <model generates here>
"""

from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class PromptBuilder:
    """
    Builds prompts for empathetic dialogue generation.

    Supports different prompt templates and formatting styles.
    """

    # System prompt that sets the empathetic assistant role
    DEFAULT_SYSTEM_PROMPT = """You are an empathetic and compassionate conversation assistant. Your role is to:
- Listen carefully to the user's feelings and experiences
- Provide emotional support and validation
- Respond with understanding and empathy
- Offer comfort and encouragement when appropriate
No matter what emotion the user expresses, respond with kindness and genuine concern."""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        include_system_prompt: bool = True,
        include_emotion_hint: bool = False,
    ):
        """
        Initialize prompt builder.

        Args:
            system_prompt: Custom system prompt (uses default if None)
            include_system_prompt: Whether to include system prompt in output
            include_emotion_hint: Whether to explicitly mention user's emotion in prompt
        """
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.include_system_prompt = include_system_prompt
        self.include_emotion_hint = include_emotion_hint

    def build_prompt(
        self,
        user_context: str,
        emotion: Optional[str] = None,
        conversation_history: Optional[list] = None,
        arousal: Optional[float] = None  # Kept for API compatibility, but not used
    ) -> str:
        """
        Build a complete prompt for LLM generation.

        Args:
            user_context: User's emotional expression/context
            emotion: Emotion label (e.g., "sad", "excited")
            conversation_history: Optional list of previous turns
                                 Format: [{"role": "user"/"assistant", "content": "..."}]
            arousal: Kept for API compatibility (not used)

        Returns:
            str: Formatted prompt ready for model input
        """
        prompt_parts = []

        # 1. System instruction (optional)
        if self.include_system_prompt:
            prompt_parts.append(f"System: {self.system_prompt}")

        # 2. Optional: Add emotion hint
        if self.include_emotion_hint and emotion:
            emotion_hint = self._get_emotion_hint(emotion)
            if emotion_hint:
                prompt_parts.append(f"\n{emotion_hint}")

        # 3. Conversation history (if provided)
        if conversation_history:
            prompt_parts.append("\n")
            for turn in conversation_history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")

        # 4. Current user input
        prompt_parts.append(f"\nUser: {user_context}")

        # 5. Assistant prefix (model will continue from here)
        prompt_parts.append("\nAssistant:")

        return "\n".join(prompt_parts).lstrip("\n")

    def _get_emotion_hint(self, emotion: str) -> str:
        """
        Get a contextual hint about the user's emotion.

        This can help the model better understand and respond
        to the user's emotional state.

        Args:
            emotion: Emotion label

        Returns:
            str: Emotion hint to include in prompt
        """
        # Map emotions to descriptive hints
        emotion_hints = {
            # High arousal positive
            "excited": "(Note: The user is feeling excited and energetic)",
            "joyful": "(Note: The user is experiencing joy)",
            "proud": "(Note: The user is feeling proud of something)",
            "grateful": "(Note: The user is expressing gratitude)",
            "impressed": "(Note: The user is impressed by something)",
            "confident": "(Note: The user is feeling confident)",

            # Low arousal positive
            "content": "(Note: The user is feeling content and peaceful)",
            "hopeful": "(Note: The user is feeling hopeful)",
            "caring": "(Note: The user is expressing care for others)",
            "trusting": "(Note: The user is feeling trusting)",
            "faithful": "(Note: The user is expressing faith or loyalty)",

            # High arousal negative
            "angry": "(Note: The user is feeling angry or frustrated)",
            "anxious": "(Note: The user is feeling anxious or worried)",
            "afraid": "(Note: The user is experiencing fear)",
            "annoyed": "(Note: The user is feeling annoyed)",
            "furious": "(Note: The user is very angry)",
            "terrified": "(Note: The user is extremely scared)",

            # Low arousal negative
            "sad": "(Note: The user is feeling sad)",
            "lonely": "(Note: The user is feeling lonely)",
            "disappointed": "(Note: The user is disappointed)",
            "devastated": "(Note: The user is deeply upset)",
            "guilty": "(Note: The user is feeling guilty)",
            "ashamed": "(Note: The user is feeling shame)",

            # Neutral/mixed
            "surprised": "(Note: The user is surprised)",
            "nostalgic": "(Note: The user is feeling nostalgic)",
            "sentimental": "(Note: The user is in a sentimental mood)",
            "anticipating": "(Note: The user is anticipating something)",
            "prepared": "(Note: The user is feeling prepared)",
            "jealous": "(Note: The user is feeling jealous)",
            "embarrassed": "(Note: The user is feeling embarrassed)",
        }

        return emotion_hints.get(emotion.lower(), f"(Note: The user is feeling {emotion})")

    def set_system_prompt(self, prompt: str):
        """Update the system prompt."""
        self.system_prompt = prompt

    def set_emotion_hint_enabled(self, enabled: bool):
        """Enable or disable emotion hints in prompts."""
        self.include_emotion_hint = enabled
