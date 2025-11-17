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

    # Alternative: shorter system prompt
    CONCISE_SYSTEM_PROMPT = """You are a caring and empathetic assistant. Listen carefully and respond with compassion."""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        include_emotion_hint: bool = False,
        enable_emotion_length_control: bool = False,
        base_response_length: int = 100,
        alpha: float = 1.0
    ):
        """
        Initialize prompt builder.

        Args:
            system_prompt: Custom system prompt (uses default if None)
            include_emotion_hint: Whether to explicitly mention user's emotion in prompt
            enable_emotion_length_control: Enable emotion-aware response length control
            base_response_length: Base response length L_0 in tokens (default: 100)
            alpha: Scaling factor α for arousal impact on response length (default: 1.0)
        """
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.include_emotion_hint = include_emotion_hint
        self.enable_emotion_length_control = enable_emotion_length_control
        self.base_response_length = base_response_length
        self.alpha = alpha

    def build_prompt(
        self,
        user_context: str,
        emotion: Optional[str] = None,
        conversation_history: Optional[list] = None,
        arousal: Optional[float] = None
    ) -> str:
        """
        Build a complete prompt for LLM generation.

        Args:
            user_context: User's emotional expression/context
            emotion: Emotion label (e.g., "sad", "excited")
            conversation_history: Optional list of previous turns
                                 Format: [{"role": "user"/"assistant", "content": "..."}]
            arousal: Arousal value for emotion-aware length control (range: [-1, 1])

        Returns:
            str: Formatted prompt ready for model input
        """
        prompt_parts = []

        # 1. System instruction
        system_prompt = self.system_prompt

        # 1.5. Add length instruction based on arousal (if enabled)
        if self.enable_emotion_length_control and arousal is not None:
            target_length = self._calculate_target_length(arousal)
            length_instruction = self._get_length_instruction(target_length)
            system_prompt = f"{system_prompt}\n{length_instruction}"

        prompt_parts.append(f"System: {system_prompt}")

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

        return "\n".join(prompt_parts)

    def build_simple_prompt(self, user_context: str, emotion: Optional[str] = None) -> str:
        """
        Build a simpler prompt without extensive system instructions.

        Good for models that don't need heavy prompting.

        Args:
            user_context: User's emotional expression
            emotion: Emotion label (optional)

        Returns:
            str: Simple formatted prompt
        """
        if emotion and self.include_emotion_hint:
            return f"User (feeling {emotion}): {user_context}\nAssistant:"
        else:
            return f"User: {user_context}\nAssistant:"

    def _calculate_target_length(self, arousal: float) -> int:
        """
        Calculate target response length based on arousal value.

        Uses formula: L_i = L_0 * (1 + α * a_i)

        Args:
            arousal: Arousal value in range [-1, 1]

        Returns:
            int: Target response length in tokens
        """
        target_length = self.base_response_length * (1 + self.alpha * arousal)
        # Ensure positive length, minimum 10 tokens
        target_length = max(10, int(target_length))
        return target_length

    def _get_length_instruction(self, target_length: int) -> str:
        """
        Generate a natural language length instruction for the prompt.

        Args:
            target_length: Target response length in tokens

        Returns:
            str: Length instruction to add to system prompt
        """
        # Provide guidance based on token count
        # Roughly: 1 token ≈ 0.75 words, so we convert to approximate word count
        approx_words = int(target_length * 0.75)

        if target_length < 40:
            style = "very brief"
        elif target_length < 80:
            style = "concise"
        elif target_length < 150:
            style = "moderate"
        else:
            style = "detailed"

        instruction = f"Please provide a {style} response of approximately {approx_words} words (around {target_length} tokens)."
        return instruction

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


def build_empathetic_prompt(
    user_context: str,
    emotion: Optional[str] = None,
    include_system: bool = True,
    include_emotion_hint: bool = False
) -> str:
    """
    Convenience function to build a prompt.

    Args:
        user_context: User's emotional expression
        emotion: Emotion label (optional)
        include_system: Whether to include system instruction
        include_emotion_hint: Whether to add emotion hint

    Returns:
        str: Formatted prompt
    """
    builder = PromptBuilder(include_emotion_hint=include_emotion_hint)

    if not include_system:
        return builder.build_simple_prompt(user_context, emotion)

    return builder.build_prompt(user_context, emotion)


def test_prompt_builder():
    """Test prompt builder with various scenarios."""
    print(f"\n{'='*60}")
    print("Testing Prompt Builder")
    print(f"{'='*60}\n")

    # Test cases
    test_cases = [
        {
            "emotion": "excited",
            "context": "I just got accepted into my dream university! I can't believe it happened!",
            "description": "High arousal positive emotion"
        },
        {
            "emotion": "sad",
            "context": "My best friend is moving away next month. We've been together since childhood.",
            "description": "Low arousal negative emotion"
        },
        {
            "emotion": "anxious",
            "context": "I have a big presentation tomorrow and I'm not sure if I'm prepared enough.",
            "description": "High arousal negative emotion"
        },
        {
            "emotion": "grateful",
            "context": "My family surprised me with a birthday party. I feel so loved.",
            "description": "Low arousal positive emotion"
        }
    ]

    # Test 1: Default prompt (with system, without emotion hint)
    print("Test 1: Default prompt format")
    print("=" * 60)
    builder = PromptBuilder()

    for test in test_cases:
        prompt = builder.build_prompt(
            user_context=test["context"],
            emotion=test["emotion"]
        )
        print(f"\nEmotion: {test['emotion']} ({test['description']})")
        print("-" * 60)
        print(prompt)
        print()

    # Test 2: With emotion hints
    print(f"\n{'='*60}")
    print("Test 2: Prompts with emotion hints")
    print("=" * 60)

    builder_with_hints = PromptBuilder(include_emotion_hint=True)

    for test in test_cases[:2]:  # Just show 2 examples
        prompt = builder_with_hints.build_prompt(
            user_context=test["context"],
            emotion=test["emotion"]
        )
        print(f"\nEmotion: {test['emotion']}")
        print("-" * 60)
        print(prompt)
        print()

    # Test 3: Simple prompt format
    print(f"\n{'='*60}")
    print("Test 3: Simple prompt format")
    print("=" * 60)

    for test in test_cases[:2]:
        prompt = builder.build_simple_prompt(
            user_context=test["context"],
            emotion=test["emotion"]
        )
        print(f"\nEmotion: {test['emotion']}")
        print("-" * 60)
        print(prompt)
        print()

    # Test 4: With conversation history
    print(f"\n{'='*60}")
    print("Test 4: Multi-turn conversation")
    print("=" * 60)

    history = [
        {"role": "user", "content": "I'm worried about my exam tomorrow."},
        {"role": "assistant", "content": "I understand your concern. What subject is the exam on?"},
    ]

    prompt = builder.build_prompt(
        user_context="It's calculus. I've been studying but I still feel unprepared.",
        emotion="anxious",
        conversation_history=history
    )

    print(prompt)
    print()

    # Test 5: Custom system prompt
    print(f"\n{'='*60}")
    print("Test 5: Custom system prompt")
    print("=" * 60)

    custom_system = "You are a professional counselor specialized in emotional support."
    builder_custom = PromptBuilder(system_prompt=custom_system)

    prompt = builder_custom.build_prompt(
        user_context=test_cases[0]["context"],
        emotion=test_cases[0]["emotion"]
    )

    print(prompt)
    print()


if __name__ == "__main__":
    test_prompt_builder()
