"""
Test for prompt builder module.
"""

from typing import Optional
from llm.prompt_builder import PromptBuilder


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
