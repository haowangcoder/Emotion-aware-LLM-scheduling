"""
Test for dataset loader module.
"""

import logging
from llm.dataset_loader import EmpatheticDialoguesLoader


def test_dataset_loader():
    """Test dataset loader functionality."""
    print(f"\n{'='*60}")
    print("Testing EmpatheticDialogues Dataset Loader")
    print(f"{'='*60}\n")

    # Initialize and load
    loader = EmpatheticDialoguesLoader(dataset_dir="./dataset")

    print("Loading dataset...")
    success = loader.load(splits=["train", "valid"])

    if not success:
        print("Failed to load dataset!")
        return

    print(f"\nDataset loaded successfully!")

    # Show available emotions
    emotions = loader.get_available_emotions()
    print(f"\nFound {len(emotions)} emotions:")
    print(", ".join(emotions))

    # Show statistics
    print(f"\nEmotion statistics:")
    stats = loader.get_emotion_statistics()
    for emotion in sorted(stats.keys())[:10]:  # Show first 10
        print(f"  {emotion}: {stats[emotion]} conversations")

    # Test sampling
    print(f"\n{'='*60}")
    print("Testing conversation sampling:")
    print(f"{'='*60}\n")

    test_emotions = ["excited", "sad", "anxious", "grateful"]

    for emotion in test_emotions:
        print(f"\nEmotion: {emotion}")
        print("-" * 40)

        context, conv_idx = loader.get_user_context_by_emotion(emotion, max_turns=2)

        if context:
            print(f"User context (index {conv_idx}):\n{context}")
        else:
            print(f"No conversation found for emotion: {emotion}")

        # Get full conversation
        conv, conv_idx = loader.get_conversation_by_emotion(emotion, max_turns=3)
        if conv:
            print(f"\nConversation ID: {conv['conv_id']} (index {conv_idx})")
            print(f"User utterances: {len(conv['user_utterances'])}")
            print(f"Assistant utterances: {len(conv['assistant_utterances'])}")

        print()


if __name__ == "__main__":
    # Set logging level
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    test_dataset_loader()
