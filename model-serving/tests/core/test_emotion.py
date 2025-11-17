"""
Test for emotion module.
"""

from core.emotion import EmotionConfig, sample_emotions_batch, get_emotion_statistics, EMOTION_AROUSAL_MAP


def test_emotion():
    """Test emotion configuration and sampling."""
    print("=" * 60)
    print("Emotion Module Test")
    print("=" * 60)

    # Create default config
    config = EmotionConfig()

    # Print emotion statistics
    stats = get_emotion_statistics(config)
    print(f"\nEmotion Statistics:")
    print(f"  Total emotions: {stats['num_emotions']}")
    print(f"  Arousal range: [{stats['arousal_min']:.2f}, {stats['arousal_max']:.2f}]")
    print(f"  Arousal mean: {stats['arousal_mean']:.2f}")
    print(f"  Arousal std: {stats['arousal_std']:.2f}")
    print(f"  High arousal emotions: {stats['high_arousal_count']}")
    print(f"  Medium arousal emotions: {stats['medium_arousal_count']}")
    print(f"  Low arousal emotions: {stats['low_arousal_count']}")

    # Sample some emotions
    print(f"\nSample Emotions (n=10):")
    samples = sample_emotions_batch(10, config)
    for i, (emotion, arousal) in enumerate(samples, 1):
        category = config.classify_arousal(arousal)
        print(f"  {i}. {emotion:15s} | Arousal: {arousal:5.2f} | Category: {category}")

    # Test with noise
    print(f"\nTesting arousal with noise (std=0.1):")
    config_noisy = EmotionConfig(arousal_noise_std=0.1)
    print(f"  Base arousal for 'excited': {EMOTION_AROUSAL_MAP['excited']:.2f}")
    for i in range(5):
        arousal = config_noisy.get_arousal('excited', add_noise=True)
        print(f"    Sample {i+1}: {arousal:.3f}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    test_emotion()
