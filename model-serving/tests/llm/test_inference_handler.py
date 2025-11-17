"""
Test for LLM inference handler module.
"""

import logging
from llm.inference_handler import LLMInferenceHandler
from core.job import Job


def test_llm_inference_handler():
    """Test LLM inference handler with sample jobs."""
    print(f"\n{'='*60}")
    print("Testing LLM Inference Handler")
    print(f"{'='*60}\n")

    # Import needed for test
    from emotion import sample_emotion, EmotionConfig
    from service_time_mapper import map_service_time, ServiceTimeConfig

    # Initialize handler (use smaller model for testing if needed)
    try:
        handler = LLMInferenceHandler(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            dataset_path="./dataset",
            cache_path="/tmp/test_llm_cache.json",
            use_cache=True
        )
    except Exception as e:
        print(f"Failed to initialize handler: {e}")
        print("Make sure you have:")
        print("1. HuggingFace token configured (for Llama models)")
        print("2. Dataset downloaded to ./dataset")
        print("3. Sufficient GPU memory or set device_map='cpu'")
        return

    print(f"\nModel info: {handler.get_model_info()}")

    # Create test jobs with different emotions
    emotion_config = EmotionConfig()
    service_time_config = ServiceTimeConfig()

    test_emotions = ["excited", "sad", "anxious"]

    jobs = []
    for i, emotion in enumerate(test_emotions):
        # Sample emotion (or use predefined)
        arousal = emotion_config.get_arousal(emotion)
        emotion_class = emotion_config.classify_arousal(arousal)

        # Calculate predicted service time
        predicted_time = map_service_time(arousal, service_time_config)

        # Create job
        job = Job(
            job_id=i,
            execution_duration=predicted_time,  # Will be replaced with actual time
            arrival_time=i * 5.0,
            predicted_execution_duration=predicted_time,
            emotion_label=emotion,
            arousal=arousal,
            emotion_class=emotion_class
        )

        jobs.append(job)

    # Execute jobs
    print(f"\n{'='*60}")
    print("Executing test jobs:")
    print(f"{'='*60}\n")

    for job in jobs:
        print(f"\nJob {job.get_job_id()} - Emotion: {job.get_emotion_label()}")
        print("-" * 60)

        success = handler.execute_job(job)

        if success:
            print(f"✓ Success")
            print(f"  Response: {job.get_response_text()[:100]}...")
            print(f"  Actual time: {job.get_actual_execution_duration():.3f}s")
            print(f"  Predicted time: {job.get_predicted_execution_duration():.3f}s")
            print(f"  Output tokens: {job.get_output_token_length()}")
            print(f"  Cached: {job.is_cached()}")
            print(f"  Fallback used: {job.is_fallback_used()}")
        else:
            print(f"✗ Failed")
            print(f"  Error: {job.get_error_msg()}")

    # Save cache and show stats
    print(f"\n{'='*60}")
    print("Cache Statistics:")
    print(f"{'='*60}\n")

    handler.save_cache()
    stats = handler.get_cache_stats()

    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    # Set logging level
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    test_llm_inference_handler()
