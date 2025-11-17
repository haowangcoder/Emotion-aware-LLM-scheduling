"""
Test for response cache module.
"""

import os
import logging
from llm.response_cache import ResponseCache


def test_response_cache():
    """Test response cache functionality."""
    print(f"\n{'='*60}")
    print("Testing Response Cache")
    print(f"{'='*60}\n")

    # Test 1: Basic get/set
    print("Test 1: Basic cache operations")
    print("-" * 60)

    cache = ResponseCache()

    # Store some responses
    prompts = [
        "User: I'm so excited about my new job!\nAssistant:",
        "User: I feel sad today.\nAssistant:",
        "User: I'm worried about my exam.\nAssistant:"
    ]

    responses = [
        "That's wonderful! Congratulations on your new job!",
        "I'm sorry to hear that. Would you like to talk about it?",
        "It's normal to feel nervous. Have you been preparing?"
    ]

    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        cache_key = cache.set(
            prompt=prompt,
            response_text=response,
            execution_time=1.5 + i * 0.5,
            output_token_length=10 + i * 5,
            model_name="test-model",
            generation_params={"temperature": 0.7, "max_tokens": 50}
        )
        print(f"Stored response {i+1} with key: {cache_key}")

    print(f"\nCache stats: {cache.get_stats()}")

    # Test 2: Retrieve cached responses
    print(f"\n{'='*60}")
    print("Test 2: Retrieve cached responses")
    print("-" * 60)

    for i, prompt in enumerate(prompts):
        result = cache.get(prompt, "test-model")
        if result:
            print(f"\nPrompt {i+1} (CACHE HIT):")
            print(f"  Response: {result['response_text']}")
            print(f"  Time: {result['execution_time']}s")
            print(f"  Tokens: {result['output_token_length']}")
        else:
            print(f"\nPrompt {i+1} (CACHE MISS)")

    # Test cache miss
    new_prompt = "User: Different prompt\nAssistant:"
    result = cache.get(new_prompt, "test-model")
    print(f"\nNew prompt (should be MISS): {'HIT' if result else 'MISS'}")

    print(f"\nFinal cache stats: {cache.get_stats()}")

    # Test 3: Save and load from disk
    print(f"\n{'='*60}")
    print("Test 3: Persistence (save/load)")
    print("-" * 60)

    test_cache_file = "/tmp/test_response_cache.json"

    # Save cache
    print(f"Saving cache to {test_cache_file}...")
    cache.save_to_disk(test_cache_file)

    # Create new cache and load
    print(f"Loading cache from {test_cache_file}...")
    cache2 = ResponseCache(cache_file=test_cache_file)

    print(f"\nLoaded cache stats: {cache2.get_stats()}")

    # Verify loaded cache works
    result = cache2.get(prompts[0], "test-model")
    if result:
        print(f"\nVerifying loaded cache (first prompt):")
        print(f"  Response: {result['response_text']}")
        print(f"  Time: {result['execution_time']}s")

    # Test 4: Cache with different models
    print(f"\n{'='*60}")
    print("Test 4: Multi-model caching")
    print("-" * 60)

    same_prompt = "User: Hello\nAssistant:"

    # Store same prompt for different models
    cache.set(
        prompt=same_prompt,
        response_text="Hi there! (Model A)",
        execution_time=1.0,
        output_token_length=5,
        model_name="model-a"
    )

    cache.set(
        prompt=same_prompt,
        response_text="Hello! (Model B)",
        execution_time=2.0,
        output_token_length=6,
        model_name="model-b"
    )

    # Retrieve for different models
    result_a = cache.get(same_prompt, "model-a")
    result_b = cache.get(same_prompt, "model-b")

    print(f"Model A response: {result_a['response_text']}")
    print(f"Model B response: {result_b['response_text']}")

    # Test 5: Cache size
    print(f"\n{'='*60}")
    print("Test 5: Cache statistics")
    print("-" * 60)

    stats = cache.get_stats()
    size_bytes = cache.get_size_bytes()

    print(f"Number of entries: {stats['num_entries']}")
    print(f"Total requests: {stats['total_requests']}")
    print(f"Cache hits: {stats['hits']}")
    print(f"Cache misses: {stats['misses']}")
    print(f"Hit rate: {stats['hit_rate']:.2%}")
    print(f"Approximate size: {size_bytes / 1024:.2f} KB")

    # Cleanup
    if os.path.exists(test_cache_file):
        os.remove(test_cache_file)
        print(f"\nCleaned up test file: {test_cache_file}")


if __name__ == "__main__":
    # Set logging level
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    test_response_cache()
