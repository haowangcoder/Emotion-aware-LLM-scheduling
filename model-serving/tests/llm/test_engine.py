"""
Test for LLM engine module.
"""

from llm.engine import LLMEngine


def test_llm_engine():
    """Test LLM engine with sample prompts."""
    engine = LLMEngine()

    # Test with a small model for quick verification
    test_model = "meta-llama/Meta-Llama-3-8B-Instruct"

    print(f"\n{'='*60}")
    print("Testing LLM Engine")
    print(f"{'='*60}\n")

    # Load model
    print("Loading model...")
    success = engine.load_model(test_model)

    if not success:
        print("Failed to load model!")
        return

    print(f"\nModel info: {engine.get_model_info()}\n")

    # Test prompts with different emotions
    test_prompts = [
        {
            "emotion": "excited",
            "prompt": "System: You are an empathetic conversation assistant.\nUser: I just got accepted into my dream university! I can't believe it!\nAssistant:"
        },
        {
            "emotion": "sad",
            "prompt": "System: You are an empathetic conversation assistant.\nUser: My pet passed away yesterday. I'm feeling so empty.\nAssistant:"
        }
    ]

    for i, test in enumerate(test_prompts, 1):
        print(f"\nTest {i} - Emotion: {test['emotion']}")
        print(f"Prompt: {test['prompt'][:100]}...")

        result = engine.generate(
            prompt=test['prompt'],
            max_new_tokens=50,
            temperature=0.7
        )

        if result['error']:
            print(f"ERROR: {result['error']}")
        else:
            print(f"Response: {result['response_text']}")
            print(f"Time: {result['execution_time']:.3f}s")
            print(f"Tokens: {result['output_token_length']}")
            print(f"Fallback used: {result['fallback_used']}")
        print("-" * 60)


if __name__ == "__main__":
    test_llm_engine()
