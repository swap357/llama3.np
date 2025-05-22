import time

import numpy as np
import pytest

import llama3_simple as llama_functional
from config import ModelArgs
from tokenizer import Tokenizer

# Set random seed for reproducibility
np.random.seed(42)


def test_text_generation():
    """Test FP32 vs FP16 with actual text generation."""
    print("\nTesting text generation:")

    # Initialize tokenizer
    tokenizer = Tokenizer("./tokenizer.model.np")

    args_fp32 = ModelArgs()
    args_fp32.dtype = "float32"

    args_fp16 = ModelArgs()
    args_fp16.dtype = "float16"

    # Initialize models
    model_fp32 = llama_functional.llama_init("./stories15M.model.npz", args_fp32)
    model_fp16 = llama_functional.llama_init("./stories15M.model.npz", args_fp16)

    test_prompts = ["Once upon a time", "I have a dream"]

    for prompt in test_prompts:
        print(f"\nInput prompt: '{prompt}'")
        input_ids = np.array([tokenizer.encode(prompt)])

        # Generate sequences and measure time
        start_fp32 = time.time()
        fp32_output = []
        for token in llama_functional.llama_generate(
            model_fp32, input_ids, args_fp32.max_new_tokens
        ):
            output_id = token[0].tolist()
            if output_id[-1] in [tokenizer.eos_id, tokenizer.bos_id]:
                break
            fp32_output.append(tokenizer.decode(output_id))
        fp32_time = time.time() - start_fp32

        start_fp16 = time.time()
        fp16_output = []
        for token in llama_functional.llama_generate(
            model_fp16, input_ids, args_fp16.max_new_tokens
        ):
            output_id = token[0].tolist()
            if output_id[-1] in [tokenizer.eos_id, tokenizer.bos_id]:
                break
            fp16_output.append(tokenizer.decode(output_id))
        fp16_time = time.time() - start_fp16

        # Compare outputs and timing
        print("\nOutput comparison:")
        print("FP32 output:", "".join(fp32_output))
        print()
        print("FP16 output:", "".join(fp16_output))

        print("\nPerformance comparison:")
        print(f"FP32 generation time: {fp32_time:.4f}s")
        print(f"FP16 generation time: {fp16_time:.4f}s")
        print(f"ratio (FP32/FP16): {fp32_time / fp16_time:.2f}x")


def test_dtype_comparison():
    """Compare model performance and accuracy between FP32 and FP16 implementations."""
    print("\nllama3_simple.py - FP32 vs FP16")

    args_fp32 = ModelArgs()
    args_fp32.dtype = "float32"

    args_fp16 = ModelArgs()
    args_fp16.dtype = "float16"

    # Initialize models with different dtypes
    model_fp32 = llama_functional.llama_init("./stories15M.model.npz", args_fp32)
    model_fp16 = llama_functional.llama_init("./stories15M.model.npz", args_fp16)

    # Create same input for both models
    input_ids = np.random.randint(0, 100, size=(1, 4), dtype=np.int32)
    start_pos = 0

    # Time and run forward passes
    # FP32 forward pass
    start_fp32 = time.time()
    logits_fp32 = llama_functional.llama_forward(model_fp32, input_ids, start_pos)
    fp32_time = time.time() - start_fp32

    # FP16 forward pass
    start_fp16 = time.time()
    logits_fp16 = llama_functional.llama_forward(model_fp16, input_ids, start_pos)
    fp16_time = time.time() - start_fp16

    print("\nPerformance Comparison:")
    print(f"FP32 forward time: {fp32_time:.4f}s")
    print(f"FP16 forward time: {fp16_time:.4f}s")
    print(f"ratio (FP32/FP16): {fp32_time / fp16_time:.2f}x")

    # Compare outputs
    diff = np.abs(logits_fp32 - logits_fp16.astype(np.float32))
    print("\nnumerical differences (FP32 vs FP16):")
    print(f"logits_fp32: {logits_fp32}")
    print(f"logits_fp16: {logits_fp16}")
    print(f"Max absolute diff: {np.max(diff):.2e}")
    print(f"Mean absolute diff: {np.mean(diff):.2e}")
    print(f"Median absolute diff: {np.median(diff):.2e}")

    # Compare top-k predictions
    k = 5
    top_k_fp32 = np.argsort(logits_fp32[0, 0])[-k:][::-1]
    top_k_fp16 = np.argsort(logits_fp16[0, 0])[-k:][::-1]

    print(f"\nTop {k} predictions")
    if not np.array_equal(top_k_fp32, top_k_fp16):
        print("FP32 top-k:", top_k_fp32)
        print("FP16 top-k:", top_k_fp16)


if __name__ == "__main__":
    pytest.main([__file__])
