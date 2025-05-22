import cProfile
import io
import pstats
import time
from pstats import SortKey

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


def test_performance_profiling():
    """Profile and compare FP32 vs FP16 performance to identify bottlenecks."""
    print("\nProfiling FP32 vs FP16 performance:")

    # Test matrix multiplication with different sizes
    sizes = [128, 256, 512, 1024]
    print("\nMatrix multiplication timing comparison:")
    print("Size\tFP32 (s)\tFP16 (s)\tRatio (FP32/FP16)")
    print("-" * 50)

    for size in sizes:
        # Create test matrices
        x = np.random.randn(size, size).astype(np.float32)
        y = np.random.randn(size, size).astype(np.float32)
        x_fp16 = x.astype(np.float16)
        y_fp16 = y.astype(np.float16)

        # Warm up
        _ = np.matmul(x, y)
        _ = np.matmul(x_fp16, y_fp16)

        # Time FP32
        start = time.time()
        for _ in range(10):  # Run multiple times for better timing
            _ = np.matmul(x, y)
        fp32_time = (time.time() - start) / 10

        # Time FP16
        start = time.time()
        for _ in range(10):
            _ = np.matmul(x_fp16, y_fp16)
        fp16_time = (time.time() - start) / 10

        print(
            f"{size}x{size}\t{fp32_time:.6f}\t{fp16_time:.6f}\t{fp32_time / fp16_time:.2f}x"
        )

    # Test type conversion overhead
    print("\nType conversion timing:")
    size = 512
    x = np.random.randn(size, size).astype(np.float32)

    # Time conversions
    start = time.time()
    for _ in range(100):
        _ = x.astype(np.float16)
    fp32_to_fp16_time = (time.time() - start) / 100

    x_fp16 = x.astype(np.float16)
    start = time.time()
    for _ in range(100):
        _ = x_fp16.astype(np.float32)
    fp16_to_fp32_time = (time.time() - start) / 100

    print(f"FP32 to FP16: {fp32_to_fp16_time:.6f}s")
    print(f"FP16 to FP32: {fp16_to_fp32_time:.6f}s")

    # Test if implicit conversion is happening
    print("\nChecking for implicit conversions:")
    size = 512  # Use consistent size
    x = np.random.randn(size, size).astype(np.float32)
    y = np.random.randn(size, size).astype(np.float32)
    x_fp16 = x.astype(np.float16)
    y_fp16 = y.astype(np.float16)

    # Profile a single matrix multiplication
    pr = cProfile.Profile()
    pr.enable()
    result = np.matmul(x_fp16, y_fp16)
    pr.disable()

    # Check result dtype
    print(f"Result dtype: {result.dtype}")

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats(10)  # Show top 10 functions
    print(s.getvalue())

    # Test memory bandwidth
    print("\nMemory bandwidth test:")
    size = 1024
    x = np.random.randn(size, size).astype(np.float32)
    x_fp16 = x.astype(np.float16)

    # Time memory operations
    start = time.time()
    for _ in range(100):
        _ = x.copy()
    fp32_copy_time = (time.time() - start) / 100

    start = time.time()
    for _ in range(100):
        _ = x_fp16.copy()
    fp16_copy_time = (time.time() - start) / 100

    print(f"FP32 copy time: {fp32_copy_time:.6f}s")
    print(f"FP16 copy time: {fp16_copy_time:.6f}s")
    print(f"Copy ratio (FP32/FP16): {fp32_copy_time / fp16_copy_time:.2f}x")

    # Test BLAS implementation
    print("\nBLAS implementation info:")
    print(f"NumPy version: {np.__version__}")
    print(f"BLAS info: {np.show_config()}")


if __name__ == "__main__":
    pytest.main([__file__])
