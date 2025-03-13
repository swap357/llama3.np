#!/usr/bin/env python
"""
Test the overhead of type annotations in the Llama model.

This script contains simplified versions of key functions from llama3.py
with type annotations removed, and benchmarks them against the original.
"""

import numpy as np
import time
import sys
import math
from llama3 import softmax as original_softmax
from llama3 import silu as original_silu
from llama3 import apply_rotary_emb as original_rope
from llama3 import ModelArgs, Llama

# Simplified versions without type annotations
def simplified_softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def simplified_silu(x):
    return x * (1 / (1 + np.exp(-x)))

def simplified_rope(xq, xk, freqs_cos, freqs_sin):
    # Get real and imaginary parts using direct indexing
    xq_r, xq_i = xq[..., ::2], xq[..., 1::2]
    xk_r, xk_i = xk[..., ::2], xk[..., 1::2]
    
    # Reshape frequencies for broadcasting
    freqs_cos = np.expand_dims(freqs_cos, axis=(0, 2))
    freqs_sin = np.expand_dims(freqs_sin, axis=(0, 2))
    
    # Apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos
    
    # Interleave real and imaginary parts
    xq_out = np.zeros_like(xq)
    xk_out = np.zeros_like(xk)
    xq_out[..., ::2] = xq_out_r
    xq_out[..., 1::2] = xq_out_i
    xk_out[..., ::2] = xk_out_r
    xk_out[..., 1::2] = xk_out_i
    
    return xq_out, xk_out

# Benchmark functions
def benchmark_function(func_original, func_simplified, *args, iterations=100):
    """Benchmark a function with and without type annotations."""
    # Warm up
    for _ in range(5):
        func_original(*args)
        func_simplified(*args)
    
    # Benchmark original implementation
    original_times = []
    for _ in range(iterations):
        start = time.time()
        original_result = func_original(*args)
        original_times.append(time.time() - start)
    
    # Benchmark simplified implementation
    simplified_times = []
    for _ in range(iterations):
        start = time.time()
        simplified_result = func_simplified(*args)
        simplified_times.append(time.time() - start)
    
    # Verify correctness
    if isinstance(original_result, tuple):
        # For functions returning multiple values (like RoPE)
        matches = all(np.allclose(orig, simp) for orig, simp in zip(original_result, simplified_result))
    else:
        matches = np.allclose(original_result, simplified_result)
    
    # Calculate statistics
    avg_original = np.mean(original_times) * 1000  # ms
    avg_simplified = np.mean(simplified_times) * 1000  # ms
    speedup = avg_original / avg_simplified if avg_simplified > 0 else 0
    
    return {
        "original_ms": avg_original,
        "simplified_ms": avg_simplified,
        "speedup": speedup,
        "output_matches": matches
    }

def benchmark_all_functions():
    """Benchmark all functions with and without type annotations."""
    results = {}
    
    # 1. Benchmark softmax
    print("Benchmarking softmax...")
    x = np.random.random((32, 6, 256, 256)).astype(np.float32)  # [batch, heads, seq_len, seq_len]
    softmax_results = benchmark_function(original_softmax, simplified_softmax, x)
    results["softmax"] = softmax_results
    print(f"  Original: {softmax_results['original_ms']:.3f}ms")
    print(f"  Simplified: {softmax_results['simplified_ms']:.3f}ms")
    print(f"  Speedup: {softmax_results['speedup']:.2f}x")
    print(f"  Output matches: {'Yes' if softmax_results['output_matches'] else 'No'}")
    
    # 2. Benchmark silu
    print("\nBenchmarking silu...")
    x = np.random.random((32, 256, 768)).astype(np.float32)  # [batch, seq_len, ffn_dim]
    silu_results = benchmark_function(original_silu, simplified_silu, x)
    results["silu"] = silu_results
    print(f"  Original: {silu_results['original_ms']:.3f}ms")
    print(f"  Simplified: {silu_results['simplified_ms']:.3f}ms")
    print(f"  Speedup: {silu_results['speedup']:.2f}x")
    print(f"  Output matches: {'Yes' if silu_results['output_matches'] else 'No'}")
    
    # 3. Benchmark RoPE
    print("\nBenchmarking RoPE...")
    from llama3 import compute_cos_sin_cache
    xq = np.random.random((32, 256, 6, 48)).astype(np.float32)  # [batch, seq_len, heads, head_dim]
    xk = np.random.random((32, 256, 6, 48)).astype(np.float32)  # [batch, seq_len, kv_heads, head_dim]
    freqs_cos, freqs_sin = compute_cos_sin_cache(48, 256)
    rope_results = benchmark_function(original_rope, simplified_rope, xq, xk, freqs_cos, freqs_sin)
    results["rope"] = rope_results
    print(f"  Original: {rope_results['original_ms']:.3f}ms")
    print(f"  Simplified: {rope_results['simplified_ms']:.3f}ms")
    print(f"  Speedup: {rope_results['speedup']:.2f}x")
    print(f"  Output matches: {'Yes' if rope_results['output_matches'] else 'No'}")
    
    # Summary
    print("\nSummary of Type Annotation Overhead:")
    total_original = sum(r['original_ms'] for r in results.values())
    total_simplified = sum(r['simplified_ms'] for r in results.values())
    overall_speedup = total_original / total_simplified if total_simplified > 0 else 0
    print(f"  Overall speedup: {overall_speedup:.2f}x")
    
    return results

if __name__ == "__main__":
    benchmark_all_functions()