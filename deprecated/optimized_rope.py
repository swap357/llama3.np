#!/usr/bin/env python
"""
Optimized implementation of Rotary Position Embedding (RoPE)

This file contains an optimized implementation of RoPE that uses direct indexing
instead of the complex reshape/split/stack operations in the original implementation.
"""

import numpy as np
import time

# Import the original implementation for comparison
from llama3 import apply_rotary_emb as original_rope

def optimized_rope(xq, xk, freqs_cos, freqs_sin):
    """
    Optimized implementation of RoPE using direct indexing.
    
    Args:
        xq: Query vectors [batch, seq_len, n_heads, head_dim]
        xk: Key vectors [batch, seq_len, n_kv_heads, head_dim]
        freqs_cos: Cosine of frequencies [seq_len, head_dim//2]
        freqs_sin: Sine of frequencies [seq_len, head_dim//2]
        
    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    # Get real and imaginary parts using direct indexing
    # Even indices are real, odd indices are imaginary
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

def benchmark_rope(batch_size=1, seq_len=256, n_heads=6, head_dim=48, n_kv_heads=6, iterations=100):
    """
    Benchmark the original and optimized RoPE implementations.
    """
    # Create random input data
    xq = np.random.random((batch_size, seq_len, n_heads, head_dim)).astype(np.float32)
    xk = np.random.random((batch_size, seq_len, n_kv_heads, head_dim)).astype(np.float32)
    
    # Create frequency caches
    from llama3 import compute_cos_sin_cache
    freqs_cos, freqs_sin = compute_cos_sin_cache(head_dim, seq_len)
    
    # Warm up
    for _ in range(5):
        original_rope(xq, xk, freqs_cos, freqs_sin)
        optimized_rope(xq, xk, freqs_cos, freqs_sin)
    
    # Benchmark original implementation
    original_times = []
    for _ in range(iterations):
        start = time.time()
        original_q, original_k = original_rope(xq, xk, freqs_cos, freqs_sin)
        original_times.append(time.time() - start)
    
    # Benchmark optimized implementation
    optimized_times = []
    for _ in range(iterations):
        start = time.time()
        optimized_q, optimized_k = optimized_rope(xq, xk, freqs_cos, freqs_sin)
        optimized_times.append(time.time() - start)
    
    # Verify correctness
    original_q, original_k = original_rope(xq, xk, freqs_cos, freqs_sin)
    optimized_q, optimized_k = optimized_rope(xq, xk, freqs_cos, freqs_sin)
    
    q_match = np.allclose(original_q, optimized_q, rtol=1e-5, atol=1e-5)
    k_match = np.allclose(original_k, optimized_k, rtol=1e-5, atol=1e-5)
    
    # Calculate statistics
    avg_original = np.mean(original_times) * 1000  # ms
    avg_optimized = np.mean(optimized_times) * 1000  # ms
    speedup = avg_original / avg_optimized if avg_optimized > 0 else 0
    
    # Print results
    print(f"RoPE Benchmark Results (batch={batch_size}, seq_len={seq_len}, n_heads={n_heads}, head_dim={head_dim}):")
    print(f"Original implementation: {avg_original:.3f} ms")
    print(f"Optimized implementation: {avg_optimized:.3f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Output matches: {'Yes' if q_match and k_match else 'No'}")
    if not q_match or not k_match:
        print(f"  - Query match: {q_match}")
        print(f"  - Key match: {k_match}")
        if not q_match:
            print(f"  - Max difference (q): {np.max(np.abs(original_q - optimized_q))}")
        if not k_match:
            print(f"  - Max difference (k): {np.max(np.abs(original_k - optimized_k))}")
    
    return {
        "original_ms": avg_original,
        "optimized_ms": avg_optimized,
        "speedup": speedup,
        "output_matches": q_match and k_match
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark RoPE implementations")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--heads", type=int, default=6, help="Number of attention heads")
    parser.add_argument("--dim", type=int, default=48, help="Head dimension")
    parser.add_argument("--iter", type=int, default=100, help="Number of iterations")
    args = parser.parse_args()
    
    benchmark_rope(
        batch_size=args.batch,
        seq_len=args.seq_len, 
        n_heads=args.heads,
        head_dim=args.dim,
        n_kv_heads=args.heads,  # Assuming equal q and kv heads
        iterations=args.iter
    )