"""
Benchmarks for individual model components.
"""

import time
import numpy as np

# Import implementations for benchmarking
from ..model.base import apply_rotary_emb as base_rope
from ..model.optimized import apply_rotary_emb as optimized_rope
from ..utils.tokenizer import Tokenizer
from ..utils.optimized_tokenizer import OptimizedTokenizer


def benchmark_tokenization(prompt, iterations=100):
    """
    Benchmark tokenization performance.
    
    Args:
        prompt: Input text to tokenize
        iterations: Number of iterations for benchmarking
        
    Returns:
        Dictionary with results
    """
    print(f"Tokenization benchmark: '{prompt}' ({len(prompt)} chars)")
    
    # Initialize tokenizers
    base_tokenizer = Tokenizer("./tokenizer.model.np")
    opt_tokenizer = OptimizedTokenizer("./tokenizer.model.np")
    
    # Warm up
    base_tokens = base_tokenizer.encode(prompt)
    opt_tokens = opt_tokenizer.encode(prompt)
    
    # Verify output
    tokens_match = base_tokens == opt_tokens
    if not tokens_match:
        print("WARNING: Tokenization outputs don't match")
        print(f"Base: {base_tokens}")
        print(f"Optimized: {opt_tokens}")
    
    # Benchmark original implementation
    start_time = time.time()
    for _ in range(iterations):
        base_tokenizer.encode(prompt)
    base_time = time.time() - start_time
    
    # Benchmark optimized implementation
    start_time = time.time()
    for _ in range(iterations):
        opt_tokenizer.encode(prompt)
    opt_time = time.time() - start_time
    
    # Calculate metrics
    base_avg_ms = base_time * 1000 / iterations
    opt_avg_ms = opt_time * 1000 / iterations
    speedup = base_avg_ms / opt_avg_ms if opt_avg_ms > 0 else 0
    
    # Print results
    print(f"Base implementation: {base_avg_ms:.2f}ms per call")
    print(f"Optimized implementation: {opt_avg_ms:.2f}ms per call")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Output match: {'Yes' if tokens_match else 'No'}")
    
    return {
        "component": "tokenization",
        "prompt": prompt,
        "iterations": iterations,
        "base_time_ms": base_avg_ms,
        "optimized_time_ms": opt_avg_ms,
        "speedup": speedup,
        "output_match": tokens_match
    }


def benchmark_rope(batch_size=32, seq_len=256, n_heads=6, head_dim=48, iterations=100):
    """
    Benchmark RoPE implementation performance.
    
    Args:
        batch_size: Batch size for benchmarking
        seq_len: Sequence length for benchmarking
        n_heads: Number of attention heads
        head_dim: Dimension of each attention head
        iterations: Number of iterations for benchmarking
        
    Returns:
        Dictionary with results
    """
    print(f"RoPE benchmark: B={batch_size}, S={seq_len}, H={n_heads}, D={head_dim}")
    
    # Create test data
    from ..model.base import compute_cos_sin_cache
    xq = np.random.random((batch_size, seq_len, n_heads, head_dim)).astype(np.float32)
    xk = np.random.random((batch_size, seq_len, n_heads, head_dim)).astype(np.float32)
    freqs_cos, freqs_sin = compute_cos_sin_cache(head_dim, seq_len)
    
    # Warm up
    base_rope(xq, xk, freqs_cos, freqs_sin)
    optimized_rope(xq, xk, freqs_cos, freqs_sin)
    
    # Verify output
    base_q, base_k = base_rope(xq, xk, freqs_cos, freqs_sin)
    opt_q, opt_k = optimized_rope(xq, xk, freqs_cos, freqs_sin)
    
    q_match = np.allclose(base_q, opt_q, rtol=1e-5, atol=1e-5)
    k_match = np.allclose(base_k, opt_k, rtol=1e-5, atol=1e-5)
    output_match = q_match and k_match
    
    if not output_match:
        print("WARNING: RoPE outputs don't match")
        if not q_match:
            print(f"Q mismatch, max diff: {np.max(np.abs(base_q - opt_q))}")
        if not k_match:
            print(f"K mismatch, max diff: {np.max(np.abs(base_k - opt_k))}")
    
    # Benchmark original implementation
    start_time = time.time()
    for _ in range(iterations):
        base_rope(xq, xk, freqs_cos, freqs_sin)
    base_time = time.time() - start_time
    
    # Benchmark optimized implementation
    start_time = time.time()
    for _ in range(iterations):
        optimized_rope(xq, xk, freqs_cos, freqs_sin)
    opt_time = time.time() - start_time
    
    # Calculate metrics
    base_avg_ms = base_time * 1000 / iterations
    opt_avg_ms = opt_time * 1000 / iterations
    speedup = base_avg_ms / opt_avg_ms if opt_avg_ms > 0 else 0
    
    # Print results
    print(f"Base implementation: {base_avg_ms:.2f}ms per call")
    print(f"Optimized implementation: {opt_avg_ms:.2f}ms per call")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Output match: {'Yes' if output_match else 'No'}")
    
    return {
        "component": "rope",
        "batch_size": batch_size,
        "seq_len": seq_len,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "iterations": iterations,
        "base_time_ms": base_avg_ms,
        "optimized_time_ms": opt_avg_ms,
        "speedup": speedup,
        "output_match": output_match
    }