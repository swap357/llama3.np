"""
Tests for llama3np optimizations.
"""

import numpy as np
import pytest

from llama3np.model.base import apply_rotary_emb as base_rope
from llama3np.model.optimized import apply_rotary_emb as optimized_rope
from llama3np.model.base import compute_cos_sin_cache


def test_rope_output_matches():
    """Test that optimized RoPE produces the same output as the base implementation."""
    # Create test data
    batch_size = 2
    seq_len = 16
    n_heads = 6
    head_dim = 48
    
    xq = np.random.random((batch_size, seq_len, n_heads, head_dim)).astype(np.float32)
    xk = np.random.random((batch_size, seq_len, n_heads, head_dim)).astype(np.float32)
    freqs_cos, freqs_sin = compute_cos_sin_cache(head_dim, seq_len)
    
    # Get outputs from both implementations
    base_q, base_k = base_rope(xq, xk, freqs_cos, freqs_sin)
    opt_q, opt_k = optimized_rope(xq, xk, freqs_cos, freqs_sin)
    
    # Check that they match
    assert np.allclose(base_q, opt_q, rtol=1e-5, atol=1e-5)
    assert np.allclose(base_k, opt_k, rtol=1e-5, atol=1e-5)


def test_optimized_rope_performance():
    """Test that optimized RoPE is faster than the base implementation."""
    # This is more of a benchmark than a test, but it's useful to verify
    # that the optimization actually improves performance
    
    # Create larger test data for more reliable timing
    batch_size = 16
    seq_len = 128
    n_heads = 6
    head_dim = 48
    
    xq = np.random.random((batch_size, seq_len, n_heads, head_dim)).astype(np.float32)
    xk = np.random.random((batch_size, seq_len, n_heads, head_dim)).astype(np.float32)
    freqs_cos, freqs_sin = compute_cos_sin_cache(head_dim, seq_len)
    
    # Warm up
    for _ in range(3):
        base_rope(xq, xk, freqs_cos, freqs_sin)
        optimized_rope(xq, xk, freqs_cos, freqs_sin)
    
    # Measure base implementation
    import time
    iterations = 10
    
    start_time = time.time()
    for _ in range(iterations):
        base_rope(xq, xk, freqs_cos, freqs_sin)
    base_time = time.time() - start_time
    
    # Measure optimized implementation
    start_time = time.time()
    for _ in range(iterations):
        optimized_rope(xq, xk, freqs_cos, freqs_sin)
    opt_time = time.time() - start_time
    
    # Skip performance check on CI
    import os
    if "CI" not in os.environ:
        # Optimized should be faster (this may not always be true depending on
        # hardware and other factors, so we don't fail the test if it's not)
        assert opt_time <= base_time * 1.1, "Optimized RoPE should be faster"