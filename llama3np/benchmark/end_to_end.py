"""
End-to-end benchmarks for model inference.
"""

import time
import numpy as np
from typing import Dict, Any

from ..model.base import Llama as BaseLlama
from ..model.optimized import Llama as OptimizedLlama
from ..utils.tokenizer import Tokenizer
from ..utils.optimized_tokenizer import OptimizedTokenizer
from ..utils.config import ModelArgs


def benchmark_inference(prompt, max_tokens=50):
    """
    Benchmark end-to-end inference performance.
    
    Args:
        prompt: Input text to generate from
        max_tokens: Maximum tokens to generate
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"End-to-end inference benchmark: '{prompt}' (generate {max_tokens} tokens)")
    
    # Initialize models
    args = ModelArgs()
    args.max_new_tokens = max_tokens
    
    print("Loading base model...")
    base_tokenizer = Tokenizer("./tokenizer.model.np")
    base_model = BaseLlama("./stories15M.model.npz", args)
    
    print("Loading optimized model...")
    opt_tokenizer = OptimizedTokenizer("./tokenizer.model.np")
    opt_model = OptimizedLlama("./stories15M.model.npz", args)
    
    # Tokenize inputs
    tokenize_start = time.time()
    base_tokens = base_tokenizer.encode(prompt)
    base_tokenize_time = time.time() - tokenize_start
    
    tokenize_start = time.time()
    opt_tokens = opt_tokenizer.encode(prompt)
    opt_tokenize_time = time.time() - tokenize_start
    
    base_input_ids = np.array([base_tokens])
    opt_input_ids = np.array([opt_tokens])
    
    # Measure prefill phase (first token)
    print("Measuring prefill phase...")
    
    # Base model prefill
    start_time = time.time()
    base_logits = base_model(base_input_ids, 0)
    base_prefill_time = time.time() - start_time
    
    # Optimized model prefill
    start_time = time.time()
    opt_logits = opt_model(opt_input_ids, 0)
    opt_prefill_time = time.time() - start_time
    
    # Calculate first tokens
    next_token_base = base_logits[:, -1, :].argmax(-1, keepdims=True)
    next_token_opt = opt_logits[:, -1, :].argmax(-1, keepdims=True)
    
    # Measure decode phase (subsequent tokens)
    print("Measuring decode phase...")
    n_decode_steps = min(10, max_tokens - 1)  # Limit for benchmarking
    
    # Base model decode
    base_tokens_generated = []
    base_decode_times = []
    token_position = len(base_tokens)
    for i in range(n_decode_steps):
        start_time = time.time()
        base_logits = base_model(next_token_base, token_position)
        next_token_base = base_logits[:, -1, :].argmax(-1, keepdims=True)
        base_decode_times.append(time.time() - start_time)
        base_tokens_generated.append(next_token_base[0, 0])
        token_position += 1
    
    # Optimized model decode
    opt_tokens_generated = []
    opt_decode_times = []
    token_position = len(opt_tokens)
    for i in range(n_decode_steps):
        start_time = time.time()
        opt_logits = opt_model(next_token_opt, token_position)
        next_token_opt = opt_logits[:, -1, :].argmax(-1, keepdims=True)
        opt_decode_times.append(time.time() - start_time)
        opt_tokens_generated.append(next_token_opt[0, 0])
        token_position += 1
    
    # Calculate metrics
    base_avg_decode = sum(base_decode_times) / len(base_decode_times)
    opt_avg_decode = sum(opt_decode_times) / len(opt_decode_times)
    
    tokenize_speedup = base_tokenize_time / opt_tokenize_time if opt_tokenize_time > 0 else 0
    prefill_speedup = base_prefill_time / opt_prefill_time if opt_prefill_time > 0 else 0
    decode_speedup = base_avg_decode / opt_avg_decode if opt_avg_decode > 0 else 0
    
    # Calculate tokens per second
    base_tokens_per_sec = 1 / base_avg_decode
    opt_tokens_per_sec = 1 / opt_avg_decode
    
    # Check output match
    tokens_match = base_tokens_generated == opt_tokens_generated
    if not tokens_match:
        print("WARNING: Generated tokens don't match")
        print(f"Base: {base_tokens_generated}")
        print(f"Optimized: {opt_tokens_generated}")
    
    # Print results
    print("\nTokenization:")
    print(f"  Base: {base_tokenize_time*1000:.2f}ms")
    print(f"  Optimized: {opt_tokenize_time*1000:.2f}ms")
    print(f"  Speedup: {tokenize_speedup:.2f}x")
    
    print("\nPrefill phase:")
    print(f"  Base: {base_prefill_time*1000:.2f}ms")
    print(f"  Optimized: {opt_prefill_time*1000:.2f}ms")
    print(f"  Speedup: {prefill_speedup:.2f}x")
    
    print("\nDecode phase (per token):")
    print(f"  Base: {base_avg_decode*1000:.2f}ms ({base_tokens_per_sec:.2f} tokens/s)")
    print(f"  Optimized: {opt_avg_decode*1000:.2f}ms ({opt_tokens_per_sec:.2f} tokens/s)")
    print(f"  Speedup: {decode_speedup:.2f}x")
    
    print(f"\nOutput match: {'Yes' if tokens_match else 'No'}")
    
    # Decode some tokens for display
    base_text = base_tokenizer.decode(base_tokens_generated[:5])
    opt_text = opt_tokenizer.decode(opt_tokens_generated[:5])
    
    print(f"\nBase output: '{base_text}'")
    print(f"Optimized output: '{opt_text}'")
    
    return {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "tokenization": {
            "base_time_ms": base_tokenize_time * 1000,
            "optimized_time_ms": opt_tokenize_time * 1000,
            "speedup": tokenize_speedup
        },
        "prefill": {
            "base_time_ms": base_prefill_time * 1000,
            "optimized_time_ms": opt_prefill_time * 1000,
            "speedup": prefill_speedup
        },
        "decode": {
            "base_time_ms": base_avg_decode * 1000,
            "optimized_time_ms": opt_avg_decode * 1000,
            "base_tokens_per_second": base_tokens_per_sec,
            "optimized_tokens_per_second": opt_tokens_per_sec,
            "speedup": decode_speedup
        },
        "output_match": tokens_match,
        "samples": {
            "base": base_text,
            "optimized": opt_text
        }
    }