#!/usr/bin/env python
"""
Direct performance comparison between original and optimized implementations.

This script directly compares the key operations that we've optimized:
1. Tokenization
2. RoPE implementation
3. End-to-end performance with prefill and decode phases
"""

import time
import numpy as np
import json
from datetime import datetime
import os
import sys

# Import implementations
from deprecated import llama3
from deprecated.optimized_llama import apply_rotary_emb as optimized_rope
from deprecated.optimized_llama import OptimizedTokenizer
from deprecated.config import ModelArgs

def measure_tokenization(prompt="Once upon a time in a land far away", iterations=100):
    """Measure tokenization performance."""
    print(f"\n=== Tokenization Benchmark ===")
    print(f"Prompt: '{prompt}' ({len(prompt)} chars)")
    print(f"Iterations: {iterations}")
    
    # Initialize tokenizers
    orig_tokenizer = llama3.Tokenizer("./tokenizer.model.np")
    opt_tokenizer = OptimizedTokenizer("./tokenizer.model.np")
    
    # Warmup
    orig_tokens = orig_tokenizer.encode(prompt)
    opt_tokens = opt_tokenizer.encode(prompt)
    
    # Verify tokenization is identical
    tokens_match = orig_tokens == opt_tokens
    if not tokens_match:
        print(f"Warning: Tokenization produced different results!")
        print(f"Original: {orig_tokens}")
        print(f"Optimized: {opt_tokens}")
    
    # Benchmark original tokenizer
    start_time = time.time()
    for _ in range(iterations):
        orig_tokenizer.encode(prompt)
    orig_time = time.time() - start_time
    
    # Benchmark optimized tokenizer
    start_time = time.time()
    for _ in range(iterations):
        opt_tokenizer.encode(prompt)
    opt_time = time.time() - start_time
    
    # Calculate speedup
    speedup = orig_time / opt_time if opt_time > 0 else 0
    
    # Print results
    print(f"Original: {orig_time*1000/iterations:.2f}ms per call")
    print(f"Optimized: {opt_time*1000/iterations:.2f}ms per call")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Output match: {'Yes' if tokens_match else 'No'}")
    
    return {
        "prompt": prompt,
        "original_time_ms": orig_time*1000/iterations,
        "optimized_time_ms": opt_time*1000/iterations,
        "speedup": speedup,
        "tokens_match": tokens_match
    }

def measure_rope(batch_size=32, seq_len=256, n_heads=6, head_dim=48, iterations=100):
    """Measure RoPE implementation performance."""
    print(f"\n=== RoPE Benchmark ===")
    print(f"Batch: {batch_size}, Seq: {seq_len}, Heads: {n_heads}, Dim: {head_dim}")
    print(f"Iterations: {iterations}")
    
    # Create test data
    xq = np.random.random((batch_size, seq_len, n_heads, head_dim)).astype(np.float32)
    xk = np.random.random((batch_size, seq_len, n_heads, head_dim)).astype(np.float32)
    freqs_cos, freqs_sin = llama3.compute_cos_sin_cache(head_dim, seq_len)
    
    # Warmup
    llama3.apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
    optimized_rope(xq, xk, freqs_cos, freqs_sin)
    
    # Benchmark original RoPE
    start_time = time.time()
    for _ in range(iterations):
        llama3.apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
    orig_time = time.time() - start_time
    
    # Benchmark optimized RoPE
    start_time = time.time()
    for _ in range(iterations):
        optimized_rope(xq, xk, freqs_cos, freqs_sin)
    opt_time = time.time() - start_time
    
    # Calculate speedup
    speedup = orig_time / opt_time if opt_time > 0 else 0
    
    # Print results
    print(f"Original: {orig_time*1000/iterations:.2f}ms per call")
    print(f"Optimized: {opt_time*1000/iterations:.2f}ms per call")
    print(f"Speedup: {speedup:.2f}x")
    
    return {
        "batch": batch_size,
        "seq_len": seq_len,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "original_time_ms": orig_time*1000/iterations,
        "optimized_time_ms": opt_time*1000/iterations,
        "speedup": speedup
    }

def measure_prefill_decode(prompt="Once upon a time", n_decode_steps=10):
    """
    Separately measure prefill and decode phases for the model.
    
    This gives us a clearer picture of performance differences between the 
    implementations in these two distinct phases.
    """
    print(f"\n=== Prefill & Decode Phase Benchmark ===")
    print(f"Prompt: '{prompt}'")
    print(f"Decode steps: {n_decode_steps}")
    
    # Initialize models and tokenizers
    args = ModelArgs()
    
    print("Loading original model...")
    orig_tokenizer = llama3.Tokenizer("./tokenizer.model.np")
    orig_model = llama3.Llama("./stories15M.model.npz", args)
    
    print("Loading optimized model...")
    opt_tokenizer = OptimizedTokenizer("./tokenizer.model.np")
    from optimized_llama import Llama as OptimizedLlama
    opt_model = OptimizedLlama("./stories15M.model.npz", args)
    
    # Tokenize input
    orig_tokens = orig_tokenizer.encode(prompt)
    opt_tokens = opt_tokenizer.encode(prompt)
    
    orig_input_ids = np.array([orig_tokens])
    opt_input_ids = np.array([opt_tokens])
    
    # Measure prefill phase (first forward pass)
    print("Measuring prefill phase...")
    
    # Original model prefill
    start_time = time.time()
    orig_logits = orig_model(orig_input_ids, 0)
    orig_prefill_time = time.time() - start_time
    
    # Optimized model prefill
    start_time = time.time()
    opt_logits = opt_model(opt_input_ids, 0)
    opt_prefill_time = time.time() - start_time
    
    # Calculate prefill speedup
    prefill_speedup = orig_prefill_time / opt_prefill_time if opt_prefill_time > 0 else 0
    
    print(f"Prefill - Original: {orig_prefill_time*1000:.2f}ms, "
          f"Optimized: {opt_prefill_time*1000:.2f}ms, "
          f"Speedup: {prefill_speedup:.2f}x")
    
    # Get next token IDs
    next_token_orig = orig_logits[:, -1, :].argmax(-1, keepdims=True)
    next_token_opt = opt_logits[:, -1, :].argmax(-1, keepdims=True)
    
    # Measure decode phase (subsequent tokens)
    print("Measuring decode phase...")
    
    # Original model decode
    token_position = len(orig_tokens)
    orig_decode_times = []
    for i in range(n_decode_steps):
        start_time = time.time()
        orig_logits = orig_model(next_token_orig, token_position)
        next_token_orig = orig_logits[:, -1, :].argmax(-1, keepdims=True)
        orig_decode_times.append(time.time() - start_time)
        token_position += 1
    
    # Optimized model decode
    token_position = len(opt_tokens)
    opt_decode_times = []
    for i in range(n_decode_steps):
        start_time = time.time()
        opt_logits = opt_model(next_token_opt, token_position)
        next_token_opt = opt_logits[:, -1, :].argmax(-1, keepdims=True)
        opt_decode_times.append(time.time() - start_time)
        token_position += 1
    
    # Calculate average decode times
    avg_orig_decode = sum(orig_decode_times) / len(orig_decode_times)
    avg_opt_decode = sum(opt_decode_times) / len(opt_decode_times)
    decode_speedup = avg_orig_decode / avg_opt_decode if avg_opt_decode > 0 else 0
    
    print(f"Decode (avg) - Original: {avg_orig_decode*1000:.2f}ms, "
          f"Optimized: {avg_opt_decode*1000:.2f}ms, "
          f"Speedup: {decode_speedup:.2f}x")
    
    # Tokens per second
    orig_tokens_per_second = 1 / avg_orig_decode
    opt_tokens_per_second = 1 / avg_opt_decode
    
    print(f"Tokens/sec - Original: {orig_tokens_per_second:.2f}, "
          f"Optimized: {opt_tokens_per_second:.2f}")
    
    return {
        "prompt": prompt,
        "prefill": {
            "original_time_ms": orig_prefill_time * 1000,
            "optimized_time_ms": opt_prefill_time * 1000,
            "speedup": prefill_speedup
        },
        "decode": {
            "original_time_ms": avg_orig_decode * 1000,
            "optimized_time_ms": avg_opt_decode * 1000,
            "speedup": decode_speedup,
            "original_tokens_per_second": orig_tokens_per_second,
            "optimized_tokens_per_second": opt_tokens_per_second
        }
    }

def save_results(results):
    """Save benchmark results to a file."""
    os.makedirs("benchmark_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_results/direct_benchmark_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary file
    summary_file = f"benchmark_results/direct_benchmark_{timestamp}_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Direct Benchmark Summary\n")
        f.write(f"======================\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Tokenization results
        f.write(f"Tokenization:\n")
        f.write(f"  Prompt: '{results['tokenization']['prompt']}'\n")
        f.write(f"  Original: {results['tokenization']['original_time_ms']:.2f}ms\n")
        f.write(f"  Optimized: {results['tokenization']['optimized_time_ms']:.2f}ms\n")
        f.write(f"  Speedup: {results['tokenization']['speedup']:.2f}x\n\n")
        
        # RoPE results
        f.write(f"RoPE Implementation:\n")
        f.write(f"  Configuration: B={results['rope']['batch']}, S={results['rope']['seq_len']}, "
                f"H={results['rope']['n_heads']}, D={results['rope']['head_dim']}\n")
        f.write(f"  Original: {results['rope']['original_time_ms']:.2f}ms\n")
        f.write(f"  Optimized: {results['rope']['optimized_time_ms']:.2f}ms\n")
        f.write(f"  Speedup: {results['rope']['speedup']:.2f}x\n\n")
        
        # Prefill & Decode results
        f.write(f"Prefill Phase:\n")
        f.write(f"  Prompt: '{results['prefill_decode']['prompt']}'\n")
        f.write(f"  Original: {results['prefill_decode']['prefill']['original_time_ms']:.2f}ms\n")
        f.write(f"  Optimized: {results['prefill_decode']['prefill']['optimized_time_ms']:.2f}ms\n")
        f.write(f"  Speedup: {results['prefill_decode']['prefill']['speedup']:.2f}x\n\n")
        
        f.write(f"Decode Phase:\n")
        f.write(f"  Original: {results['prefill_decode']['decode']['original_time_ms']:.2f}ms/token "
                f"({results['prefill_decode']['decode']['original_tokens_per_second']:.2f} tokens/s)\n")
        f.write(f"  Optimized: {results['prefill_decode']['decode']['optimized_time_ms']:.2f}ms/token "
                f"({results['prefill_decode']['decode']['optimized_tokens_per_second']:.2f} tokens/s)\n")
        f.write(f"  Speedup: {results['prefill_decode']['decode']['speedup']:.2f}x\n\n")
        
        f.write(f"Overall Impact:\n")
        f.write(f"  Tokenization: {results['tokenization']['speedup']:.2f}x faster\n")
        f.write(f"  RoPE: {results['rope']['speedup']:.2f}x faster\n")
        f.write(f"  Prefill: {results['prefill_decode']['prefill']['speedup']:.2f}x faster\n")
        f.write(f"  Decode: {results['prefill_decode']['decode']['speedup']:.2f}x faster\n")
    
    print(f"\nResults saved to {summary_file}")
    return summary_file

if __name__ == "__main__":
    print("Starting direct benchmarks to compare original and optimized implementations")
    
    all_results = {}
    
    # Tokenization benchmark
    tokenization_results = measure_tokenization(
        prompt="Once upon a time in a land far away, there lived a brave knight who dreamed of adventures.",
        iterations=20
    )
    all_results["tokenization"] = tokenization_results
    
    # RoPE benchmark
    rope_results = measure_rope(
        batch_size=16,
        seq_len=128,
        n_heads=6,
        head_dim=48,
        iterations=20
    )
    all_results["rope"] = rope_results
    
    # Prefill & Decode benchmark
    prefill_decode_results = measure_prefill_decode(
        prompt="Once upon a time in a land far away",
        n_decode_steps=10
    )
    all_results["prefill_decode"] = prefill_decode_results
    
    # Save results
    save_results(all_results)