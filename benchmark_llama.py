#!/usr/bin/env python
"""
Benchmark script to compare the performance of original and optimized llama3.np implementations.

This script measures performance differences between the original and optimized implementations,
ensuring output consistency and providing detailed metrics.
"""

import time
import sys
import argparse
import numpy as np
import json
from datetime import datetime
import os
import subprocess
from typing import List, Dict, Any, Tuple

# Import both implementations
import llama3
from optimized_llama import Llama as OptimizedLlama
from optimized_llama import OptimizedTokenizer
from config import ModelArgs

def measure_memory_usage():
    """Measure current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / 1024 / 1024
    except ImportError:
        print("Warning: psutil not available, memory measurement disabled")
        return 0

def run_benchmark(prompt: str, max_tokens: int = 50) -> Dict[str, Any]:
    """
    Run benchmark comparing original and optimized implementations.
    
    Args:
        prompt: Input text to use for inference
        max_tokens: Number of tokens to generate
        
    Returns:
        Dictionary with benchmark results
    """
    results = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "timestamp": datetime.now().isoformat(),
        "metrics": {},
    }
    
    # Load models
    args = ModelArgs()
    args.max_new_tokens = max_tokens
    
    # Measure initial memory
    initial_memory = measure_memory_usage()
    results["metrics"]["initial_memory_mb"] = initial_memory
    
    # Original implementation
    print("Loading original model...")
    orig_load_start = time.time()
    orig_tokenizer = llama3.Tokenizer("./tokenizer.model.np")
    orig_model = llama3.Llama("./stories15M.model.npz", args)
    orig_load_time = time.time() - orig_load_start
    
    # Measure memory after loading original model
    orig_memory = measure_memory_usage()
    results["metrics"]["original_model_memory_mb"] = orig_memory - initial_memory
    
    # Optimized implementation
    print("Loading optimized model...")
    opt_load_start = time.time()
    opt_tokenizer = OptimizedTokenizer("./tokenizer.model.np")
    opt_model = OptimizedLlama("./stories15M.model.npz", args)
    opt_load_time = time.time() - opt_load_start
    
    # Measure memory after loading optimized model
    opt_memory = measure_memory_usage()
    results["metrics"]["optimized_model_memory_mb"] = opt_memory - orig_memory
    
    print(f"Memory usage - Original: {results['metrics']['original_model_memory_mb']:.2f} MB, " 
          f"Optimized: {results['metrics']['optimized_model_memory_mb']:.2f} MB")
    
    # Tokenize prompt
    print("Tokenizing input...")
    orig_tokenize_start = time.time()
    orig_tokens = orig_tokenizer.encode(prompt)
    orig_tokenize_time = time.time() - orig_tokenize_start
    results["metrics"]["original_tokenize_time"] = orig_tokenize_time
    
    opt_tokenize_start = time.time()
    opt_tokens = opt_tokenizer.encode(prompt)
    opt_tokenize_time = time.time() - opt_tokenize_start
    results["metrics"]["optimized_tokenize_time"] = opt_tokenize_time
    results["metrics"]["tokenize_speedup"] = orig_tokenize_time / opt_tokenize_time if opt_tokenize_time > 0 else 0
    
    print(f"Tokenization - Original: {orig_tokenize_time*1000:.2f}ms, "
          f"Optimized: {opt_tokenize_time*1000:.2f}ms, "
          f"Speedup: {results['metrics']['tokenize_speedup']:.2f}x")
    
    # Check if tokens match
    tokens_match = orig_tokens == opt_tokens
    results["metrics"]["tokens_match"] = tokens_match
    if not tokens_match:
        print("Warning: Tokenization produced different results!")
        print(f"Original: {orig_tokens}")
        print(f"Optimized: {opt_tokens}")
    
    # Prepare for generation
    orig_input_ids = np.array([orig_tokens])
    opt_input_ids = np.array([opt_tokens])
    
    # Run generation - explicitly generate max_tokens for both
    print(f"Generating {max_tokens} tokens with original model...")
    orig_output_tokens = []
    orig_gen_start = time.time()
    token_counter = 0
    for token in orig_model.generate(orig_input_ids, max_tokens):
        # Print the token id for debugging
        token_id = token[0].tolist()[0]
        print(f"Original generated token {token_counter}: {token_id} => '{orig_tokenizer.decode([token_id])}'")
        orig_output_tokens.append(token_id)
        token_counter += 1
        if token_counter >= max_tokens:
            break
    orig_gen_time = time.time() - orig_gen_start
    results["metrics"]["original_generation_time"] = orig_gen_time
    results["metrics"]["original_tokens_per_second"] = len(orig_output_tokens) / orig_gen_time
    
    print(f"Generating {max_tokens} tokens with optimized model...")
    opt_output_tokens = []
    opt_gen_start = time.time()
    token_counter = 0
    for token in opt_model.generate(opt_input_ids, max_tokens):
        # Print the token id for debugging
        token_id = token[0].tolist()[0]
        print(f"Optimized generated token {token_counter}: {token_id} => '{opt_tokenizer.decode([token_id])}'")
        opt_output_tokens.append(token_id)
        token_counter += 1
        if token_counter >= max_tokens:
            break
    opt_gen_time = time.time() - opt_gen_start
    results["metrics"]["optimized_generation_time"] = opt_gen_time
    results["metrics"]["optimized_tokens_per_second"] = len(opt_output_tokens) / opt_gen_time
    results["metrics"]["generation_speedup"] = orig_gen_time / opt_gen_time if opt_gen_time > 0 else 0
    
    print(f"Generation - Original: {orig_gen_time:.2f}s ({results['metrics']['original_tokens_per_second']:.2f} tokens/s), "
          f"Optimized: {opt_gen_time:.2f}s ({results['metrics']['optimized_tokens_per_second']:.2f} tokens/s), "
          f"Speedup: {results['metrics']['generation_speedup']:.2f}x")
    
    # Compare generated tokens for correctness
    min_length = min(len(orig_output_tokens), len(opt_output_tokens))
    matching_tokens = sum(1 for i in range(min_length) if orig_output_tokens[i] == opt_output_tokens[i])
    results["metrics"]["token_match_percentage"] = matching_tokens / min_length if min_length > 0 else 0
    results["metrics"]["token_count_match"] = len(orig_output_tokens) == len(opt_output_tokens)
    
    print(f"Output - Length: Original={len(orig_output_tokens)}, Optimized={len(opt_output_tokens)}, "
          f"Match: {results['metrics']['token_match_percentage']*100:.1f}%")
    
    # Decode outputs
    orig_output = orig_tokenizer.decode(orig_output_tokens)
    opt_output = opt_tokenizer.decode(opt_output_tokens)
    results["original_output"] = orig_output
    results["optimized_output"] = opt_output
    
    # Calculate total speedup
    total_orig_time = orig_tokenize_time + orig_gen_time
    total_opt_time = opt_tokenize_time + opt_gen_time
    results["metrics"]["total_speedup"] = total_orig_time / total_opt_time if total_opt_time > 0 else 0
    
    print(f"Overall speedup: {results['metrics']['total_speedup']:.2f}x")
    
    return results

def save_results(results: Dict[str, Any]) -> str:
    """Save benchmark results to a JSON file."""
    os.makedirs("benchmark_results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_results/llama_benchmark_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
        
    # Also create a human-readable summary
    summary_file = f"benchmark_results/llama_benchmark_{timestamp}_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Llama3.np Benchmark Summary\n")
        f.write(f"========================\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Prompt: '{results['prompt']}'\n")
        f.write(f"Max tokens: {results['max_tokens']}\n\n")
        
        f.write(f"Memory Usage:\n")
        f.write(f"  Original model: {results['metrics']['original_model_memory_mb']:.2f} MB\n")
        f.write(f"  Optimized model: {results['metrics']['optimized_model_memory_mb']:.2f} MB\n\n")
        
        f.write(f"Tokenization Performance:\n")
        f.write(f"  Original: {results['metrics']['original_tokenize_time']*1000:.2f}ms\n")
        f.write(f"  Optimized: {results['metrics']['optimized_tokenize_time']*1000:.2f}ms\n")
        f.write(f"  Speedup: {results['metrics']['tokenize_speedup']:.2f}x\n")
        f.write(f"  Tokens match: {'Yes' if results['metrics']['tokens_match'] else 'No'}\n\n")
        
        f.write(f"Generation Performance:\n")
        f.write(f"  Original: {results['metrics']['original_generation_time']:.2f}s "
                f"({results['metrics']['original_tokens_per_second']:.2f} tokens/s)\n")
        f.write(f"  Optimized: {results['metrics']['optimized_generation_time']:.2f}s "
                f"({results['metrics']['optimized_tokens_per_second']:.2f} tokens/s)\n")
        f.write(f"  Speedup: {results['metrics']['generation_speedup']:.2f}x\n\n")
        
        f.write(f"Output Comparison:\n")
        f.write(f"  Token match: {results['metrics']['token_match_percentage']*100:.1f}%\n\n")
        
        f.write(f"Overall Performance:\n")
        f.write(f"  Total speedup: {results['metrics']['total_speedup']:.2f}x\n\n")
        
        f.write(f"Original output: '{results['original_output']}'\n\n")
        f.write(f"Optimized output: '{results['optimized_output']}'\n")
    
    return summary_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark llama3.np performance")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Input prompt for generation")
    parser.add_argument("--tokens", type=int, default=30, help="Number of tokens to generate")
    args = parser.parse_args()
    
    print(f"Starting benchmark with prompt: '{args.prompt}'")
    results = run_benchmark(args.prompt, args.tokens)
    
    summary_file = save_results(results)
    print(f"\nResults saved to {summary_file}")