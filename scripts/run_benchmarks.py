#!/usr/bin/env python
"""
Unified benchmarking script for llama3np.

This script runs component-level and end-to-end benchmarks,
saving results to JSON files and printing summaries.
"""

import os
import json
import argparse
from datetime import datetime
import time

from llama3np.benchmark.components import benchmark_tokenization, benchmark_rope
from llama3np.benchmark.end_to_end import benchmark_inference


def save_benchmark_results(results, name, output_dir="benchmark_results"):
    """Save benchmark results to a file and print summary."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/{name}_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {filename}")
    return filename


def run_component_benchmarks(args):
    """Run benchmarks for individual components."""
    results = {"timestamp": datetime.now().isoformat(), "components": {}}
    
    # Tokenization benchmarks
    if args.tokenization:
        print("\n=== Tokenization Benchmarks ===")
        
        short_prompt = "Hello world"
        medium_prompt = "Once upon a time in a land far away, there lived a brave knight."
        long_prompt = ("Once upon a time in a land far away, there lived a brave knight who dreamed "
                      "of adventure. Every day, he would polish his armor and sharpen his sword, "
                      "preparing for the moment when he would be called to fulfill his destiny.")
        
        # Run benchmarks with different prompt lengths
        results["components"]["tokenization_short"] = benchmark_tokenization(
            short_prompt, iterations=args.iterations
        )
        
        results["components"]["tokenization_medium"] = benchmark_tokenization(
            medium_prompt, iterations=args.iterations
        )
        
        results["components"]["tokenization_long"] = benchmark_tokenization(
            long_prompt, iterations=args.iterations
        )
    
    # RoPE benchmarks
    if args.rope:
        print("\n=== RoPE Benchmarks ===")
        
        # Benchmark with different batch sizes and sequence lengths
        results["components"]["rope_small"] = benchmark_rope(
            batch_size=1,
            seq_len=64,
            n_heads=6,
            head_dim=48,
            iterations=args.iterations
        )
        
        results["components"]["rope_medium"] = benchmark_rope(
            batch_size=16,
            seq_len=128,
            n_heads=6,
            head_dim=48,
            iterations=args.iterations
        )
        
        results["components"]["rope_large"] = benchmark_rope(
            batch_size=32,
            seq_len=256,
            n_heads=6,
            head_dim=48,
            iterations=args.iterations
        )
    
    save_benchmark_results(results, "component_benchmarks")
    return results


def run_inference_benchmarks(args):
    """Run end-to-end inference benchmarks."""
    results = {"timestamp": datetime.now().isoformat(), "inference": {}}
    
    # Run benchmarks with different prompt lengths
    if args.inference:
        print("\n=== Inference Benchmarks ===")
        
        results["inference"]["short_prompt"] = benchmark_inference(
            "Hello world", max_tokens=args.max_tokens
        )
        
        results["inference"]["medium_prompt"] = benchmark_inference(
            "Once upon a time in a land far away", max_tokens=args.max_tokens
        )
    
    save_benchmark_results(results, "inference_benchmarks")
    return results


def main():
    """Run benchmarks based on command line arguments."""
    parser = argparse.ArgumentParser(description="Run llama3np benchmarks")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--tokenization", action="store_true", help="Run tokenization benchmarks")
    parser.add_argument("--rope", action="store_true", help="Run RoPE implementation benchmarks")
    parser.add_argument("--inference", action="store_true", help="Run end-to-end inference benchmarks")
    parser.add_argument("--iterations", type=int, default=20, help="Number of iterations for component benchmarks")
    parser.add_argument("--max-tokens", type=int, default=10, help="Number of tokens to generate in inference benchmarks")
    args = parser.parse_args()
    
    # If no specific benchmark is specified, run all
    if not any([args.tokenization, args.rope, args.inference]) or args.all:
        args.tokenization = args.rope = args.inference = True
    
    start_time = time.time()
    
    # Run component benchmarks
    component_results = run_component_benchmarks(args)
    
    # Run inference benchmarks
    inference_results = run_inference_benchmarks(args)
    
    # Print summary
    total_time = time.time() - start_time
    print(f"\nAll benchmarks completed in {total_time:.2f} seconds")


if __name__ == "__main__":
    main()