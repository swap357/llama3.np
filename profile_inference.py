#!/usr/bin/env python
"""
Baseline profiling for llama3.np

This script provides a clean, focused profiling of the Llama3 model inference,
recording both high-level metrics and detailed function-level performance data.
"""
import time
import cProfile
import pstats
import io
import argparse
import numpy as np
from contextlib import contextmanager
import os
import json
from datetime import datetime

# Import the Llama model and related utilities
from deprecated.llama3 import Llama, ModelArgs
from deprecated.tokenizer import Tokenizer
from deprecated.utils import load_parameters

@contextmanager
def profile_section(name, results):
    """Profile a section of code and add results to a dictionary"""
    start_time = time.time()
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        yield
    finally:
        profiler.disable()
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Add elapsed time to results
        results[name] = {
            "elapsed_seconds": elapsed,
        }
        
        # Add profiler stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Print top 20 functions by cumulative time
        results[name]["profile"] = s.getvalue()
        
        # Save raw profiler data for later analysis
        profile_dir = os.path.join("profiles", name.replace(" ", "_"))
        os.makedirs(profile_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ps.dump_stats(os.path.join(profile_dir, f"profile_{timestamp}.prof"))

def run_inference(prompt, max_tokens=100, temperature=0.8, generate_only=False):
    """Run inference with profiling"""
    results = {}
    
    # Record memory usage before model initialization
    initial_mem = get_memory_usage()
    results["initial_memory_mb"] = initial_mem
    
    # Profile model loading
    with profile_section("model_loading", results):
        model_path = "stories15M.model.npz"
        tokenizer_path = "tokenizer.model.np"
        
        # Load tokenizer
        tokenizer = Tokenizer(tokenizer_path)
        
        # Set model arguments
        args = ModelArgs()
        args.max_seq_len = 256  # Limit context size for profiling
        args.max_new_tokens = max_tokens
        
        # Load model - need to pass model_path to Llama directly
        model = Llama(model_path, args)
    
    # Record memory after model loading
    post_load_mem = get_memory_usage()
    results["post_load_memory_mb"] = post_load_mem
    results["model_memory_mb"] = post_load_mem - initial_mem
    
    # Tokenize input
    with profile_section("tokenization", results):
        # Convert to the expected 2D array format
        tokens = np.array([tokenizer.encode(prompt)])
    
    # Profile inference
    output_tokens = []
    with profile_section("inference", results):
        start_time = time.time()
        if generate_only:
            # Just generate tokens, don't collect them (for pure perf measurement)
            for token in model.generate(tokens, max_tokens):
                pass
        else:
            # Generate and collect tokens
            for token in model.generate(tokens, max_tokens):
                # Extract token ID from the tensor
                token_id = token[0].tolist()
                output_tokens.append(token_id)
        inference_time = time.time() - start_time
    
    # Calculate tokens per second
    total_gen_tokens = len(output_tokens) if not generate_only else max_tokens
    results["tokens_per_second"] = total_gen_tokens / inference_time
    results["inference_time_seconds"] = inference_time
    results["tokens_generated"] = total_gen_tokens
    
    # Decode output tokens if we collected them
    if not generate_only and output_tokens:
        with profile_section("decoding", results):
            # For each token, decode it individually and concatenate
            output_text = ""
            for token in output_tokens:
                output_text += tokenizer.decode(token)
        results["output_text"] = output_text
    
    # Final memory usage
    results["final_memory_mb"] = get_memory_usage()
    
    return results

def get_memory_usage():
    """Get current memory usage of the Python process in MB"""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert bytes to MB

def save_results(results, filename=None):
    """Save profiling results to a JSON file"""
    if filename is None:
        os.makedirs("profiles", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"profiles/profile_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    return filename

def print_summary(results):
    """Print a summary of the profiling results"""
    print("\n" + "=" * 50)
    print("LLAMA3.NP INFERENCE PROFILING SUMMARY")
    print("=" * 50)
    
    print(f"\nModel Memory: {results['model_memory_mb']:.2f} MB")
    print(f"Tokens Generated: {results['tokens_generated']}")
    print(f"Inference Time: {results['inference_time_seconds']:.2f} seconds")
    print(f"Performance: {results['tokens_per_second']:.2f} tokens/second")
    
    print("\nTime Breakdown:")
    print(f"  Model Loading: {results['model_loading']['elapsed_seconds']:.2f}s")
    print(f"  Tokenization: {results['tokenization']['elapsed_seconds']:.2f}s")
    print(f"  Inference: {results['inference']['elapsed_seconds']:.2f}s")
    if 'decoding' in results:
        print(f"  Decoding: {results['decoding']['elapsed_seconds']:.2f}s")
    
    # Print top functions from inference profile
    print("\nTop Functions (Inference):")
    inference_profile = results['inference']['profile']
    # Extract and print only the function stats (skip the header)
    lines = inference_profile.split('\n')
    stat_lines = [line for line in lines if line and not line.startswith('   ncalls')]
    for line in stat_lines[:10]:  # Print top 10 functions
        if line.strip():  # Skip empty lines
            print(f"  {line.strip()}")
    
    print("\nDetailed profile data saved to 'profiles/' directory")

if __name__ == "__main__":
    # Ensure profiles directory exists
    os.makedirs("profiles", exist_ok=True)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Profile Llama3 model inference")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=50, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature")
    parser.add_argument("--generate-only", action="store_true", help="Only measure generation, don't collect tokens")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    args = parser.parse_args()
    
    # We need psutil for memory measurement
    try:
        import psutil
    except ImportError:
        print("Warning: psutil not installed. Memory measurements will be disabled.")
        get_memory_usage = lambda: 0
    
    print(f"Profiling inference with prompt: '{args.prompt}'")
    print(f"Generating up to {args.max_tokens} tokens...")
    
    # Run profiling
    results = run_inference(
        args.prompt, 
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        generate_only=args.generate_only
    )
    
    # Save results
    results_file = save_results(results, args.output)
    print(f"\nResults saved to {results_file}")
    
    # Print summary
    print_summary(results)