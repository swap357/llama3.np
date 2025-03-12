#!/usr/bin/env python
"""
Orchestrates the analysis and profiling of llama3.np

This script runs various profiling and analysis tools in sequence,
collecting and compiling their results.
"""

import subprocess
import argparse
import os
import time
import json
from datetime import datetime

def run_command(command, description=None, env=None):
    """Run a command and return its output"""
    if description:
        print(f"\n{description}:")
        print("-" * 50)
    
    print(f"Running: {command}")
    start_time = time.time()
    
    # Execute command
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        env=env
    )
    
    elapsed = time.time() - start_time
    print(f"Command completed in {elapsed:.2f} seconds")
    
    # Check for errors
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False, None
    
    return True, result.stdout

def main():
    parser = argparse.ArgumentParser(description="Run analysis on llama3.np model")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt for text generation")
    parser.add_argument("--tokens", type=int, default=20, help="Number of tokens to generate")
    parser.add_argument("--output-dir", type=str, default="analysis_results", help="Directory to store results")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save run parameters
    with open(os.path.join(run_dir, "parameters.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # 1. First, let's run a basic inference to establish baseline performance
    success, output = run_command(
        f"python3 llama3.py '{args.prompt}' --max-new-tokens {args.tokens}",
        "Baseline Inference"
    )
    if success and output is not None:
        with open(os.path.join(run_dir, "baseline_output.txt"), "w") as f:
            f.write(output)
    
    # 2. Run bytecode analysis
    success, output = run_command(
        f"python3 analyze_bytecode.py --output {os.path.join(run_dir, 'bytecode_analysis.json')}",
        "Bytecode Analysis"
    )
    if success and output is not None:
        with open(os.path.join(run_dir, "bytecode_summary.txt"), "w") as f:
            f.write(output)
    
    # 3. Run detailed profiling
    success, output = run_command(
        f"python3 profile_inference.py --prompt '{args.prompt}' --max-tokens {args.tokens} --output {os.path.join(run_dir, 'profile_results.json')}",
        "Detailed Profiling"
    )
    if success and output is not None:
        with open(os.path.join(run_dir, "profile_summary.txt"), "w") as f:
            f.write(output)
    
    # 4. Run with Python's built-in profiler for comparison
    success, output = run_command(
        f"python3 -m cProfile -o {os.path.join(run_dir, 'cprofile.prof')} llama3.py '{args.prompt}' --max-new-tokens {args.tokens}",
        "cProfile Run"
    )
    
    # Format the cProfile results
    success, output = run_command(
        f"python3 -c \"import pstats; p = pstats.Stats('{os.path.join(run_dir, 'cprofile.prof')}'); p.sort_stats('cumulative').print_stats(30)\"",
        "cProfile Results"
    )
    if success and output is not None:
        with open(os.path.join(run_dir, "cprofile_summary.txt"), "w") as f:
            f.write(output)
    
    print(f"\nAnalysis complete. Results saved to: {run_dir}")
    print("You can review these files to understand the model's performance characteristics.")

if __name__ == "__main__":
    main()