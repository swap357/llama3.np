#!/usr/bin/env python
"""
Python bytecode analyzer for llama3.np

This script analyzes the bytecode of key functions in the Llama model to understand
Python interpreter overhead and potential optimization points.
"""
import dis
import inspect
import argparse
from llama3 import Llama, softmax, silu, apply_rotary_emb, repeat_kv
from datetime import datetime
import os
import json

def analyze_function(func, name=None):
    """Analyze bytecode of a function and return the results"""
    if name is None:
        name = func.__name__
    
    # Get source code
    try:
        source = inspect.getsource(func)
    except (TypeError, OSError):
        source = "Source code not available"
    
    # Get bytecode
    bytecode = dis.Bytecode(func)
    instructions = list(bytecode)
    
    # Analyze bytecode
    instruction_counts = {}
    for instr in instructions:
        if instr.opname in instruction_counts:
            instruction_counts[instr.opname] += 1
        else:
            instruction_counts[instr.opname] = 1
    
    # Format bytecode as text
    # Get bytecode as string using string buffer
    from io import StringIO
    buffer = StringIO()
    dis.dis(func, file=buffer)
    bytecode_text = buffer.getvalue()
    
    # Create result dictionary
    result = {
        "name": name,
        "source": source,
        "bytecode": bytecode_text,
        "instruction_summary": instruction_counts,
        "instruction_count": len(instructions),
    }
    
    return result

def analyze_llama_model():
    """Analyze key functions in the Llama model"""
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "description": "Bytecode analysis of llama3.np functions"
        },
        "functions": {}
    }
    
    # Analyze utility functions
    for func, name in [
        (softmax, "softmax"),
        (silu, "silu"),
        (apply_rotary_emb, "apply_rotary_emb"),
        (repeat_kv, "repeat_kv")
    ]:
        results["functions"][name] = analyze_function(func)
    
    # Analyze Llama class methods
    llama_methods = [
        ("__init__", Llama.__init__),
        ("__call__", Llama.__call__),
        ("generate", Llama.generate),
    ]
    
    for name, method in llama_methods:
        results["functions"][f"Llama.{name}"] = analyze_function(method, f"Llama.{name}")
    
    return results

def save_results(results, filename=None):
    """Save analysis results to a JSON file"""
    if filename is None:
        os.makedirs("analysis", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis/bytecode_analysis_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    return filename

def print_summary(results):
    """Print a summary of the bytecode analysis"""
    print("\n" + "=" * 50)
    print("LLAMA3.NP BYTECODE ANALYSIS SUMMARY")
    print("=" * 50)
    
    print("\nFunction Complexity (by instruction count):")
    # Sort functions by instruction count
    sorted_funcs = sorted(
        [(name, data["instruction_count"]) for name, data in results["functions"].items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    for name, count in sorted_funcs:
        print(f"  {name}: {count} instructions")
    
    print("\nMost Common Instructions (across all functions):")
    # Combine instruction counts across all functions
    all_instructions = {}
    for func_data in results["functions"].values():
        for instr, count in func_data["instruction_summary"].items():
            if instr in all_instructions:
                all_instructions[instr] += count
            else:
                all_instructions[instr] = count
    
    # Sort by count
    sorted_instrs = sorted(
        [(instr, count) for instr, count in all_instructions.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    for instr, count in sorted_instrs[:15]:  # Show top 15
        print(f"  {instr}: {count}")
    
    print("\nDetailed analysis saved to 'analysis/' directory")

if __name__ == "__main__":
    # Ensure analysis directory exists
    os.makedirs("analysis", exist_ok=True)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Analyze Llama3 model bytecode")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    args = parser.parse_args()
    
    print("Analyzing Llama3 model bytecode...")
    
    # Run analysis
    results = analyze_llama_model()
    
    # Save results
    results_file = save_results(results, args.output)
    print(f"\nResults saved to {results_file}")
    
    # Print summary
    print_summary(results)