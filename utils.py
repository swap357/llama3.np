import numpy as np
from collections import defaultdict
import logging
import time

# Set up logging
logger = logging.getLogger(__name__)

# Global timing dictionary to store execution times
timings: dict[str, list[float]] = defaultdict(list)

# Define operation hierarchy
OPERATION_HIERARCHY = {
    'generate.iterations': {
        'llama.total': {
            'llama.layers': {
                'transformer_block.total': {
                    'transformer_block.feedforward': {
                        'feedforward.compute': {}
                    },
                    'transformer_block.attention': {
                        'attention.total': {
                            'attention.qkv_proj': {},
                            'attention.reshape': {},
                            'attention.rope': {},
                            'attention.cache': {},
                            'attention.repeat_kv': {},
                            'attention.transpose': {},
                            'attention.scores': {},
                            'attention.output': {}
                        }
                    }
                }
            },
            'llama.norm': {},
            'llama.head': {}
        }
    }
}

def log_time(func_name: str, operation_name: str, start_time: float):
    """Helper function to log execution time"""
    elapsed = (time.time() - start_time) * 1000  # Convert to milliseconds
    key = f"{func_name}.{operation_name}"
    
    # Combine generate.iteration entries
    if key.startswith('generate.iteration_'):
        timings['generate.iterations'].append(elapsed)
    else:
        timings[key].append(elapsed)
    
    logger.info(f"{key} took {elapsed:.2f}ms")

def print_timing_stats():
    """Print timing statistics in a formatted table"""
    print("\n=== Performance Statistics ===")
    print(f"{'Operation':<40} {'Calls':<8} {'Total(ms)':<12} {'Avg(ms)':<10} {'%Total':<8}")
    print("-" * 80)
    
    # Calculate total time using only the top-level operation
    total_time = sum(timings['generate.iterations']) if 'generate.iterations' in timings and timings['generate.iterations'] else 0
    if total_time == 0:
        print("No timing data collected")
        return
    
    def _print_hierarchy(name, level=0):
        if name not in timings or not timings[name]:
            return
        
        times = timings[name]
        total = sum(times)
        avg = total / len(times)
        pct = (total / total_time) * 100
        
        # Print current operation with proper indentation and alignment
        indent = "  " * level
        print(f"{indent}{name:<{40-level*2}} {len(times):<8d} {total:>11.2f}ms {avg:>8.2f}ms {pct:>7.1f}%")
        
        # Recursively print children if they exist in hierarchy
        if name in OPERATION_HIERARCHY:
            for child_name, child_dict in OPERATION_HIERARCHY[name].items():
                if child_name in timings and timings[child_name]:
                    _print_hierarchy(child_name, level + 1)
                    # Recursively handle nested children
                    if child_dict:
                        for nested_name, nested_dict in child_dict.items():
                            if nested_name in timings and timings[nested_name]:
                                _print_hierarchy(nested_name, level + 2)
                                if nested_dict:
                                    for deep_name, deep_dict in nested_dict.items():
                                        if deep_name in timings and timings[deep_name]:
                                            _print_hierarchy(deep_name, level + 3)
                                            if deep_dict:
                                                for deeper_name in deep_dict:
                                                    if deeper_name in timings and timings[deeper_name]:
                                                        _print_hierarchy(deeper_name, level + 4)
    
    # Print top-level operations in hierarchy
    for top_op in OPERATION_HIERARCHY:
        _print_hierarchy(top_op)

def load_parameters(model_path):
    return np.load(model_path)
