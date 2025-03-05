#!/usr/bin/env python
"""
Simple benchmark for softmax implementations (NumPy vs Numba)
"""
import time
import numpy as np
from numba import njit

# Direct implementations of softmax functions

# NumPy implementation
def softmax_numpy(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Numba implementation - basic version
@njit(cache=True)
def softmax_numba(x):
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    result = np.empty_like(x_2d)
    
    for i in range(x_2d.shape[0]):
        row_max = x_2d[i].max()
        num = np.exp(x_2d[i] - row_max)
        den = num.sum()    
        result[i] = num / den
    
    return result.reshape(orig_shape)

# Numba implementation - optimized with fastmath
@njit(cache=True, fastmath=True)
def softmax_numba_fastmath(x):
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    result = np.empty_like(x_2d)
    
    for i in range(x_2d.shape[0]):
        row = x_2d[i]
        row_max = row.max()
        exp_vals = np.exp(row - row_max)
        den = exp_vals.sum()
        result[i] = exp_vals / den
    
    return result.reshape(orig_shape)

# Try to import parallel module from numba, but don't fail if it's not available
try:
    from numba import prange
    
    # Parallel implementation
    @njit(cache=True, fastmath=True, parallel=True)
    def softmax_numba_parallel(x):
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        result = np.empty_like(x_2d)
        
        # Use prange for parallelization
        for i in prange(x_2d.shape[0]):
            row = x_2d[i]
            row_max = row.max()
            exp_vals = np.exp(row - row_max)
            den = exp_vals.sum()
            result[i] = exp_vals / den
        
        return result.reshape(orig_shape)
    
    HAS_PARALLEL = True
except ImportError:
    softmax_numba_parallel = None
    HAS_PARALLEL = False

def benchmark(batch_size=4, vocab_size=32000, iterations=100, warmup_iterations=20, dtype=np.float32):
    """Run benchmark comparing NumPy and Numba implementations of softmax"""
    print(f"Benchmarking softmax with shape=({batch_size}, {vocab_size}), iterations={iterations}")
    
    # Create random input data
    x = np.random.normal(0, 1, size=(batch_size, vocab_size)).astype(dtype)
    
    # Create result dictionary to store timings
    results = {}
    
    # Benchmark NumPy implementation
    print("\nBenchmarking NumPy implementation...")
    # Warmup
    for _ in range(5):
        softmax_numpy(x)
    
    # Actual benchmark
    start_time = time.time()
    for _ in range(iterations):
        softmax_numpy(x)
    numpy_time = time.time() - start_time
    numpy_avg_ms = (numpy_time / iterations) * 1000
    results["numpy"] = numpy_avg_ms
    print(f"NumPy: {numpy_avg_ms:.4f} ms per call")
    
    # Wait a moment between benchmarks
    time.sleep(0.5)
    
    # First let's test that the implementations work correctly
    print("\nVerifying implementations...")
    numpy_result = softmax_numpy(x)
    numba_result = softmax_numba(x)
    numba_fastmath_result = softmax_numba_fastmath(x)
    
    # Check basic numba implementation
    if np.allclose(numpy_result, numba_result):
        print("✅ Basic Numba implementation produces correct results")
    else:
        print("❌ Basic Numba implementation produces incorrect results")
        
    # Check fastmath numba implementation
    if np.allclose(numpy_result, numba_fastmath_result):
        print("✅ FastMath Numba implementation produces correct results")
    else:
        print("❌ FastMath Numba implementation produces incorrect results")
    
    # Check parallel implementation if available
    if HAS_PARALLEL:
        numba_parallel_result = softmax_numba_parallel(x)
        if np.allclose(numpy_result, numba_parallel_result):
            print("✅ Parallel Numba implementation produces correct results")
        else:
            print("❌ Parallel Numba implementation produces incorrect results")
    
    # Benchmark basic numba implementation
    print("\nBenchmarking basic Numba implementation...")
    # Compilation (first call)
    compilation_start = time.time()
    _ = softmax_numba(x)
    compilation_time = time.time() - compilation_start
    print(f"Compilation took {compilation_time*1000:.2f} ms")
    
    # Warmup
    for _ in range(warmup_iterations):
        softmax_numba(x)
    
    # Actual benchmark
    start_time = time.time()
    for _ in range(iterations):
        softmax_numba(x)
    numba_time = time.time() - start_time
    numba_avg_ms = (numba_time / iterations) * 1000
    results["numba_basic"] = numba_avg_ms
    print(f"Basic Numba: {numba_avg_ms:.4f} ms per call")
    
    # Wait between benchmarks
    time.sleep(0.5)
    
    # Benchmark fastmath numba implementation
    print("\nBenchmarking FastMath Numba implementation...")
    # Compilation
    _ = softmax_numba_fastmath(x)
    
    # Warmup
    for _ in range(warmup_iterations):
        softmax_numba_fastmath(x)
        
    # Actual benchmark
    start_time = time.time()
    for _ in range(iterations):
        softmax_numba_fastmath(x)
    numba_fastmath_time = time.time() - start_time
    numba_fastmath_avg_ms = (numba_fastmath_time / iterations) * 1000
    results["numba_fastmath"] = numba_fastmath_avg_ms
    print(f"FastMath Numba: {numba_fastmath_avg_ms:.4f} ms per call")
    
    # Benchmark parallel numba implementation if available
    if HAS_PARALLEL:
        print("\nBenchmarking parallel Numba implementation...")
        # Compilation
        _ = softmax_numba_parallel(x)
        
        # Warmup
        for _ in range(warmup_iterations):
            softmax_numba_parallel(x)
            
        # Actual benchmark
        start_time = time.time()
        for _ in range(iterations):
            softmax_numba_parallel(x)
        numba_parallel_time = time.time() - start_time
        numba_parallel_avg_ms = (numba_parallel_time / iterations) * 1000
        results["numba_parallel"] = numba_parallel_avg_ms
        print(f"Parallel Numba: {numba_parallel_avg_ms:.4f} ms per call")
    
    # Report results and speedups
    print("\n===== RESULTS SUMMARY =====")
    print(f"NumPy: {results['numpy']:.4f} ms per call")
    print(f"Basic Numba: {results['numba_basic']:.4f} ms per call (speedup: {results['numpy']/results['numba_basic']:.2f}x)")
    print(f"FastMath Numba: {results['numba_fastmath']:.4f} ms per call (speedup: {results['numpy']/results['numba_fastmath']:.2f}x)")
    
    if HAS_PARALLEL:
        print(f"Parallel Numba: {results['numba_parallel']:.4f} ms per call (speedup: {results['numpy']/results['numba_parallel']:.2f}x)")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark NumPy vs Numba softmax implementations")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--vocab", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--iter", type=int, default=100, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=20, help="Number of warmup iterations")
    
    args = parser.parse_args()
    
    # Run the benchmark with command-line arguments
    benchmark(args.batch, args.vocab, args.iter, args.warmup) 