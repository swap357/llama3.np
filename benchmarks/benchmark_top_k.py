#!/usr/bin/env python
"""
Simple benchmark for top_k_logits implementations (NumPy vs Numba)
"""
import time
import numpy as np
from numba import njit

# Direct implementations of top_k_logits functions

# NumPy implementation
def top_k_numpy(nxt_logits, k):
    _bs, vs = nxt_logits.shape
    assert k < vs
    idxes = nxt_logits.argpartition(-k, axis=-1)[:,[-k]]
    k_vals = np.take_along_axis(nxt_logits, idxes, axis=1)
    scores = np.where(nxt_logits < k_vals, -np.inf, nxt_logits)
    return scores

# Numba implementation - basic version
@njit(cache=True)
def top_k_numba(nxt_logits, k):
    bs, vs = nxt_logits.shape
    assert k < vs
    
    # Create output array
    scores = np.empty_like(nxt_logits)
    
    for b in range(bs):
        # Find the k-th largest value
        logits_row = nxt_logits[b]
        k_val = np.partition(logits_row, -k)[-k]
        
        # Set values less than k_val to -inf
        for v in range(vs):
            if logits_row[v] < k_val:
                scores[b, v] = -np.inf
            else:
                scores[b, v] = logits_row[v]
    
    return scores

# Numba implementation - optimized version
@njit(cache=True, fastmath=True)
def top_k_numba_optimized(nxt_logits, k):
    bs, vs = nxt_logits.shape
    assert k < vs
    
    # Create output array
    scores = np.empty_like(nxt_logits)
    
    for b in range(bs):
        # Find the k-th largest value
        logits_row = nxt_logits[b]
        k_val = np.partition(logits_row, -k)[-k]
        
        # Vectorized approach - this works better within Numba
        for v in range(vs):
            scores[b, v] = logits_row[v] if logits_row[v] >= k_val else -np.inf
    
    return scores

# Try to import parallel module from numba, but don't fail if it's not available
try:
    from numba import prange
    
    # Parallel implementation
    @njit(cache=True, fastmath=True, parallel=True)
    def top_k_numba_parallel(nxt_logits, k):
        bs, vs = nxt_logits.shape
        assert k < vs
        
        # Create output array
        scores = np.empty_like(nxt_logits)
        
        # Use prange for parallelization across batches
        for b in prange(bs):
            logits_row = nxt_logits[b]
            k_val = np.partition(logits_row, -k)[-k]
            
            for v in range(vs):
                scores[b, v] = logits_row[v] if logits_row[v] >= k_val else -np.inf
        
        return scores
    
    HAS_PARALLEL = True
except ImportError:
    top_k_numba_parallel = None
    HAS_PARALLEL = False

def benchmark(batch_size=4, vocab_size=32000, k=50, iterations=100, warmup_iterations=20, dtype=np.float32):
    """Run benchmark comparing NumPy and Numba implementations of top_k_logits"""
    print(f"Benchmarking top_k_logits with shape=({batch_size}, {vocab_size}), k={k}, iterations={iterations}")
    
    # Create random input data
    logits = np.random.normal(0, 1, size=(batch_size, vocab_size)).astype(dtype)
    
    # Create result dictionary to store timings
    results = {}
    
    # Benchmark NumPy implementation
    print("\nBenchmarking NumPy implementation...")
    # Warmup
    for _ in range(5):
        top_k_numpy(logits, k)
    
    # Actual benchmark
    start_time = time.time()
    for _ in range(iterations):
        top_k_numpy(logits, k)
    numpy_time = time.time() - start_time
    numpy_avg_ms = (numpy_time / iterations) * 1000
    results["numpy"] = numpy_avg_ms
    print(f"NumPy: {numpy_avg_ms:.4f} ms per call")
    
    # Wait a moment between benchmarks
    time.sleep(0.5)
    
    # First let's test that the implementations work correctly
    print("\nVerifying implementations...")
    numpy_result = top_k_numpy(logits, k)
    numba_result = top_k_numba(logits, k)
    numba_opt_result = top_k_numba_optimized(logits, k)
    
    # Check basic numba implementation
    if np.allclose(numpy_result, numba_result, equal_nan=True):
        print("✅ Basic Numba implementation produces correct results")
    else:
        print("❌ Basic Numba implementation produces incorrect results")
        
    # Check optimized numba implementation
    if np.allclose(numpy_result, numba_opt_result, equal_nan=True):
        print("✅ Optimized Numba implementation produces correct results")
    else:
        print("❌ Optimized Numba implementation produces incorrect results")
    
    # Check parallel implementation if available
    if HAS_PARALLEL:
        numba_parallel_result = top_k_numba_parallel(logits, k)
        if np.allclose(numpy_result, numba_parallel_result, equal_nan=True):
            print("✅ Parallel Numba implementation produces correct results")
        else:
            print("❌ Parallel Numba implementation produces incorrect results")
    
    # Benchmark basic numba implementation
    print("\nBenchmarking basic Numba implementation...")
    # Compilation (first call)
    compilation_start = time.time()
    _ = top_k_numba(logits, k)
    compilation_time = time.time() - compilation_start
    print(f"Compilation took {compilation_time*1000:.2f} ms")
    
    # Warmup
    for _ in range(warmup_iterations):
        top_k_numba(logits, k)
    
    # Actual benchmark
    start_time = time.time()
    for _ in range(iterations):
        top_k_numba(logits, k)
    numba_time = time.time() - start_time
    numba_avg_ms = (numba_time / iterations) * 1000
    results["numba_basic"] = numba_avg_ms
    print(f"Basic Numba: {numba_avg_ms:.4f} ms per call")
    
    # Wait between benchmarks
    time.sleep(0.5)
    
    # Benchmark optimized numba implementation
    print("\nBenchmarking optimized Numba implementation...")
    # Compilation
    _ = top_k_numba_optimized(logits, k)
    
    # Warmup
    for _ in range(warmup_iterations):
        top_k_numba_optimized(logits, k)
        
    # Actual benchmark
    start_time = time.time()
    for _ in range(iterations):
        top_k_numba_optimized(logits, k)
    numba_opt_time = time.time() - start_time
    numba_opt_avg_ms = (numba_opt_time / iterations) * 1000
    results["numba_optimized"] = numba_opt_avg_ms
    print(f"Optimized Numba: {numba_opt_avg_ms:.4f} ms per call")
    
    # Benchmark parallel numba implementation if available
    if HAS_PARALLEL:
        print("\nBenchmarking parallel Numba implementation...")
        # Compilation
        _ = top_k_numba_parallel(logits, k)
        
        # Warmup
        for _ in range(warmup_iterations):
            top_k_numba_parallel(logits, k)
            
        # Actual benchmark
        start_time = time.time()
        for _ in range(iterations):
            top_k_numba_parallel(logits, k)
        numba_parallel_time = time.time() - start_time
        numba_parallel_avg_ms = (numba_parallel_time / iterations) * 1000
        results["numba_parallel"] = numba_parallel_avg_ms
        print(f"Parallel Numba: {numba_parallel_avg_ms:.4f} ms per call")
    
    # Report results and speedups
    print("\n===== RESULTS SUMMARY =====")
    print(f"NumPy: {results['numpy']:.4f} ms per call")
    print(f"Basic Numba: {results['numba_basic']:.4f} ms per call (speedup: {results['numpy']/results['numba_basic']:.2f}x)")
    print(f"Optimized Numba: {results['numba_optimized']:.4f} ms per call (speedup: {results['numpy']/results['numba_optimized']:.2f}x)")
    
    if HAS_PARALLEL:
        print(f"Parallel Numba: {results['numba_parallel']:.4f} ms per call (speedup: {results['numpy']/results['numba_parallel']:.2f}x)")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark NumPy vs Numba top_k_logits implementations")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--vocab", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--k", type=int, default=50, help="k value for top-k")
    parser.add_argument("--iter", type=int, default=100, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=20, help="Number of warmup iterations")
    
    args = parser.parse_args()
    
    # Run the benchmark with command-line arguments
    benchmark(args.batch, args.vocab, args.k, args.iter, args.warmup) 