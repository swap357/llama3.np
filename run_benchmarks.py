import subprocess
import json
import os
import time
from datetime import datetime
import sys
import platform
import numpy as np
from typing import Dict, Any
import re

class BenchmarkLogger:
    def __init__(self, log_dir: str = "benchmark_results"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
    
    def _get_system_info(self) -> Dict[str, str]:
        return {
            "os": platform.system(),
            "os_version": platform.version(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "numpy_version": np.__version__,
        }
    
    def _parse_benchmark_output(self, output: str) -> Dict[str, Any]:
        """Parse benchmark output to extract results and verification status."""
        results = {
            "verification": {},
            "timings": {},
            "speedups": {}
        }
        
        # Parse verification results
        verification_pattern = r"([✅❌]) (.*?) implementation produces correct results"
        for match in re.finditer(verification_pattern, output):
            status, impl = match.groups()
            results["verification"][impl] = status == "✅"
        
        # Parse timing results
        timing_pattern = r"([\w\s]+): ([\d.]+) ms per call"
        for match in re.finditer(timing_pattern, output):
            impl, time_ms = match.groups()
            results["timings"][impl.strip()] = float(time_ms)
        
        # Parse speedup results
        speedup_pattern = r"speedup: ([\d.]+)x"
        for match in re.finditer(speedup_pattern, output):
            impl = match.group(0).split(":")[0].strip()
            speedup = float(match.group(1))
            results["speedups"][impl] = speedup
        
        return results
    
    def log_benchmark(self, 
                     benchmark_name: str,
                     command: str,
                     output: str,
                     params: Dict[str, Any]) -> tuple[str, str]:
        """Log benchmark results with detailed information."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Parse benchmark results
        results = self._parse_benchmark_output(output)
        
        # Prepare benchmark data
        benchmark_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "system_info": self._get_system_info(),
                "command": command,
            },
            "parameters": params,
            "results": results
        }
        
        # Save JSON results
        json_filename = f"{benchmark_name}_{timestamp}.json"
        json_path = os.path.join(self.log_dir, json_filename)
        with open(json_path, 'w') as f:
            json.dump(benchmark_data, f, indent=2)
        
        # Create human-readable summary
        summary_filename = f"{benchmark_name}_{timestamp}_summary.txt"
        summary_path = os.path.join(self.log_dir, summary_filename)
        with open(summary_path, 'w') as f:
            f.write(f"{benchmark_name} Benchmark Summary\n")
            f.write(f"========================\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Command: {command}\n\n")
            
            f.write(f"System Information:\n")
            f.write(f"------------------\n")
            for key, value in self._get_system_info().items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write(f"Benchmark Parameters:\n")
            f.write(f"-------------------\n")
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write(f"Implementation Verification:\n")
            f.write(f"-------------------------\n")
            for impl, status in results["verification"].items():
                f.write(f"{impl}: {'✅ Correct' if status else '❌ Incorrect'}\n")
            f.write("\n")
            
            f.write(f"Performance Results:\n")
            f.write(f"------------------\n")
            for impl, time_ms in results["timings"].items():
                f.write(f"{impl}: {time_ms:.4f} ms per call\n")
            f.write("\n")
            
            f.write(f"Speedup Results:\n")
            f.write(f"--------------\n")
            for impl, speedup in results["speedups"].items():
                f.write(f"{impl}: {speedup:.2f}x\n")
            f.write("\n")
            
            f.write(f"Raw Output:\n")
            f.write(f"----------\n")
            f.write(output)
        
        return json_path, summary_path

def run_benchmark(benchmark_script: str, params: Dict[str, Any]):
    """Run a benchmark script and log the results."""
    # Construct command
    cmd_parts = ["python", benchmark_script]
    for key, value in params.items():
        cmd_parts.append(f"--{key}")
        cmd_parts.append(str(value))
    command = " ".join(cmd_parts)
    
    # Run the benchmark and capture output
    start_time = time.time()
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    end_time = time.time()
    
    # Log benchmark results
    logger = BenchmarkLogger()
    benchmark_name = os.path.splitext(os.path.basename(benchmark_script))[0]
    json_path, summary_path = logger.log_benchmark(
        benchmark_name=benchmark_name,
        command=command,
        output=result.stdout,
        params=params
    )
    
    # Print results
    print("\nBenchmark Output:")
    print(result.stdout)
    print("\nBenchmark results saved to:")
    print(f"JSON: {json_path}")
    print(f"Summary: {summary_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run and document benchmark results')
    parser.add_argument('--batch', type=int, default=4, help='Batch size')
    parser.add_argument('--vocab', type=int, default=32000, help='Vocabulary size')
    parser.add_argument('--k', type=int, default=50, help='k value for top-k (only for top_k benchmark)')
    parser.add_argument('--iter', type=int, default=100, help='Number of iterations')
    parser.add_argument('--warmup', type=int, default=20, help='Number of warmup iterations')
    args = parser.parse_args()
    
    # Common parameters for both benchmarks
    common_params = {
        "batch": args.batch,
        "vocab": args.vocab,
        "iter": args.iter,
        "warmup": args.warmup
    }
    
    # Run softmax benchmark
    print("\nRunning softmax benchmark...")
    run_benchmark("benchmarks/benchmark_softmax.py", common_params)
    
    # Run top_k benchmark
    print("\nRunning top_k benchmark...")
    top_k_params = {**common_params, "k": args.k}
    run_benchmark("benchmarks/benchmark_top_k.py", top_k_params) 