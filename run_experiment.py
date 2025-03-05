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

class ExperimentLogger:
    def __init__(self, log_dir: str = "experiment_results"):
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
    
    def _parse_timing_stats(self, output: str) -> Dict[str, Dict[str, float]]:
        """Parse timing statistics from the model output."""
        stats = {}
        timing_pattern = r"(\w+(?:\.\w+)*) took ([\d.]+)ms"
        
        for match in re.finditer(timing_pattern, output):
            op_name = match.group(1)
            time_ms = float(match.group(2))
            
            if op_name not in stats:
                stats[op_name] = {
                    "total_ms": 0.0,
                    "num_calls": 0,
                    "min_ms": float('inf'),
                    "max_ms": 0.0
                }
            
            stats[op_name]["total_ms"] += time_ms
            stats[op_name]["num_calls"] += 1
            stats[op_name]["min_ms"] = min(stats[op_name]["min_ms"], time_ms)
            stats[op_name]["max_ms"] = max(stats[op_name]["max_ms"], time_ms)
        
        # Calculate averages
        for op_stats in stats.values():
            op_stats["avg_ms"] = op_stats["total_ms"] / op_stats["num_calls"]
        
        return stats
    
    def _parse_performance_metrics(self, output: str) -> Dict[str, Any]:
        """Parse performance metrics from the model output."""
        metrics = {}
        
        # Parse tokens and time
        perf_pattern = r"Performance: (\d+) tokens in ([\d.]+)s \((\d+) tokens/s\)"
        match = re.search(perf_pattern, output)
        if match:
            metrics["total_tokens"] = int(match.group(1))
            metrics["elapsed_time"] = float(match.group(2))
            metrics["tokens_per_second"] = int(match.group(3))
        
        return metrics
    
    def _parse_model_output(self, output: str) -> Dict[str, str]:
        """Parse the model's generated text output."""
        # Split output into lines
        lines = output.split('\n')
        
        # Find the start of the model output (after the prompt)
        start_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("Input prompt:"):
                start_idx = i + 1
                break
        
        # Find the end of the model output (before performance stats)
        end_idx = len(lines)
        for i, line in enumerate(lines):
            if line.startswith("Performance:"):
                end_idx = i
                break
        
        # Extract the model output
        model_output = '\n'.join(lines[start_idx:end_idx]).strip()
        
        return {
            "model_output": model_output,
            "performance_stats": '\n'.join(lines[end_idx:]).strip()
        }
    
    def log_experiment(self, 
                      command: str,
                      output: str,
                      model_args: Dict[str, Any]) -> tuple[str, str]:
        """Log experiment results with detailed information."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Parse metrics and stats
        metrics = self._parse_performance_metrics(output)
        timing_stats = self._parse_timing_stats(output)
        model_output = self._parse_model_output(output)
        
        # Prepare experiment data
        experiment_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "system_info": self._get_system_info(),
                "command": command,
            },
            "model_args": model_args,
            "results": {
                **metrics,
                "model_output": model_output["model_output"],
                "performance_stats": model_output["performance_stats"],
                "timing_stats": timing_stats,
            }
        }
        
        # Save JSON results
        json_filename = f"llama_run_{timestamp}.json"
        json_path = os.path.join(self.log_dir, json_filename)
        with open(json_path, 'w') as f:
            json.dump(experiment_data, f, indent=2)
        
        # Create human-readable summary
        summary_filename = f"llama_run_{timestamp}_summary.txt"
        summary_path = os.path.join(self.log_dir, summary_filename)
        with open(summary_path, 'w') as f:
            f.write(f"LLaMA Model Run Summary\n")
            f.write(f"======================\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Command: {command}\n\n")
            
            f.write(f"System Information:\n")
            f.write(f"------------------\n")
            for key, value in self._get_system_info().items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write(f"Model Arguments:\n")
            f.write(f"---------------\n")
            for key, value in model_args.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write(f"Model Output:\n")
            f.write(f"------------\n")
            f.write(model_output["model_output"])
            f.write("\n\n")
            
            f.write(f"Performance Metrics:\n")
            f.write(f"------------------\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write(f"Performance Statistics:\n")
            f.write(f"--------------------\n")
            f.write(model_output["performance_stats"])
            f.write("\n\n")

        
        return json_path, summary_path

def run_experiment(batch_size: int, prompt: str):
    """Run the LLaMA model and log the experiment results."""
    # Construct command
    command = f"python llama3.py --batch-size {batch_size} '{prompt}'"
    
    # Run the model and capture output
    start_time = time.time()
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    end_time = time.time()
    
    # Get model arguments from config
    from config import ModelArgs
    model_args = ModelArgs()
    model_args.max_batch_size = batch_size
    
    # Log experiment results
    logger = ExperimentLogger()
    json_path, summary_path = logger.log_experiment(
        command=command,
        output=result.stdout,
        model_args=model_args.__dict__
    )
    
    # Print results
    print("\nModel Output:")
    print(result.stdout)
    print("\nExperiment results saved to:")
    print(f"JSON: {json_path}")
    print(f"Summary: {summary_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run LLaMA model experiment with logging')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('prompt', nargs='?', default="once upon a time", help='Input prompt')
    args = parser.parse_args()
    
    run_experiment(args.batch_size, args.prompt) 