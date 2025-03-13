#!/usr/bin/env python
"""
CLI script for running the Llama3 model.

This script allows running either the base or optimized model
for comparison and interactive use.
"""

import time
import argparse
import sys
import numpy as np

from llama3np.utils.config import ModelArgs
from llama3np.utils.tokenizer import Tokenizer
from llama3np.utils.optimized_tokenizer import OptimizedTokenizer
from llama3np.model.base import Llama as BaseLlama
from llama3np.model.optimized import Llama as OptimizedLlama


def generate_text(model, tokenizer, prompt, max_tokens=100, stream=True):
    """
    Generate text using the model.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer to use
        prompt: Input prompt
        max_tokens: Maximum tokens to generate
        stream: Whether to stream output tokens
        
    Returns:
        Generated text
    """
    # Tokenize input
    input_ids = np.array([tokenizer.encode(prompt)])
    
    # Print prompt if streaming
    if stream:
        print(prompt, end="", flush=True)
    
    # Generate tokens
    output_ids = []
    start_time = time.time()
    _, L = input_ids.shape
    for i, next_id in enumerate(model.generate(input_ids, max_tokens)):
        token_id = next_id[0, 0]
        output_ids.append(token_id)
        
        # Stop on special tokens
        if token_id in [tokenizer.eos_id, tokenizer.bos_id]:
            break
        
        # Print token if streaming
        if stream:
            token_text = tokenizer.decode([token_id])
            print(token_text, end="", flush=True)
    
    # Calculate stats
    elapsed = time.time() - start_time
    tokens_per_second = len(output_ids) / elapsed if elapsed > 0 else 0
    
    # Print stats
    if stream:
        print(f"\n\nGenerated {len(output_ids)} tokens in {elapsed:.2f}s "
              f"({tokens_per_second:.2f} tokens/s)")
    
    # Get full output text
    output_text = tokenizer.decode(output_ids)
    
    return {
        "prompt": prompt,
        "output": output_text,
        "tokens_generated": len(output_ids),
        "time_seconds": elapsed,
        "tokens_per_second": tokens_per_second
    }


def main():
    """Parse arguments and run text generation."""
    parser = argparse.ArgumentParser(description="Run Llama3 model for text generation")
    parser.add_argument("--prompt", type=str, default="Once upon a time", 
                        help="Input prompt for generation")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--optimized", action="store_true",
                        help="Use optimized model implementation")
    parser.add_argument("--no-stream", action="store_true",
                        help="Don't stream output tokens")
    args = parser.parse_args()
    
    # Configure model
    model_args = ModelArgs()
    model_args.max_new_tokens = args.max_tokens
    
    # Initialize model and tokenizer
    if args.optimized:
        print("Using optimized implementation")
        tokenizer = OptimizedTokenizer("./tokenizer.model.np")
        model = OptimizedLlama("./stories15M.model.npz", model_args)
    else:
        print("Using base implementation")
        tokenizer = Tokenizer("./tokenizer.model.np")
        model = BaseLlama("./stories15M.model.npz", model_args)
    
    # Generate text
    result = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        stream=not args.no_stream
    )
    
    # If not streaming, print the full output
    if args.no_stream:
        print(f"\nPrompt: {result['prompt']}")
        print(f"\nOutput: {result['output']}")
        print(f"\nGenerated {result['tokens_generated']} tokens in {result['time_seconds']:.2f}s "
              f"({result['tokens_per_second']:.2f} tokens/s)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())