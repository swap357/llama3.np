import numpy as np
import time
import sys
import psutil
import os
from llama3 import Llama
from jit_llama3 import JitLlama
from config import ModelArgs
from tokenizer import Tokenizer

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def main():
    # Initialize models and tokenizer
    args = ModelArgs()
    tokenizer = Tokenizer("./tokenizer.model.np")
    
    # Memory usage before model initialization
    mem_before = get_memory_usage()
    
    # Load llama3 model and measure memory usage
    llama_model = Llama("./stories15M.model.npz", args)
    mem_after_llama = get_memory_usage()
    
    # Load JIT model and measure memory usage
    jit_model = JitLlama("./stories15M.model.npz", args)
    mem_after_jit = get_memory_usage()
    
    print("==== MEMORY USAGE ====")
    print(f"Base memory usage: {mem_before:.2f} MB")
    print(f"llama3.py memory: {mem_after_llama - mem_before:.2f} MB")
    print(f"Additional memory for jit_llama3.py: {mem_after_jit - mem_after_llama:.2f} MB")
    print(f"Total memory: {mem_after_jit:.2f} MB\n")
    
    # Test prompt
    prompt = "Hello world"
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    
    # Encode prompt
    input_ids = np.array([tokenizer.encode(prompt)])
    print(f"Prompt: \"{prompt}\"")
    print(f"Input shape: {input_ids.shape}\n")
    
    # Compare first token generation
    print("==== SINGLE TOKEN GENERATION COMPARISON ====")
    
    # Time llama3.py
    start = time.time()
    llama_logits = llama_model(input_ids, 0)
    llama_time = time.time() - start
    llama_token = llama_logits.argmax(-1)[0, 0].item()
    print(f"llama3.py token: {tokenizer.decode([llama_token])}")
    print(f"Time: {llama_time*1000:.2f}ms")
    
    # Time jit_llama3.py
    start = time.time()
    jit_logits = jit_model.forward(input_ids, 0)
    jit_time = time.time() - start
    jit_token = jit_logits.argmax(-1)[0, 0].item()
    print(f"jit_llama3.py token: {tokenizer.decode([jit_token])}")
    print(f"Time: {jit_time*1000:.2f}ms")
    
    print(f"Speedup: {llama_time/jit_time:.2f}x")
    
    # Compare multiple token generation
    print("\n==== MULTI-TOKEN GENERATION COMPARISON ====")
    max_tokens = 50
    
    # Time llama3.py
    start = time.time()
    llama_tokens = []
    token_count = input_ids.shape[1]
    pos = 0
    
    for i, curr_pos in enumerate(range(token_count, token_count + max_tokens)):
        if i == 0:  # Prefill Phase
            inputs = input_ids
            pos = 0
        else:  # Decode Phase
            inputs = np.array([[next_token]])
            pos = curr_pos - 1
            
        logits = llama_model(inputs, pos)
        next_token = logits.argmax(-1)[0, 0].item()
        llama_tokens.append(next_token)
        if next_token in [tokenizer.eos_id, tokenizer.bos_id]:
            break
    
    llama_time = time.time() - start
    llama_tps = len(llama_tokens) / llama_time
    
    # Time jit_llama3.py
    start = time.time()
    jit_tokens = []
    token_count = input_ids.shape[1]
    
    for i, token_id in enumerate(jit_model.generate(input_ids, max_tokens)):
        output_id = token_id[0, 0].item()
        jit_tokens.append(output_id)
        if output_id in [tokenizer.eos_id, tokenizer.bos_id] or i >= max_tokens:
            break
    
    jit_time = time.time() - start
    jit_tps = len(jit_tokens) / jit_time
    
    # Print results
    print(f"llama3.py: {len(llama_tokens)} tokens in {llama_time:.2f}s ({llama_tps:.2f} tokens/s)")
    print(f"jit_llama3.py: {len(jit_tokens)} tokens in {jit_time:.2f}s ({jit_tps:.2f} tokens/s)")
    print(f"Speedup: {jit_tps/llama_tps:.2f}x")
    
    # Check if tokens match
    match_count = sum(1 for i in range(min(len(llama_tokens), len(jit_tokens))) if llama_tokens[i] == jit_tokens[i])
    total = min(len(llama_tokens), len(jit_tokens))
    print(f"\nToken match: {match_count}/{total} ({match_count/total*100:.1f}%)")

if __name__ == "__main__":
    main()