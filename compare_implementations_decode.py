import numpy as np
import time
import sys
import tracemalloc
from llama3 import Llama
from jit_llama3 import JitLlama
from config import ModelArgs
from tokenizer import Tokenizer

def main():
    # Initialize models and tokenizer
    args = ModelArgs()
    tokenizer = Tokenizer("./tokenizer.model.np")
    llama_model = Llama("./stories15M.model.npz", args)
    jit_model = JitLlama("./stories15M.model.npz", args)
    
    # Test prompt
    prompt = "Hello world"
    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    
    # Encode prompt
    input_ids = np.array([tokenizer.encode(prompt)])
    print(f"Prompt: \"{prompt}\"")
    print(f"Input shape: {input_ids.shape}\n")
    
    # Generate first token to set up KV cache
    print("Prefill phase: processing initial prompt...")
    
    # llama3.py
    llama_logits = llama_model(input_ids, 0)
    llama_next_id = llama_logits.argmax(-1)[0, 0]
    print(f"First token: {tokenizer.decode([llama_next_id])}")
    
    # jit_llama3.py
    jit_logits = jit_model.forward(input_ids, 0)
    jit_next_id = jit_logits.argmax(-1)[0, 0]
    
    # Set up for decode phase
    print("\n==== DECODE PHASE ====")
    print("Measuring time to generate 50 tokens one by one")
    
    # Time llama3.py decode phase
    print("\nllama3.py decode performance:")
    llama_tokens = [llama_next_id]
    token_count = input_ids.shape[1]
    
    tracemalloc.start()
    start = time.time()
    curr_token = np.array([[llama_next_id]])
    
    for i in range(49):  # Generate 49 more tokens
        pos = token_count + i
        logits = llama_model(curr_token, pos)
        next_token = logits.argmax(-1)[0, 0].item()
        llama_tokens.append(next_token)
        curr_token = np.array([[next_token]])
        
        # Print progress
        if i % 10 == 0:
            print(f"Generated {i+1} tokens...")
    
    llama_time = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    llama_tps = 49 / llama_time
    llama_memory = current / (1024 * 1024)  # MB
    
    print(f"llama3.py: Generated 49 tokens in {llama_time:.2f}s ({llama_tps:.2f} tokens/s)")
    print(f"Memory usage: {llama_memory:.2f} MB")
    
    # Time jit_llama3.py decode phase
    print("\njit_llama3.py decode performance:")
    jit_tokens = [jit_next_id]
    token_count = input_ids.shape[1]
    
    tracemalloc.start()
    start = time.time()
    curr_token = np.array([[jit_next_id]])
    
    for i in range(49):  # Generate 49 more tokens
        pos = token_count + i
        logits = jit_model.forward(curr_token, pos)
        next_token = logits.argmax(-1)[0, 0].item()
        jit_tokens.append(next_token)
        curr_token = np.array([[next_token]])
        
        # Print progress
        if i % 10 == 0:
            print(f"Generated {i+1} tokens...")
    
    jit_time = time.time() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    jit_tps = 49 / jit_time
    jit_memory = current / (1024 * 1024)  # MB
    
    print(f"jit_llama3.py: Generated 49 tokens in {jit_time:.2f}s ({jit_tps:.2f} tokens/s)")
    print(f"Memory usage: {jit_memory:.2f} MB")
    
    # Print speedup
    print(f"\nDecode phase speedup: {jit_tps/llama_tps:.2f}x")
    
    # Print generated text
    print("\nllama3.py generated:")
    print(tokenizer.decode(llama_tokens))
    
    print("\njit_llama3.py generated:")
    print(tokenizer.decode(jit_tokens))

if __name__ == "__main__":
    main()