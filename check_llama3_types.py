import numpy as np
import time
from llama3 import Llama
from config import ModelArgs

def inspect_llama_types():
    args = ModelArgs()
    model = Llama("./stories15M.model.npz", args)

    print("==== LLAMA3 DATA TYPE INSPECTION ====")
    print(f"Token embedding dtype: {model.tok_embedding.dtype}")
    print(f"LM head dtype: {model.lm_head_weight.dtype}")
    print(f"Freq cos dtype: {model.freqs_cos.dtype}")

    # Check first layer parameters
    layer0 = model.layers[0]
    print("\nLayer 0 parameters:")
    print(f"Attention q_weight dtype: {layer0.attention.q_weight.dtype}")
    print(f"Attention k_weight dtype: {layer0.attention.k_weight.dtype}")
    print(f"Feed-forward weights dtype: {layer0.feed_forward.up_weight.dtype}")

def run_speed_comparison():
    print("\n==== SPEED COMPARISON ====")
    args = ModelArgs()
    model = Llama("./stories15M.model.npz", args)
    
    # Prepare input
    from tokenizer import Tokenizer
    tokenizer = Tokenizer("./tokenizer.model.np")
    prompt = "Hello world"
    input_ids = np.array([tokenizer.encode(prompt)])
    
    # Run timing test
    max_tokens = 50
    start = time.time()
    token_count = input_ids.shape[1]
    
    for _ in range(max_tokens):
        _ = model(input_ids, 0)
        token_count += 1
    
    elapsed = time.time() - start
    tokens_per_sec = max_tokens / elapsed
    print(f"Llama3: Generated {max_tokens} tokens in {elapsed:.2f}s")
    print(f"Speed: {tokens_per_sec:.2f} tokens/s")

if __name__ == "__main__":
    inspect_llama_types()
    run_speed_comparison()