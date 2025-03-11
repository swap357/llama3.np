import numpy as np
import time
import sys
import math
from numba import njit
from config import ModelArgs
from tokenizer import Tokenizer
from utils import load_parameters

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

@njit
def softmax_jit(x_2d):
    """Optimized 2D softmax for Numba JIT"""
    x_max = np.max(x_2d, axis=1)
    e_x = np.exp(x_2d - x_max[:, np.newaxis])
    return e_x / np.sum(e_x, axis=1)[:, np.newaxis]

def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))

@njit
def silu_jit(x):
    return x * (1.0 / (1.0 + np.exp(-x)))

def compute_cos_sin_cache(head_dim, max_seq_len, base=10000):
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2)[: (head_dim // 2)] / head_dim))
    t = np.arange(max_seq_len)
    freqs = np.outer(t, inv_freq)
    return np.cos(freqs), np.sin(freqs)

def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    # Optimized RoPE implementation using direct indexing
    xq_r, xq_i = xq[..., ::2], xq[..., 1::2]
    xk_r, xk_i = xk[..., ::2], xk[..., 1::2]
    
    # Reshape frequencies for broadcasting
    freqs_cos = np.expand_dims(freqs_cos, axis=(0, 2))
    freqs_sin = np.expand_dims(freqs_sin, axis=(0, 2))
    
    # Apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos
    
    # Interleave real and imaginary parts
    xq_out = np.zeros_like(xq)
    xk_out = np.zeros_like(xk)
    xq_out[..., ::2] = xq_out_r
    xq_out[..., 1::2] = xq_out_i
    xk_out[..., ::2] = xk_out_r
    xk_out[..., 1::2] = xk_out_i
    
    return xq_out, xk_out

class RMSNorm:
    def __init__(self, weight, eps):
        self.weight = weight
        self.eps = eps
        
    def __call__(self, x):
        variance = np.mean(x**2, axis=-1, keepdims=True)
        x_norm = x / np.sqrt(variance + self.eps)
        return x_norm * self.weight

class FeedForward:
    def __init__(self, up_weight, gate_weight, down_weight):
        self.up_weight = up_weight.T
        self.gate_weight = gate_weight.T
        self.down_weight = down_weight.T
        
    def __call__(self, x):
        swish = silu(x @ self.gate_weight)
        x_up = x @ self.up_weight
        x = swish * x_up
        x = x @ self.down_weight
        return x

class Attention:
    def __init__(self, q_weight, k_weight, v_weight, o_weight, args):
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        
        self.q_weight = q_weight.T
        self.k_weight = k_weight.T
        self.v_weight = v_weight.T
        self.o_weight = o_weight.T
        
        self.cache_k = np.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = np.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        
    def __call__(self, x, start_pos, mask, freqs_cos, freqs_sin):
        batch_size, seq_len, _ = x.shape
        
        # Project q, k, v
        xq = x @ self.q_weight
        xk = x @ self.k_weight
        xv = x @ self.v_weight
        
        # Reshape for multi-head attention
        xq = xq.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = xk.reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.reshape(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # Apply rotary positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        
        # Update KV cache
        self.cache_k[:batch_size, start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos+seq_len] = xv
        
        # Get the full k/v sequences including the cached values
        k_seq = self.cache_k[:batch_size, :start_pos+seq_len]
        v_seq = self.cache_v[:batch_size, :start_pos+seq_len]
        
        # Handle grouped-query attention if needed
        if self.n_heads > self.n_kv_heads:
            n_rep = self.n_heads // self.n_kv_heads
            k_seq = np.repeat(k_seq, n_rep, axis=2)
            v_seq = np.repeat(v_seq, n_rep, axis=2)
        
        # Reshape for attention computation
        xq = xq.transpose(0, 2, 1, 3)  # [batch, heads, seq_len, head_dim]
        k_seq = k_seq.transpose(0, 2, 1, 3)
        v_seq = v_seq.transpose(0, 2, 1, 3)
        
        # Compute attention scores efficiently using einsum
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = np.einsum('bhqd,bhkd->bhqk', xq, k_seq) * scale
        if mask is not None:
            attn_scores = attn_scores + mask[None, None, :, :]
        
        attn_weights = softmax(attn_scores)
        attn_output = np.einsum('bhqk,bhkd->bhqd', attn_weights, v_seq)
        
        # Reshape and project output
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        output = attn_output @ self.o_weight
        
        return output

class TransformerBlock:
    def __init__(self, weights, layer_id, args):
        layer_prefix = f"model.layers.{layer_id}."
        
        # Initialize attention
        self.attention = Attention(
            weights.get(f"{layer_prefix}self_attn.q_proj.weight"),
            weights.get(f"{layer_prefix}self_attn.k_proj.weight"),
            weights.get(f"{layer_prefix}self_attn.v_proj.weight"),
            weights.get(f"{layer_prefix}self_attn.o_proj.weight"),
            args
        )
        
        # Initialize feed-forward
        self.feed_forward = FeedForward(
            weights.get(f"{layer_prefix}mlp.up_proj.weight"),
            weights.get(f"{layer_prefix}mlp.gate_proj.weight"),
            weights.get(f"{layer_prefix}mlp.down_proj.weight")
        )
        
        # Initialize layer norms
        self.input_layernorm = RMSNorm(
            weights.get(f"{layer_prefix}input_layernorm.weight"),
            eps=args.norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            weights.get(f"{layer_prefix}post_attention_layernorm.weight"),
            eps=args.norm_eps
        )
    
    def __call__(self, x, start_pos, mask, freqs_cos, freqs_sin):
        # Layer norm 1
        norm_x = self.input_layernorm(x)
        
        # Self-attention
        h = self.attention(norm_x, start_pos, mask, freqs_cos, freqs_sin)
        
        # Residual connection
        x = x + h
        
        # Layer norm 2
        norm_x = self.post_attention_layernorm(x)
        
        # Feed-forward
        h = self.feed_forward(norm_x)
        
        # Residual connection
        x = x + h
        
        return x

class JitLlama:
    def __init__(self, model_path, args):
        self.args = args
        weights = load_parameters(model_path)
        
        # Extract token embeddings and head
        self.tok_embedding = weights.get("model.embed_tokens.weight")
        self.lm_head = weights.get("lm_head.weight").T
        
        # Pre-compute rotary embeddings
        self.freqs_cos, self.freqs_sin = compute_cos_sin_cache(
            args.dim // args.n_heads, 
            args.max_seq_len
        )
        
        # Initialize transformer layers
        self.layers = []
        for i in range(args.n_layers):
            self.layers.append(TransformerBlock(weights, i, args))
            
        # Final layer norm
        self.norm = RMSNorm(weights.get("model.norm.weight"), eps=args.norm_eps)
    
    def forward(self, input_ids, start_pos):
        # Ensure input_ids is a 2D array [batch_size, seq_len]
        if len(input_ids.shape) == 1:
            input_ids = input_ids.reshape(1, -1)
        batch_size, seq_len = input_ids.shape
        
        # Get token embeddings
        hidden = self.tok_embedding[input_ids]
        
        # Get rotary embedding frequencies for this segment
        freqs_cos = self.freqs_cos[start_pos:start_pos + seq_len]
        freqs_sin = self.freqs_sin[start_pos:start_pos + seq_len]
        
        # Create causal mask for attention
        mask = None
        if seq_len > 1:
            mask = np.full((seq_len, seq_len), float("-inf"))
            mask = np.triu(mask, k=1)
            mask = np.concatenate([np.zeros((seq_len, start_pos)), mask], axis=1)
        
        # Process through transformer layers
        for layer in self.layers:
            hidden = layer(hidden, start_pos, mask, freqs_cos, freqs_sin)
        
        # Final layer norm
        hidden = self.norm(hidden)
        
        # Get logits for next token prediction (only for the last position)
        logits = hidden[:, [-1], :] @ self.lm_head
        
        return logits
    
    def generate(self, input_ids, max_new_tokens):
        batch_size, seq_len = input_ids.shape
        
        for i in range(max_new_tokens):
            if i == 0:  # Prefill phase
                inputs = input_ids
                pos = 0
            else:  # Decode phase
                inputs = next_id
                pos = seq_len + i - 1
                
            logits = self.forward(inputs, pos)
            next_id = logits.argmax(-1)
            next_id = next_id.reshape(batch_size, 1)
            yield next_id

if __name__ == '__main__':
    args = ModelArgs()
    tokenizer = Tokenizer("./tokenizer.model.np")
    model = JitLlama("./stories15M.model.npz", args)
    
    prompt = "I have a dream" if len(sys.argv) == 1 else sys.argv[1]
    
    print(f"\n{prompt}", end="")
    input_ids = np.array([tokenizer.encode(prompt)])
    start = time.time()
    token_count = input_ids.shape[1]
    
    for id in model.generate(input_ids, args.max_new_tokens):
        token_count += 1
        output_id = id[0, 0].item()
        if output_id in [tokenizer.eos_id, tokenizer.bos_id]:
            break
        print(tokenizer.decode([output_id]), end="")
        sys.stdout.flush()
        
    elapsed = time.time() - start
    print(f"\n\nToken count: {token_count}, elapsed: {elapsed:.2f}s, {round(token_count / elapsed)} tokens/s")