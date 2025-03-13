"""
Optimized implementation of Llama3 model using NumPy.

This module contains optimized implementations of key components:
1. Optimized RoPE using direct indexing
2. Code structure optimizations
"""

from __future__ import annotations

import math
import time
from typing import TypeVar, Generic, Optional

import numpy as np

from ..utils.config import ModelArgs
from ..utils.optimized_tokenizer import OptimizedTokenizer

Shape = TypeVar("Shape")


class Array(np.ndarray, Generic[Shape]): ...


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def silu(x):
    return x * (1 / (1 + np.exp(-x)))


def compute_cos_sin_cache(head_dim: int, max_seq_len: int, base: int = 10000):
    """Compute cosine and sine frequency cache for rotary embeddings"""
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2)[: (head_dim // 2)] / head_dim))
    t = np.arange(max_seq_len)
    freqs = np.outer(t, inv_freq)
    
    return np.cos(freqs), np.sin(freqs)


def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    """
    Optimized implementation of RoPE using direct indexing.
    
    Args:
        xq: Query vectors [batch, seq_len, n_heads, head_dim]
        xk: Key vectors [batch, seq_len, n_kv_heads, head_dim]
        freqs_cos: Cosine of frequencies [seq_len, head_dim//2]
        freqs_sin: Sine of frequencies [seq_len, head_dim//2]
        
    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    # Get real and imaginary parts using direct indexing
    # Even indices are real, odd indices are imaginary
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


def repeat_kv(x, n_rep: int):
    if n_rep == 1:
        return x
    else:
        return np.repeat(x, n_rep, axis=2)


class RMSNorm:
    def __init__(self, weight, eps: float = 1e-6):
        self.weight = weight
        self.eps = eps

    def __call__(self, x):
        # (B, L, D) -> (B, L, 1)
        variance = np.mean(x * x, axis=-1, keepdims=True)
        x = x / np.sqrt(variance + self.eps)
        return self.weight * x


class Attention:
    def __init__(
            self,
            q_weight,
            k_weight,
            v_weight,
            o_weight,
            head_dim: int,
            n_local_heads: int,
            n_local_kv_heads: int,
            max_seq_len: int,
            max_batch_size: int,
    ):
        self.q_weight = q_weight
        self.k_weight = k_weight
        self.v_weight = v_weight
        self.o_weight = o_weight
        self.head_dim = head_dim
        self.n_heads = n_local_heads
        self.n_kv_heads = n_local_kv_heads
        self.cache_k = np.zeros((max_batch_size, max_seq_len, n_local_kv_heads, head_dim))
        self.cache_v = np.zeros((max_batch_size, max_seq_len, n_local_kv_heads, head_dim))
        # If n_local_heads > n_local_kv_heads, we need to repeat the kv heads to match the q heads
        self.n_rep = n_local_heads // n_local_kv_heads if n_local_kv_heads > 0 else 0

    def __call__(self, x, start_pos: int, mask: Optional[Array["L, L"]],
                 freqs_cos, freqs_sin):
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
            k_seq = repeat_kv(k_seq, self.n_rep)
            v_seq = repeat_kv(v_seq, self.n_rep)
        
        # Reshape for attention computation
        xq = xq.transpose(0, 2, 1, 3)  # [batch, heads, seq_len, head_dim]
        k_seq = k_seq.transpose(0, 2, 1, 3)
        v_seq = v_seq.transpose(0, 2, 1, 3)
        
        # Compute attention scores
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = (xq @ k_seq.transpose(0, 1, 3, 2)) * scale
        
        if mask is not None:
            attn_scores = attn_scores + mask[None, None, :, :]
        
        attn_weights = softmax(attn_scores)
        attn_output = attn_weights @ v_seq
        
        # Reshape and project output
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        output = attn_output @ self.o_weight
        
        return output


class FeedForward:
    def __init__(self, up_weight, gate_weight, down_weight):
        self.up_weight = up_weight
        self.gate_weight = gate_weight
        self.down_weight = down_weight

    def __call__(self, x):
        # Get gate projection
        gate_proj = x @ self.gate_weight
        
        # Apply SiLU activation
        swish = silu(gate_proj)
        
        # Complete the feed-forward computation
        x_up = x @ self.up_weight
        x = swish * x_up
        x = x @ self.down_weight
        
        return x


class TransformerBlock:
    def __init__(
            self,
            attention: Attention,
            feed_forward: FeedForward,
            norm1_weight,
            norm2_weight,
            norm_eps: float,
    ):
        self.attention = attention
        self.feed_forward = feed_forward
        self.norm1 = RMSNorm(norm1_weight, norm_eps)
        self.norm2 = RMSNorm(norm2_weight, norm_eps)

    def __call__(self, x, start_pos: int, mask: Optional[Array["L, L"]],
                 freqs_cos, freqs_sin):
        # Self-attention with normalization
        h = self.norm1(x)
        h = self.attention(h, start_pos, mask, freqs_cos, freqs_sin)
        x = x + h

        # Feedforward with normalization
        h = self.norm2(x)
        h = self.feed_forward(h)
        x = x + h

        return x


class Llama:
    def __init__(self, model_path: str, args: ModelArgs):
        self.args = args
        from ..utils.loader import load_parameters
        weights = load_parameters(model_path)
        self.tok_embedding = weights.get("model.embed_tokens.weight")
        self.lm_head_weight = weights.get("lm_head.weight").T
        self.norm_weight = weights.get("model.norm.weight")

        self.layers = []
        for i in range(args.n_layers):
            block = TransformerBlock(
                attention=Attention(
                    q_weight=weights.get(f"model.layers.{i}.self_attn.q_proj.weight"),
                    k_weight=weights.get(f"model.layers.{i}.self_attn.k_proj.weight"),
                    v_weight=weights.get(f"model.layers.{i}.self_attn.v_proj.weight"),
                    o_weight=weights.get(f"model.layers.{i}.self_attn.o_proj.weight"),
                    head_dim=args.dim // args.n_heads,
                    n_local_heads=args.n_heads,
                    n_local_kv_heads=args.n_kv_heads if args.n_kv_heads is not None else args.n_heads,
                    max_seq_len=args.max_seq_len,
                    max_batch_size=args.max_batch_size,
                ),
                feed_forward=FeedForward(
                    up_weight=weights.get(f"model.layers.{i}.mlp.up_proj.weight"),
                    gate_weight=weights.get(f"model.layers.{i}.mlp.gate_proj.weight"),
                    down_weight=weights.get(f"model.layers.{i}.mlp.down_proj.weight"),
                ),
                norm1_weight=weights.get(f"model.layers.{i}.input_layernorm.weight"),
                norm2_weight=weights.get(f"model.layers.{i}.post_attention_layernorm.weight"),
                norm_eps=args.norm_eps,
            )
            self.layers.append(block)

        self.freqs_cos, self.freqs_sin = compute_cos_sin_cache(args.dim // args.n_heads, args.max_seq_len)

    def __call__(self, input_ids, start_pos: int = 0):
        # (B, L) -> (B, L, D)
        h = self.tok_embedding[input_ids]
        L = h.shape[1]
        mask = None
        if L > 1:
            mask = np.full((L, L), -np.inf)
            mask = np.triu(mask, k=1)

        # Transformer layers
        for layer in self.layers:
            h = layer(h, start_pos, mask, self.freqs_cos[:L], self.freqs_sin[:L])

        # Output normalization
        h = RMSNorm(self.norm_weight, self.args.norm_eps)(h)

        # Language model head
        logit = h @ self.lm_head_weight
        return logit

    def generate(self, input_ids, max_new_tokens: int):
        _, L = input_ids.shape
        for i, curr_pos in enumerate(range(L, L + max_new_tokens)):
            if i == 0:  # Prefill Phase
                inputs = input_ids
                pos = 0
            else:  # Decode Phase
                inputs = next_id
                pos = curr_pos - 1
            logits = self(inputs, pos)
            next_id = logits[:, -1, :].argmax(-1, keepdims=True)
            yield next_id