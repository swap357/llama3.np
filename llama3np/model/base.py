"""
Base implementation of Llama3 model using NumPy.

This is the original implementation without optimizations.
"""

from __future__ import annotations

import math
import time
from typing import TypeVar, Generic, Optional

import numpy as np

from ..utils.config import ModelArgs
from ..utils.tokenizer import Tokenizer

Shape = TypeVar("Shape")


class Array(np.ndarray, Generic[Shape]): ...


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def silu(x):
    return x * (1 / (1 + np.exp(-x)))


def compute_cos_sin_cache(head_dim: int, max_seq_len: int, base: int = 10000):
    inv_freq: Array["HD//2"] = 1.0 / (base ** (np.arange(0, head_dim, 2)[: (head_dim // 2)] / head_dim))
    t: Array["M"] = np.arange(max_seq_len)
    freqs: Array["M, HD//2"] = np.outer(t, inv_freq)
    return np.cos(freqs), np.sin(freqs)


def apply_rotary_emb(xq: Array["B, L or 1, QHN,  HD"], xk: Array["B, L or 1, KVHN, HD"],
                     freqs_cos: Array["L or 1, HD//2"], freqs_sin: Array["L or 1, HD//2"]):
    # ["B, L or 1, QHN, HD"] -> ["B, L or 1, QHN,  HD//2, 2"]
    xqri: Array["B, L or 1, QHN,  HD//2, 2"] = xq.reshape(xq.shape[:-1] + (-1, 2))
    xkri: Array["B, L or 1, KVHN, HD//2, 2"] = xk.reshape(xk.shape[:-1] + (-1, 2))

    # Reshape `xq` and `xk` to match the complex representation.
    xq_r, xq_i = np.split(xqri, 2, axis=-1)
    xq_r: Array["B, L or 1, QHN,  HD//2"] = xq_r.squeeze(-1)
    xq_i: Array["B, L or 1, QHN,  HD//2"] = xq_i.squeeze(-1)

    xk_r, xk_i = np.split(xkri, 2, axis=-1)
    xk_r: Array["B, L or 1, KVHN, HD//2"] = xk_r.squeeze(-1)
    xk_i: Array["B, L or 1, KVHN, HD//2"] = xk_i.squeeze(-1)

    # Reshape `freqs_cos` and `freqs_sin` for broadcasting.
    freqs_cos: Array["B, L or 1, 1, HD//2"] = np.expand_dims(freqs_cos, axis=(0, 2))
    freqs_sin: Array["B, L or 1, 1, HD//2"] = np.expand_dims(freqs_sin, axis=(0, 2))

    # Apply rotation using real numbers.
    xq_out_r: Array["B, L or 1, QHN,  HD//2"] = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i: Array["B, L or 1, QHN,  HD//2"] = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r: Array["B, L or 1, KVHN, HD//2"] = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i: Array["B, L or 1, KVHN, HD//2"] = xk_r * freqs_sin + xk_i * freqs_cos

    # Flatten last two dimensions.
    xq_out: Array["B, L or 1, QHN,  HD//2, 2"] = np.stack([xq_out_r, xq_out_i], axis=-1)
    xk_out: Array["B, L or 1, KVHN, HD//2, 2"] = np.stack([xk_out_r, xk_out_i], axis=-1)
    xq_out: Array["B, L or 1, QHN,  HD"] = xq_out.reshape(xq_out.shape[:-2] + (-1,))
    xk_out: Array["B, L or 1, KVHN, HD"] = xk_out.reshape(xk_out.shape[:-2] + (-1,))
    
    return xq_out, xk_out


def repeat_kv(x: Array["B, L, KVHN, HD"], n_rep: int):
    if n_rep == 1:
        return x
    else:
        return np.repeat(x, n_rep, axis=2)


class RMSNorm:
    def __init__(self, weight: Array["D"], eps: float = 1e-6):
        self.weight = weight
        self.eps = eps

    def __call__(self, x: Array["B, L, D"]) -> Array["B, L, D"]:
        # (B, L, D) -> (B, L, 1)
        variance: Array["B, L, 1"] = np.mean(x * x, axis=-1, keepdims=True)
        x = x / np.sqrt(variance + self.eps)
        return self.weight * x


class Attention:
    def __init__(
            self,
            wq: Array["D, QHD"],
            wk: Array["D, KVHD"],
            wv: Array["D, KVHD"],
            wo: Array["QHD, D"],
            n_heads: int,
            n_kv_heads: int,
            max_seq_len: int,
            max_batch_size: int,
    ):
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = wq.shape[1] // n_heads
        self.n_rep = n_heads // n_kv_heads if n_kv_heads > 0 else 0
        self.cache_k = np.zeros((max_batch_size, max_seq_len, n_kv_heads, self.head_dim))
        self.cache_v = np.zeros((max_batch_size, max_seq_len, n_kv_heads, self.head_dim))

    def __call__(self, x: Array["B, L or 1, D"], start_pos: int, mask: Optional[Array["L, L"]],
                 freqs_cos: Array["L or 1, HD//2"], freqs_sin: Array["L or 1, HD//2"]):
        B, L, _ = x.shape

        # compute query, key, values
        xq: Array["B, L or 1, HD"] = x @ self.wq
        xk: Array["B, L or 1, HD"] = x @ self.wk
        xv: Array["B, L or 1, HD"] = x @ self.wv

        # reshape for heads
        xq: Array["B, L or 1, H, HD"] = xq.reshape(B, L, self.n_heads, self.head_dim)
        xk: Array["B, L or 1, H, HD"] = xk.reshape(B, L, self.n_kv_heads, self.head_dim)
        xv: Array["B, L or 1, H, HD"] = xv.reshape(B, L, self.n_kv_heads, self.head_dim)

        # apply rotary embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # store kv cache
        self.cache_k[:B, start_pos: start_pos + L] = xk
        self.cache_v[:B, start_pos: start_pos + L] = xv
        keys: Array["B, L, H, HD"] = self.cache_k[:B, : start_pos + L]
        values: Array["B, L, H, HD"] = self.cache_v[:B, : start_pos + L]

        # expand for multi-query attention
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # transpose for attention calculation
        xq: Array["B, H, L or 1, HD"] = xq.transpose(0, 2, 1, 3)
        keys: Array["B, H, L, HD"] = keys.transpose(0, 2, 1, 3)
        values: Array["B, H, L, HD"] = values.transpose(0, 2, 1, 3)

        # compute attention scores
        scores: Array["B, H, L or 1, L"] = xq @ keys.transpose(0, 1, 3, 2) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask[None, None, :, :]
        scores = softmax(scores)
        output: Array["B, H, L or 1, HD"] = scores @ values

        # transpose back and reshape
        output: Array["B, L or 1, H, HD"] = output.transpose(0, 2, 1, 3)
        output: Array["B, L or 1, D"] = output.reshape(B, L, -1)
        
        # final projection
        output: Array["B, L or 1, D"] = output @ self.wo
        
        return output


class FeedForward:
    def __init__(self, w1: Array["D, FD"], w2: Array["FD, D"], w3: Array["D, FD"]):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def __call__(self, x: Array["B, L or 1, D"]):
        # Apply SwiGLU
        swish: Array["B, L or 1, FD"] = silu(x @ self.w3)
        x_proj: Array["B, L or 1, FD"] = x @ self.w1
        x_gate: Array["B, L or 1, FD"] = swish * x_proj
        
        # Output projection
        output: Array["B, L or 1, D"] = x_gate @ self.w2
        return output


class TransformerBlock:
    def __init__(
            self,
            weights: dict,
            layer_id: int,
            args: ModelArgs,
    ):
        self.attention = Attention(
            wq=weights.get(f"model.layers.{layer_id}.self_attn.q_proj.weight"),
            wk=weights.get(f"model.layers.{layer_id}.self_attn.k_proj.weight"),
            wv=weights.get(f"model.layers.{layer_id}.self_attn.v_proj.weight"),
            wo=weights.get(f"model.layers.{layer_id}.self_attn.o_proj.weight"),
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads if args.n_kv_heads is not None else args.n_heads,
            max_seq_len=args.max_seq_len,
            max_batch_size=args.max_batch_size,
        )
        
        self.feed_forward = FeedForward(
            w1=weights.get(f"model.layers.{layer_id}.mlp.up_proj.weight"),
            w2=weights.get(f"model.layers.{layer_id}.mlp.down_proj.weight"),
            w3=weights.get(f"model.layers.{layer_id}.mlp.gate_proj.weight"),
        )
        
        self.attention_norm = RMSNorm(
            weight=weights.get(f"model.layers.{layer_id}.input_layernorm.weight"),
            eps=args.norm_eps,
        )
        
        self.ffn_norm = RMSNorm(
            weight=weights.get(f"model.layers.{layer_id}.post_attention_layernorm.weight"),
            eps=args.norm_eps,
        )

    def __call__(self, x: Array["B, L, D"], start_pos: int, mask: Optional[Array["L, L"]],
                 freqs_cos: Array["L, HD//2"], freqs_sin: Array["L, HD//2"]) -> Array["B, L, D"]:
        # Self-attention with residual connection
        h = self.attention_norm(x)
        h = self.attention(h, start_pos, mask, freqs_cos, freqs_sin)
        x = x + h

        # Feed-forward with residual connection
        h = self.ffn_norm(x)
        h = self.feed_forward(h)
        x = x + h

        return x


class Llama:
    def __init__(self, model_path: str, args: ModelArgs):
        self.args = args

        from ..utils.loader import load_parameters
        weight = load_parameters(model_path)
        self.tok_embedding: Array["VS, D"] = weight.get("model.embed_tokens.weight")

        # RoPE frequency cache
        self.freqs_cos, self.freqs_sin = compute_cos_sin_cache(args.dim // args.n_heads, args.max_seq_len)

        self.layers = []
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(weight, layer_id, args))

        self.norm = RMSNorm(weight.get("model.norm.weight"), eps=args.norm_eps)
        self.lm_head_weight: Array["D, VS"] = weight.get("lm_head.weight").T

    def __call__(self, input_ids: Array["B, L"], start_pos: int):
        _, L = input_ids.shape
        h: Array["B, L or 1, D"] = self.tok_embedding[input_ids]
        # ["M, HD//2"] -> ["L or 1, HD//2"]
        freqs_cos: Array["L or 1, HD//2"] = self.freqs_cos[start_pos: start_pos + L]
        freqs_sin: Array["L or 1, HD//2"] = self.freqs_sin[start_pos: start_pos + L]

        # `mask` is generated only once at the beginning.
        mask: Array["L, L"] = None
        if L > 1:
            mask = np.full((L, L), float("-inf"))
            mask = np.triu(mask, k=1)
            mask = np.concatenate([np.zeros((L, start_pos)), mask], axis=1)

        # Transformer Layers
        for i, layer in enumerate(self.layers):
            h: Array["B, L or 1, D"] = layer(h, start_pos, mask, freqs_cos, freqs_sin)

        # RMSNorm
        h: Array["B, L or 1, D"] = self.norm(h)
        # Only forward the output from the last position.
        # ["B, 1, VS"] = ["B, 1(L), D"] @ ["D, VS"]
        logit: Array["B, L, VS"] = h @ self.lm_head_weight
        return logit

    def generate(self, input_ids: Array["B, L"], max_new_tokens: int):
        _, L = input_ids.shape
        for i, curr_pos in enumerate(range(L, L + max_new_tokens)):
            if i == 0:  # Prefill Phase
                inputs = input_ids
                pos = 0
            else:  # Decode Phase
                inputs = next_id
                pos = curr_pos - 1
            logits: Array["B, 1, VS"] = self(inputs, pos)
            next_id = logits[:, -1, :].argmax(-1, keepdims=True)
            yield next_id