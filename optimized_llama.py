"""
Optimized implementation of the llama3.np model.

This version incorporates the optimizations identified in our experiments:
1. Improved RoPE implementation using direct indexing
2. Optimized tokenizer using dictionary-based lookup
"""

from __future__ import annotations

import math
import sys
import time
from typing import TypeVar, Generic, Optional, Dict, List, Tuple

import numpy as np

from config import ModelArgs
from utils import load_parameters

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


def repeat_kv(x: Array["B, L, KVHN, HD"], n_rep: int):
    if n_rep == 1:
        return x
    else:
        return np.repeat(x, n_rep, axis=2)


class FeedForward:
    def __init__(self, up_weight: Array["FD, D"], gate_weight: Array["FD, D"], down_weight: Array["D, FD"]):
        self.up_weight = up_weight.T
        self.gate_weight = gate_weight.T
        self.down_weight = down_weight.T

    def __call__(self, x: Array["B, L or 1, D"]):
        # FD = 2 * 4 * D / 3
        swish: Array["B, L or 1, FD"] = silu(x @ self.gate_weight)
        x_V: Array["B, L or 1, FD"] = x @ self.up_weight
        x: Array["B, L or 1, FD"] = swish * x_V
        x: Array["B, L or 1, D"] = x @ self.down_weight
        return x


class Attention:
    def __init__(
            self,
            q_weight: Array["D, HND"],
            k_weight: Array["D, KVHND"],
            v_weight: Array["D, KVHND"],
            o_weight: Array["HND, D"],
            head_dim: int,
            n_local_heads: int,
            n_local_kv_heads: int,
            max_seq_len: int,
            max_batch_size: int,
    ):
        self.q_weight = q_weight.T
        self.k_weight = k_weight.T
        self.v_weight = v_weight.T
        self.o_weight = o_weight.T
        self.head_dim = head_dim
        self.n_local_heads = n_local_heads
        self.n_local_kv_heads = n_local_kv_heads
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.cache_k = np.zeros((max_batch_size, max_seq_len, n_local_kv_heads, head_dim))
        self.cache_v = np.zeros((max_batch_size, max_seq_len, n_local_kv_heads, head_dim))
        # If n_local_heads > n_local_kv_heads, we need to repeat the kv heads to match the q heads
        self.n_rep = n_local_heads // n_local_kv_heads

    def __call__(self, x: Array["B, L or 1, D"], start_pos: int, mask: Optional[Array["L, L"]],
                 freqs_cos: Array["L or 1, HD//2"], freqs_sin: Array["L or 1, HD//2"]):
        B, L, _ = x.shape

        # QKV
        xq: Array["B, L or 1, D"] = x @ self.q_weight
        xk: Array["B, L or 1, D"] = x @ self.k_weight
        xv: Array["B, L or 1, D"] = x @ self.v_weight

        xq: Array["B, L or 1, QHN,  HD"] = xq.reshape(B, L, self.n_local_heads, self.head_dim)
        xk: Array["B, L or 1, KVHN, HD"] = xk.reshape(B, L, self.n_local_kv_heads, self.head_dim)
        xv: Array["B, L or 1, KVHN, HD"] = xv.reshape(B, L, self.n_local_kv_heads, self.head_dim)

        # RoPE - using our optimized implementation
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # KV Cache
        self.cache_k[:B, start_pos: start_pos + L] = xk
        self.cache_v[:B, start_pos: start_pos + L] = xv
        ks: Array["B, L, KVHN, HD"] = self.cache_k[:B, : start_pos + L]
        vs: Array["B, L, KVHN, HD"] = self.cache_v[:B, : start_pos + L]

        # GQA
        xk: Array["B, L, HN, HD"] = repeat_kv(ks, self.n_rep)
        xv: Array["B, L, HN, HD"] = repeat_kv(vs, self.n_rep)

        # ["B, L, HN, HD"] -> ["B, HN, L, HD"]
        xq: Array["B, HN, L or 1, HD"] = xq.transpose(0, 2, 1, 3)
        xk: Array["B, HN, L, HD"] = xk.transpose(0, 2, 1, 3)
        xv: Array["B, HN, L, HD"] = xv.transpose(0, 2, 1, 3)

        # Scaled Dot-Product Attention
        # ["B, HN, L or 1, HD"] @ ["B, HN, HD, L"] -> ["B, HN, L or 1, L"]
        attention: Array["B, HN, L or 1, L"] = xq @ xk.transpose(0, 1, 3, 2) / math.sqrt(self.head_dim)
        # `mask` is used only once at the beginning.
        if mask is not None:
            attention = attention + mask[None, None, :, :]
        attention = softmax(attention)
        output: Array["B, HN, L or 1, HD"] = attention @ xv

        # ["B, HN, L or 1, HD"] -> ["B, L or 1, D"]
        output: Array["B, L or 1, D"] = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output: Array["B, L or 1, D"] = output @ self.o_weight

        return output


class RMSNorm:
    def __init__(self, weight: Array["D"], eps: float = 1e-6):
        self.weight = weight
        self.eps = eps

    def __call__(self, x: Array["B, L, D"]) -> Array["B, L, D"]:
        # (B, L, D) -> (B, L, 1)
        variance: Array["B, L, 1"] = np.mean(x * x, axis=-1, keepdims=True)
        x = x / np.sqrt(variance + self.eps)
        return self.weight * x


class TransformerBlock:
    def __init__(
            self,
            attention: Attention,
            feed_forward: FeedForward,
            norm1_weight: Array["D"],
            norm2_weight: Array["D"],
            norm_eps: float,
    ):
        self.attention = attention
        self.feed_forward = feed_forward
        self.norm1 = RMSNorm(norm1_weight, norm_eps)
        self.norm2 = RMSNorm(norm2_weight, norm_eps)

    def __call__(self, x: Array["B, L, D"], start_pos: int, mask: Optional[Array["L, L"]],
                 freqs_cos: Array["L, HD//2"], freqs_sin: Array["L, HD//2"]) -> Array["B, L, D"]:
        # Self-attention with normalization
        h: Array["B, L, D"] = self.norm1(x)
        h = self.attention(h, start_pos, mask, freqs_cos, freqs_sin)
        x = x + h

        # Feedforward with normalization
        h = self.norm2(x)
        h = self.feed_forward(h)
        x = x + h

        return x


class OptimizedTokenizer:
    def __init__(self, model_path: str):
        """
        Initialize the tokenizer with a dictionary-based lookup for improved performance.
        """
        import json

        with open(model_path, "r", encoding="utf-8") as f:
            model = json.load(f)
        self.vocab = model["tokens"]
        self.scores = model["scores"]
        self.bos_id = 1
        self.eos_id = 2
        
        # Create a dictionary mapping from tokens to their first occurrence index
        self.token_to_id = {}
        for i, token in enumerate(self.vocab):
            # Only add if not already in the dictionary (keep first occurrence)
            if token not in self.token_to_id:
                self.token_to_id[token] = i

    def str_lookup(self, token: str) -> int:
        """
        Optimized token lookup using a dictionary instead of list.index()
        """
        return self.token_to_id.get(token, -1)

    def encode(
            self,
            text: str,
            add_bos: bool = True,
            add_eos: bool = False,
    ) -> List[int]:
        tokens = []
        for pos, char in enumerate(text):
            id = self.str_lookup(char)
            if id >= 0:
                tokens.append(id)
        while True:
            best_score = -1e10
            best_id = -1
            best_idx = -1

            for i in range(len(tokens) - 1):
                # Check if we can merge the pair (tokens[i], tokens[i+1])
                string = self.vocab[tokens[i]] + self.vocab[tokens[i + 1]]
                id = self.str_lookup(string)
                if id != -1 and self.scores[id] > best_score:
                    best_score = self.scores[id]
                    best_id = id
                    best_idx = i

            if best_idx == -1:
                break

            # Merge the consecutive pair (best_idx, best_idx+1) into new token best_id
            tokens[best_idx] = best_id
            # Delete token at position best_idx+1, shift the entire sequence back 1
            tokens = tokens[0: best_idx + 1] + tokens[best_idx + 2:]
        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, ids: List[int]) -> str:
        res = []
        for i in ids:
            token = self.vocab[i]
            res.append(token)
        text = "".join(res)
        text = text.strip("<s>").strip("</s>")
        return text


class Llama:
    def __init__(self, model_path: str, args: ModelArgs):
        self.args = args
        weights = load_parameters(model_path)
        self.tok_embedding = weights.get("model.embed_tokens.weight")
        self.lm_head_weight = weights.get("lm_head.weight").T  # Need to transpose
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

    # Method removed as it's no longer needed

    def __call__(self, input_ids: Array["B, L"], start_pos: int = 0) -> Array["B, L, VS"]:
        # (B, L) -> (B, L, D)
        h: Array["B, L, D"] = self.tok_embedding[input_ids]
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


if __name__ == '__main__':
    args = ModelArgs()

    tokenizer = OptimizedTokenizer("./tokenizer.model.np")
    model = Llama("./stories15M.model.npz", args)

    if len(sys.argv) == 1:
        prompt = "I have a dream"
    else:
        prompt = sys.argv[1]
        # Check for max tokens argument
        if len(sys.argv) > 2 and sys.argv[2].isdigit():
            args.max_new_tokens = int(sys.argv[2])

    print(f"\n{prompt}", end="")
    input_ids = np.array([tokenizer.encode(prompt)])
    start = time.time()
    _, L = input_ids.shape
    for id in model.generate(input_ids, args.max_new_tokens):
        L += 1
        output_id = id[0].tolist()
        if output_id[-1] in [tokenizer.eos_id, tokenizer.bos_id]:
            break
        print(tokenizer.decode(output_id), end="")
        sys.stdout.flush()
    elapsed = time.time() - start
    print(f"\n\nToken count: {L}, elapsed: {elapsed:.2f}s, {round(L / elapsed)} tokens/s")