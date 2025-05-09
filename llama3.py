from __future__ import annotations

import math
import sys
import time
from typing import TypeVar, Generic, Optional

import numpy as np

from config import ModelArgs
from tokenizer import Tokenizer
from utils import load_parameters

np.random.seed(42)

Shape = TypeVar("Shape")


class Array(np.ndarray, Generic[Shape]): ...


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def silu(x):
    return x * (1 / (1 + np.exp(-x)))


def compute_cos_sin_cache(head_dim: int, max_seq_len: int, base: int = 10000, dtype=np.float32):
    inv_freq: Array["HD//2"] = 1.0 / (base ** (np.arange(0, head_dim, 2)[: (head_dim // 2)] / head_dim))
    t: Array["M"] = np.arange(max_seq_len)
    freqs: Array["M, HD//2"] = np.outer(t, inv_freq)

    return np.cos(freqs).astype(dtype), np.sin(freqs).astype(dtype)


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
    z: Array["B, L, QHN, HD"] = np.repeat(x, n_rep, axis=2)
    return z


class FeedForward:
    def __init__(self, up_weight: Array["FD, D"], gate_weight: Array["FD, D"], down_weight: Array["D, FD"], dtype=np.float32):
        self.up_weight = up_weight.T.astype(dtype)
        self.gate_weight = gate_weight.T.astype(dtype)
        self.down_weight = down_weight.T.astype(dtype)
        self.dtype = dtype

    def __call__(self, x: Array["B, L or 1, D"]):
        swish: Array["B, L or 1, FD"] = silu(x @ self.gate_weight)
        x_V: Array["B, L or 1, FD"] = x @ self.up_weight
        x_ff: Array["B, L or 1, FD"] = swish * x_V
        x_out: Array["B, L or 1, D"] = x_ff @ self.down_weight
        return x_out.astype(self.dtype)


class RMSNorm:
    def __init__(self, weight: Array["H"], eps: float, dtype=np.float32):
        self.weight = weight.astype(dtype)
        self.eps = eps
        self.dtype = dtype

    def __call__(self, x: Array["B, L or 1, D"]):
        z_float32: Array["B, L or 1, 1"] = (x.astype(np.float32) ** 2).mean(-1, keepdims=True) + self.eps
        z: Array["B, L or 1, D"] = x / np.sqrt(z_float32.astype(self.dtype))
        return (z * self.weight).astype(self.dtype)


class Attention:
    def __init__(self, q_weight: Array["D, D"], k_weight: Array["D, D"], v_weight: Array["D, D"],
                 o_weight: Array["D, D"], args: ModelArgs, dtype=np.float32):
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.dtype = dtype

        self.q_weight = q_weight.T.astype(self.dtype)
        self.k_weight = k_weight.T.astype(self.dtype)
        self.v_weight = v_weight.T.astype(self.dtype)
        self.o_weight = o_weight.T.astype(self.dtype)

        self.cache_k = np.zeros((args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim), dtype=self.dtype)
        self.cache_v = np.zeros((args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim), dtype=self.dtype)

    def __call__(self, x: Array["B, L or 1, D"], start_pos: int, mask: Optional[Array["L, L"]],
                 freqs_cos: Array["L or 1, HD//2"], freqs_sin: Array["L or 1, HD//2"]):
        B, L, _ = x.shape

        # QKV
        xq: Array["B, L or 1, D"] = x @ self.q_weight
        xk: Array["B, L or 1, D"] = x @ self.k_weight
        xv: Array["B, L or 1, D"] = x @ self.v_weight

        xq: Array["B, L or 1, QHN,  HD"] = xq.reshape(B, L, self.n_local_heads, self.head_dim).astype(self.dtype)
        xk: Array["B, L or 1, KVHN, HD"] = xk.reshape(B, L, self.n_local_kv_heads, self.head_dim).astype(self.dtype)
        xv: Array["B, L or 1, KVHN, HD"] = xv.reshape(B, L, self.n_local_kv_heads, self.head_dim).astype(self.dtype)

        # RoPE #2
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
        attention_float32: Array["B, HN, L or 1, L"] = (xq.astype(np.float32) @ xk.transpose(0, 1, 3, 2).astype(np.float32)) / math.sqrt(self.head_dim)
        if mask is not None:
            attention_float32 = attention_float32 + mask[None, None, :, :].astype(np.float32)
        attention = softmax(attention_float32).astype(self.dtype)
        output: Array["B, HN, L or 1, HD"] = attention @ xv

        # ["B, HN, L or 1, HD"] -> ["B, L or 1, D"]
        output: Array["B, L or 1, D"] = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        output: Array["B, L or 1, D"] = output @ self.o_weight

        return output.astype(self.dtype)


class TransformerBlock:
    def __init__(self, weight: dict, layer_id: int, args: ModelArgs):
        self.dtype = getattr(np, args.dtype)
        self.attention = Attention(
            weight.get(f"model.layers.{layer_id}.self_attn.q_proj.weight"),
            weight.get(f"model.layers.{layer_id}.self_attn.k_proj.weight"),
            weight.get(f"model.layers.{layer_id}.self_attn.v_proj.weight"),
            weight.get(f"model.layers.{layer_id}.self_attn.o_proj.weight"),
            args,
            dtype=self.dtype
        )
        self.feed_forward = FeedForward(
            weight.get(f"model.layers.{layer_id}.mlp.up_proj.weight"),
            weight.get(f"model.layers.{layer_id}.mlp.gate_proj.weight"),
            weight.get(f"model.layers.{layer_id}.mlp.down_proj.weight"),
            dtype=self.dtype
        )
        self.input_layernorm = RMSNorm(
            weight.get(f"model.layers.{layer_id}.input_layernorm.weight"),
            eps=args.norm_eps,
            dtype=self.dtype
        )
        self.post_attention_layernorm = RMSNorm(
            weight.get(f"model.layers.{layer_id}.post_attention_layernorm.weight"),
            eps=args.norm_eps,
            dtype=self.dtype
        )

    def __call__(self, x: Array["B, L or 1, D"], start_pos: int, mask: Array["L, L"],
                 freqs_cos: Array["L or 1, HD//2"], freqs_sin: Array["L or 1, HD//2"]):
        norm_x: Array["B, L or 1, D"] = self.input_layernorm(x)
        h1: Array["B, L or 1, D"] = self.attention(norm_x, start_pos, mask, freqs_cos, freqs_sin)
        z = (x + h1).astype(self.dtype)

        norm_z = self.post_attention_layernorm(z)
        h2: Array["B, L or 1, D"] = self.feed_forward(norm_z)
        out = (z + h2).astype(self.dtype)

        return out


class Llama:
    def __init__(self, model_path: str, args: ModelArgs):
        self.args = args
        self.dtype = getattr(np, args.dtype)

        weight = load_parameters(model_path)
        self.tok_embedding: Array["VS, D"] = weight.get("model.embed_tokens.weight").astype(self.dtype)

        self.freqs_cos, self.freqs_sin = compute_cos_sin_cache(
            args.dim // args.n_heads, args.max_seq_len, dtype=self.dtype
        )

        self.layers = []
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(weight, layer_id, args))

        self.norm = RMSNorm(weight.get("model.norm.weight"), eps=args.norm_eps, dtype=self.dtype)
        self.lm_head_weight: Array["D, VS"] = weight.get("lm_head.weight").T.astype(self.dtype)

        del weight

    def __call__(self, input_ids: Array["B, L"], start_pos: int):
        _, L = input_ids.shape
        h: Array["B, L or 1, D"] = self.tok_embedding[input_ids].astype(self.dtype)
        freqs_cos: Array["L or 1, HD//2"] = self.freqs_cos[start_pos: start_pos + L]
        freqs_sin: Array["L or 1, HD//2"] = self.freqs_sin[start_pos: start_pos + L]

        mask: Array["L, L"] = None
        if L > 1:
            mask = np.full((L, L), float("-inf"), dtype=self.dtype)
            mask = np.triu(mask, k=1)
            zeros_shape = (L, start_pos)
            if self.dtype is np.float16:
                mask = np.concatenate([np.zeros(zeros_shape, dtype=self.dtype), mask], axis=1)
            else:
                mask = np.concatenate([np.zeros(zeros_shape), mask], axis=1).astype(self.dtype)

        for i, layer in enumerate(self.layers):
            h: Array["B, L or 1, D"] = layer(h, start_pos, mask, freqs_cos, freqs_sin)

        h: Array["B, L or 1, D"] = self.norm(h)
        logit: Array["B, 1, VS"] = (h[:, [-1], :] @ self.lm_head_weight.astype(np.float32)).astype(np.float32)
        return logit

    def generate(self, input_ids: Array["B, L"], max_new_tokens: int):
        B, L_prompt = input_ids.shape
        curr_L = L_prompt
        for i in range(max_new_tokens):
            curr_pos = L_prompt + i
            if i == 0:  # Prefill Phase
                current_input_ids = input_ids
                pos = 0
            else:  # Decode Phase
                current_input_ids = next_id
                pos = curr_pos -1
            
            logits: Array["B, 1, VS"] = self(current_input_ids, pos)
            next_id = logits[:, -1, :].argmax(-1, keepdims=True).astype(np.int32)
            yield next_id
            curr_L +=1
            if curr_L >= self.args.max_seq_len:
                 break


if __name__ == '__main__':
    args = ModelArgs()

    print(f"Using precision: {args.dtype}")

    tokenizer = Tokenizer("./tokenizer.model.np")
    model = Llama("./stories15M.model.npz", args)

    if len(sys.argv) == 1:
        prompt = "I have a dream"
    else:
        prompt = sys.argv[1]

    print(f"\n{prompt}", end="")
    input_ids = np.array([tokenizer.encode(prompt)])
    start = time.time()
    _, L = input_ids.shape
    generated_tokens_count = 0
    for id_val in model.generate(input_ids, args.max_new_tokens):
        L += 1
        generated_tokens_count += 1
        output_id = id_val[0].tolist()
        if output_id[-1] in [tokenizer.eos_id, tokenizer.bos_id]:
            break
        print(tokenizer.decode(output_id), end="")
        sys.stdout.flush()
    elapsed = time.time() - start
    print(f"\n\nToken count: {L}, elapsed: {elapsed:.2f}s, {round(L / elapsed)} tokens/s")
