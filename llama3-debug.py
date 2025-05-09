from __future__ import annotations

import math
import sys
import time
from typing import TypeVar, Generic, Optional

import numpy as np

from config import ModelArgs
from tokenizer import Tokenizer
from utils import load_parameters

Shape = TypeVar("Shape")

np.random.seed(42)

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
    z: Array["B, L, QHN, HD"] = np.repeat(x, n_rep, axis=2)
    return z


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


class RMSNorm:
    def __init__(self, weight: Array["H"], eps: float):
        self.weight = weight
        self.eps = eps

    def __call__(self, x: Array["B, L or 1, D"]):
        z: Array["B, L or 1, 1"] = (x ** 2).mean(-1, keepdims=True) + self.eps
        z: Array["B, L or 1, D"] = x / np.sqrt(z)
        return z * self.weight


class Attention:
    _debug_dtypes_printed_attention = False

    def __init__(self, q_weight: Array["D, D"], k_weight: Array["D, D"], v_weight: Array["D, D"],
                 o_weight: Array["D, D"], args: ModelArgs):
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.model_args_dtype = args.dtype

        self.q_weight = q_weight.T
        self.k_weight = k_weight.T
        self.v_weight = v_weight.T
        self.o_weight = o_weight.T

        self.cache_k = np.zeros((args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim))
        self.cache_v = np.zeros((args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim))
        
        if not Attention._debug_dtypes_printed_attention:
            print("\nattention init")
            for name, arr in [( "q_weight", self.q_weight), ("k_weight", self.k_weight),
                              ("v_weight", self.v_weight), ("o_weight", self.o_weight),
                              ("cache_k", self.cache_k), ("cache_v", self.cache_v)]:
                print(f"  {name: <10}: dtype={str(arr.dtype): <8}, shape={str(arr.shape): <25}, min={np.min(arr): .4f}, max={np.max(arr): .4f}, mean={np.mean(arr): .4f}")

    def __call__(self, x: Array["B, L or 1, D"], start_pos: int, mask: Optional[Array["L, L"]],
                 freqs_cos: Array["L or 1, HD//2"], freqs_sin: Array["L or 1, HD//2"]):

        if not Attention._debug_dtypes_printed_attention:
            print("\nattention call (first pass): trace")
            print(f"  config.dtype: {self.model_args_dtype}")
            print(f"  input x:      dtype={str(x.dtype): <8}, shape={str(x.shape): <25}, min={np.min(x): .4f}, max={np.max(x): .4f}, mean={np.mean(x): .4f}")
            if mask is not None:
                print(f"  input mask:   dtype={str(mask.dtype): <8}, shape={str(mask.shape): <25} (stats omitted due to -inf)")
            else:
                print("  input mask:   none")
            print(f"  freqs_cos:    dtype={str(freqs_cos.dtype): <8}, shape={str(freqs_cos.shape): <25}, min={np.min(freqs_cos): .4f}, max={np.max(freqs_cos): .4f}, mean={np.mean(freqs_cos): .4f}")
            print(f"  freqs_sin:    dtype={str(freqs_sin.dtype): <8}, shape={str(freqs_sin.shape): <25}, min={np.min(freqs_sin): .4f}, max={np.max(freqs_sin): .4f}, mean={np.mean(freqs_sin): .4f}")

        B, L, D_in = x.shape

        # QKV Projections
        xq_proj: Array["B, L or 1, D"] = x @ self.q_weight
        xk_proj: Array["B, L or 1, D"] = x @ self.k_weight
        xv_proj: Array["B, L or 1, D"] = x @ self.v_weight

        if not Attention._debug_dtypes_printed_attention:
            print(f"  qkv_proj xq:  dtype={str(xq_proj.dtype): <8}, shape={str(xq_proj.shape): <25}, min={np.min(xq_proj): .4f}, max={np.max(xq_proj): .4f}, mean={np.mean(xq_proj): .4f}")
            print(f"  qkv_proj xk:  dtype={str(xk_proj.dtype): <8}, shape={str(xk_proj.shape): <25}, min={np.min(xk_proj): .4f}, max={np.max(xk_proj): .4f}, mean={np.mean(xk_proj): .4f}")
            print(f"  qkv_proj xv:  dtype={str(xv_proj.dtype): <8}, shape={str(xv_proj.shape): <25}, min={np.min(xv_proj): .4f}, max={np.max(xv_proj): .4f}, mean={np.mean(xv_proj): .4f}")

        # Reshape QKV
        xq: Array["B, L or 1, QHN,  HD"] = xq_proj.reshape(B, L, self.n_local_heads, self.head_dim)
        xk: Array["B, L or 1, KVHN, HD"] = xk_proj.reshape(B, L, self.n_local_kv_heads, self.head_dim)
        xv: Array["B, L or 1, KVHN, HD"] = xv_proj.reshape(B, L, self.n_local_kv_heads, self.head_dim)

        if not Attention._debug_dtypes_printed_attention:
            print(f"  qkv_reshape xq: dtype={str(xq.dtype): <8}, shape={str(xq.shape): <25}, min={np.min(xq): .4f}, max={np.max(xq): .4f}, mean={np.mean(xq): .4f}")
            print(f"  qkv_reshape xk: dtype={str(xk.dtype): <8}, shape={str(xk.shape): <25}, min={np.min(xk): .4f}, max={np.max(xk): .4f}, mean={np.mean(xk): .4f}")
            print(f"  qkv_reshape xv: dtype={str(xv.dtype): <8}, shape={str(xv.shape): <25}, min={np.min(xv): .4f}, max={np.max(xv): .4f}, mean={np.mean(xv): .4f}")

        # Apply RoPE
        xq_rope, xk_rope = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        if not Attention._debug_dtypes_printed_attention:
            print(f"  rope xq:      dtype={str(xq_rope.dtype): <8}, shape={str(xq_rope.shape): <25}, min={np.min(xq_rope): .4f}, max={np.max(xq_rope): .4f}, mean={np.mean(xq_rope): .4f}")
            print(f"  rope xk:      dtype={str(xk_rope.dtype): <8}, shape={str(xk_rope.shape): <25}, min={np.min(xk_rope): .4f}, max={np.max(xk_rope): .4f}, mean={np.mean(xk_rope): .4f}")

        # Update KV Cache
        self.cache_k[:B, start_pos: start_pos + L] = xk_rope
        self.cache_v[:B, start_pos: start_pos + L] = xv
        
        if not Attention._debug_dtypes_printed_attention:
            print(f"  kv_cache write xk_rope (to k_cache): dtype={str(xk_rope.dtype):<8}, shape={str(xk_rope.shape):<25}, min={np.min(xk_rope):.4f}, max={np.max(xk_rope):.4f}, mean={np.mean(xk_rope):.4f}")
            print(f"  kv_cache write xv (to v_cache):    dtype={str(xv.dtype):<8}, shape={str(xv.shape):<25}, min={np.min(xv):.4f}, max={np.max(xv):.4f}, mean={np.mean(xv):.4f}")
            print(f"  k_cache overall:                  dtype={str(self.cache_k.dtype): <8}, shape={str(self.cache_k.shape): <25}")
            print(f"  v_cache overall:                  dtype={str(self.cache_v.dtype): <8}, shape={str(self.cache_v.shape): <25}")

        # Retrieve cached K and V
        ks: Array["B, L_cache, KVHN, HD"] = self.cache_k[:B, : start_pos + L]
        vs: Array["B, L_cache, KVHN, HD"] = self.cache_v[:B, : start_pos + L]

        if not Attention._debug_dtypes_printed_attention:
            print(f"  kv_cache read ks: dtype={str(ks.dtype): <8}, shape={str(ks.shape): <25}, min={np.min(ks): .4f}, max={np.max(ks): .4f}, mean={np.mean(ks): .4f}")
            print(f"  kv_cache read vs: dtype={str(vs.dtype): <8}, shape={str(vs.shape): <25}, min={np.min(vs): .4f}, max={np.max(vs): .4f}, mean={np.mean(vs): .4f}")

        # GQA: Repeat K/V heads
        xk_gqa: Array["B, L_cache, HN, HD"] = repeat_kv(ks, self.n_rep)
        xv_gqa: Array["B, L_cache, HN, HD"] = repeat_kv(vs, self.n_rep)

        if not Attention._debug_dtypes_printed_attention:
            print(f"  gqa xk:       dtype={str(xk_gqa.dtype): <8}, shape={str(xk_gqa.shape): <25}, min={np.min(xk_gqa): .4f}, max={np.max(xk_gqa): .4f}, mean={np.mean(xk_gqa): .4f}")
            print(f"  gqa xv:       dtype={str(xv_gqa.dtype): <8}, shape={str(xv_gqa.shape): <25}, min={np.min(xv_gqa): .4f}, max={np.max(xv_gqa): .4f}, mean={np.mean(xv_gqa): .4f}")

        # Transpose for score calculation
        xq_transposed: Array["B, HN, L_token, HD"] = xq_rope.transpose(0, 2, 1, 3)
        xk_transposed: Array["B, HN, L_cache, HD"] = xk_gqa.transpose(0, 2, 1, 3)
        xv_transposed: Array["B, HN, L_cache, HD"] = xv_gqa.transpose(0, 2, 1, 3)

        if not Attention._debug_dtypes_printed_attention:
            print(f"  transpose xq:   dtype={str(xq_transposed.dtype): <8}, shape={str(xq_transposed.shape): <25}, min={np.min(xq_transposed): .4f}, max={np.max(xq_transposed): .4f}, mean={np.mean(xq_transposed): .4f}")
            print(f"  transpose xk:   dtype={str(xk_transposed.dtype): <8}, shape={str(xk_transposed.shape): <25}, min={np.min(xk_transposed): .4f}, max={np.max(xk_transposed): .4f}, mean={np.mean(xk_transposed): .4f}")
            print(f"  transpose xv:   dtype={str(xv_transposed.dtype): <8}, shape={str(xv_transposed.shape): <25}, min={np.min(xv_transposed): .4f}, max={np.max(xv_transposed): .4f}, mean={np.mean(xv_transposed): .4f}")

        # Scaled Dot-Product Attention
        scores_raw: Array["B, HN, L_token, L_cache"] = xq_transposed @ xk_transposed.transpose(0, 1, 3, 2)
        scores_scaled: Array["B, HN, L_token, L_cache"] = scores_raw / math.sqrt(self.head_dim)
        
        if not Attention._debug_dtypes_printed_attention:
            print(f"  scores_raw:     dtype={str(scores_raw.dtype): <8}, shape={str(scores_raw.shape): <25}, min={np.min(scores_raw): .4f}, max={np.max(scores_raw): .4f}, mean={np.mean(scores_raw): .4f}")
            print(f"  scores_scaled:  dtype={str(scores_scaled.dtype): <8}, shape={str(scores_scaled.shape): <25}, min={np.min(scores_scaled): .4f}, max={np.max(scores_scaled): .4f}, mean={np.mean(scores_scaled): .4f}")

        scores_masked = scores_scaled
        if mask is not None:
            current_mask_slice = mask[None, None, -L:, :start_pos+L] if L == 1 and start_pos > 0 else mask[None,None,:,:]
            scores_masked = scores_scaled + current_mask_slice
            if not Attention._debug_dtypes_printed_attention:
                if np.all(np.isfinite(current_mask_slice)):
                    print(f"  mask_slice:   dtype={str(current_mask_slice.dtype): <8}, shape={str(current_mask_slice.shape): <25}, min={np.min(current_mask_slice): .4f}, max={np.max(current_mask_slice): .4f}, mean={np.mean(current_mask_slice): .4f}")
                else:
                    print(f"  mask_slice:   dtype={str(current_mask_slice.dtype): <8}, shape={str(current_mask_slice.shape): <25} (contains non-finite)")
                print(f"  scores_masked:  dtype={str(scores_masked.dtype): <8}, shape={str(scores_masked.shape): <25}, min={np.min(scores_masked): .4f}, max={np.max(scores_masked): .4f}, mean={np.mean(scores_masked): .4f}")
        
        attention_weights: Array["B, HN, L_token, L_cache"] = softmax(scores_masked)
        
        if not Attention._debug_dtypes_printed_attention:
            print(f"  attn_weights:   dtype={str(attention_weights.dtype): <8}, shape={str(attention_weights.shape): <25}, min={np.min(attention_weights): .4f}, max={np.max(attention_weights): .4f}, mean={np.mean(attention_weights): .4f}")

        # Output projection
        output_attended: Array["B, HN, L_token, HD"] = attention_weights @ xv_transposed
        
        if not Attention._debug_dtypes_printed_attention:
            print(f"  output_attn:    dtype={str(output_attended.dtype): <8}, shape={str(output_attended.shape): <25}, min={np.min(output_attended): .4f}, max={np.max(output_attended): .4f}, mean={np.mean(output_attended): .4f}")

        # Reshape and final projection
        output_reshaped: Array["B, L_token, D"] = output_attended.transpose(0, 2, 1, 3).reshape(B, L, -1)
        final_output: Array["B, L_token, D"] = output_reshaped @ self.o_weight

        if not Attention._debug_dtypes_printed_attention:
            print(f"  output_reshape: dtype={str(output_reshaped.dtype): <8}, shape={str(output_reshaped.shape): <25}, min={np.min(output_reshaped): .4f}, max={np.max(output_reshaped): .4f}, mean={np.mean(output_reshaped): .4f}")
            print(f"  final_output:   dtype={str(final_output.dtype): <8}, shape={str(final_output.shape): <25}, min={np.min(final_output): .4f}, max={np.max(final_output): .4f}, mean={np.mean(final_output): .4f}")
            print("--- end attention call trace ---\n")
            Attention._debug_dtypes_printed_attention = True

        return final_output


class TransformerBlock:
    def __init__(self, weight: dict, layer_id: int, args: ModelArgs):
        self.attention = Attention(
            weight.get(f"model.layers.{layer_id}.self_attn.q_proj.weight"),
            weight.get(f"model.layers.{layer_id}.self_attn.k_proj.weight"),
            weight.get(f"model.layers.{layer_id}.self_attn.v_proj.weight"),
            weight.get(f"model.layers.{layer_id}.self_attn.o_proj.weight"),
            args
        )
        self.feed_forward = FeedForward(
            weight.get(f"model.layers.{layer_id}.mlp.up_proj.weight"),
            weight.get(f"model.layers.{layer_id}.mlp.gate_proj.weight"),
            weight.get(f"model.layers.{layer_id}.mlp.down_proj.weight"),
        )
        self.input_layernorm = RMSNorm(
            weight.get(f"model.layers.{layer_id}.input_layernorm.weight"),
            eps=args.norm_eps
        )
        self.post_attention_layernorm = RMSNorm(
            weight.get(f"model.layers.{layer_id}.post_attention_layernorm.weight"),
            eps=args.norm_eps
        )

    def __call__(self, x: Array["B, L or 1, D"], start_pos: int, mask: Array["L, L"],
                 freqs_cos: Array["L or 1, HD//2"], freqs_sin: Array["L or 1, HD//2"]):
        # RMSNorm
        norm_x: Array["B, L or 1, D"] = self.input_layernorm(x)
        # Masked Multi-Head Attention
        h1: Array["B, L or 1, D"] = self.attention(norm_x, start_pos, mask, freqs_cos, freqs_sin)
        z = x + h1

        # RMSNorm
        norm_z = self.post_attention_layernorm(z)
        # Feed Forward + SwiGLU
        h2: Array["B, L or 1, D"] = self.feed_forward(norm_z)
        out = z + h2

        return out


class Llama:
    def __init__(self, model_path: str, args: ModelArgs):
        self.args = args

        weight = load_parameters(model_path)
        self.tok_embedding: Array["VS, D"] = weight.get("model.embed_tokens.weight")

        # RoPE #1
        self.freqs_cos, self.freqs_sin = compute_cos_sin_cache(args.dim // args.n_heads, args.max_seq_len)

        self.layers = []
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(weight, layer_id, args))

        self.norm = RMSNorm(weight.get("model.norm.weight"), eps=args.norm_eps)
        self.lm_head_weight: Array["D, VS"] = weight.get("lm_head.weight").T

        del weight

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
        logit: Array["B, 1, VS"] = h[:, [-1], :] @ self.lm_head_weight
        return logit

    def generate(self, input_ids: Array["B, L"], max_new_tokens: int):
        _, L = input_ids.shape
        for i, curr_pos in enumerate(range(L, max_new_tokens)):
            if i == 0:  # Prefill Phase
                inputs = input_ids
                pos = 0
            else:  # Decode Phase
                inputs = next_id
                pos = curr_pos
            logits: Array["B, 1, VS"] = self(inputs, pos)
            next_id = logits[:, -1, :].argmax(-1, keepdims=True)
            yield next_id


if __name__ == '__main__':
    args = ModelArgs()

    print(f"Starting LLaMA run with configured ModelArgs.dtype: {args.dtype}")

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
    for id in model.generate(input_ids, args.max_new_tokens):
        L += 1
        output_id = id[0].tolist()
        if output_id[-1] in [tokenizer.eos_id, tokenizer.bos_id]:
            break
        print(tokenizer.decode(output_id), end="")
        sys.stdout.flush()
    elapsed = time.time() - start
    print(f"\n\nToken count: {L}, elapsed: {elapsed:.2f}s, {round(L / elapsed)} tokens/s")
