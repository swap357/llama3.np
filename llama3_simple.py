import logging
import math
import os
import sys
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler

import numpy as np

from config import ModelArgs
from tokenizer import Tokenizer
from utils import load_parameters

# Create debug directory if it doesn't exist
debug_dir = os.path.join("output", "debug")
os.makedirs(debug_dir, exist_ok=True)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create formatters
debug_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Create timestamped log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
debug_file = os.path.join(debug_dir, f"llama3_debug_{timestamp}.log")

# File handler for debug logs
file_handler = RotatingFileHandler(
    debug_file,
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5,
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(debug_formatter)

# Add only file handler to logger (no console output for debug logs)
logger.addHandler(file_handler)

# Log the start of a new run
logger.debug(f"Starting new run at {timestamp}")

# Create a separate logger for stdout output
stdout_logger = logging.getLogger("stdout")
stdout_logger.setLevel(logging.INFO)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_logger.addHandler(stdout_handler)

np.random.seed(42)


def softmax(x):
    logger.debug(f"softmax input dtype: {x.dtype}")
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    result = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    logger.debug(f"softmax output dtype: {result.dtype}")
    return result


def silu(x):
    logger.debug(f"silu input dtype: {x.dtype}")
    result = x * (1 / (1 + np.exp(-x)))
    logger.debug(f"silu output dtype: {result.dtype}")
    return result


def compute_cos_sin_cache(head_dim, max_seq_len, base=10000, dtype=np.float32):
    logger.debug(f"compute_cos_sin_cache input dtype: {dtype}")
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2)[: (head_dim // 2)] / head_dim))
    t = np.arange(max_seq_len)
    freqs = np.outer(t, inv_freq)
    cos_result = np.cos(freqs)
    sin_result = np.sin(freqs)
    logger.debug(
        f"compute_cos_sin_cache output dtypes: cos={cos_result.dtype}, sin={sin_result.dtype}"
    )
    return cos_result, sin_result


def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    logger.debug(
        f"apply_rotary_emb input dtypes: xq={xq.dtype}, xk={xk.dtype}, freqs_cos={freqs_cos.dtype}, freqs_sin={freqs_sin.dtype}"
    )
    xqri = xq.reshape(xq.shape[:-1] + (-1, 2))
    xkri = xk.reshape(xk.shape[:-1] + (-1, 2))
    xq_r, xq_i = np.split(xqri, 2, axis=-1)
    xq_r = xq_r.squeeze(-1)
    xq_i = xq_i.squeeze(-1)
    xk_r, xk_i = np.split(xkri, 2, axis=-1)
    xk_r = xk_r.squeeze(-1)
    xk_i = xk_i.squeeze(-1)
    freqs_cos = np.expand_dims(freqs_cos, axis=(0, 2))
    freqs_sin = np.expand_dims(freqs_sin, axis=(0, 2))
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos
    xq_out = np.stack([xq_out_r, xq_out_i], axis=-1).reshape(
        xq_out_r.shape[:-1] + (-1,)
    )
    xk_out = np.stack([xk_out_r, xk_out_i], axis=-1).reshape(
        xk_out_r.shape[:-1] + (-1,)
    )
    logger.debug(
        f"apply_rotary_emb output dtypes: xq_out={xq_out.dtype}, xk_out={xk_out.dtype}"
    )
    return xq_out, xk_out


def repeat_kv(x, n_rep):
    logger.debug(f"repeat_kv input dtype: {x.dtype}")
    if n_rep == 1:
        return x
    result = np.repeat(x, n_rep, axis=2)
    logger.debug(f"repeat_kv output dtype: {result.dtype}")
    return result


def feed_forward(x, up_weight, gate_weight, down_weight, dtype):
    logger.debug(
        f"feed_forward input dtypes: x={x.dtype}, up_weight={up_weight.dtype}, gate_weight={gate_weight.dtype}, down_weight={down_weight.dtype}"
    )
    swish = silu(x @ gate_weight.T)
    x_v = x @ up_weight.T
    x_ff = swish * x_v
    x_out = x_ff @ down_weight.T
    logger.debug(f"feed_forward output dtype: {x_out.dtype}")
    return x_out


def rmsnorm(x, weight, eps, dtype):
    logger.debug(f"rmsnorm input dtypes: x={x.dtype}, weight={weight.dtype}")
    z_float = (x**2).mean(-1, keepdims=True) + eps
    z = x / np.sqrt(z_float)
    result = z * weight
    logger.debug(f"rmsnorm output dtype: {result.dtype}")
    return result


def attention(
    x,
    start_pos,
    mask,
    freqs_cos,
    freqs_sin,
    attn_weights,
    args,
    cache_k,
    cache_v,
    dtype,
):
    logger.debug(
        f"attention input dtypes: x={x.dtype}, mask={mask.dtype if mask is not None else None}, freqs_cos={freqs_cos.dtype}, freqs_sin={freqs_sin.dtype}"
    )
    q_weight, k_weight, v_weight, o_weight = [w.T for w in attn_weights]
    logger.debug(
        f"attention weights dtypes: q={q_weight.dtype}, k={k_weight.dtype}, v={v_weight.dtype}, o={o_weight.dtype}"
    )

    n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
    assert args.n_heads % n_kv_heads == 0
    n_local_heads = args.n_heads
    n_local_kv_heads = n_kv_heads
    n_rep = n_local_heads // n_local_kv_heads
    head_dim = args.dim // args.n_heads

    batch_size, seq_len, _ = x.shape

    xq = x @ q_weight
    xk = x @ k_weight
    xv = x @ v_weight

    xq = xq.reshape(batch_size, seq_len, n_local_heads, head_dim)
    xk = xk.reshape(batch_size, seq_len, n_local_kv_heads, head_dim)
    xv = xv.reshape(batch_size, seq_len, n_local_kv_heads, head_dim)
    logger.debug(
        f"attention intermediate dtypes: xq={xq.dtype}, xk={xk.dtype}, xv={xv.dtype}"
    )

    xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

    cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
    cache_v[:batch_size, start_pos : start_pos + seq_len] = xv
    ks = cache_k[:batch_size, : start_pos + seq_len]
    vs = cache_v[:batch_size, : start_pos + seq_len]

    xk = repeat_kv(ks, n_rep)
    xv = repeat_kv(vs, n_rep)

    xq = xq.transpose(0, 2, 1, 3)
    xk = xk.transpose(0, 2, 1, 3)
    xv = xv.transpose(0, 2, 1, 3)

    attention_scores = (xq @ xk.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
    logger.debug(f"attention_scores dtype: {attention_scores.dtype}")
    if mask is not None:
        attention_scores = attention_scores + mask[None, None, :, :]
    logger.debug(f"attention_scores dtype after mask: {attention_scores.dtype}")
    attn = softmax(attention_scores)
    logger.debug(f"attn dtype after softmax: {attn.dtype}")
    output = attn @ xv
    output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
    output = output @ o_weight
    logger.debug(f"attention output dtype: {output.dtype}")
    return output, cache_k, cache_v


def transformer_block(
    x,
    start_pos,
    mask,
    freqs_cos,
    freqs_sin,
    block_weights,
    args,
    cache_k,
    cache_v,
    dtype,
):
    logger.debug(f"transformer_block input dtype: x={x.dtype}")
    attn_weights, ff_weights, in_norm_weight, post_norm_weight = block_weights
    logger.debug(
        f"transformer_block weights dtypes: in_norm={in_norm_weight.dtype}, post_norm={post_norm_weight.dtype}"
    )

    norm_x = rmsnorm(x, in_norm_weight, args.norm_eps, dtype)
    h1, cache_k, cache_v = attention(
        norm_x,
        start_pos,
        mask,
        freqs_cos,
        freqs_sin,
        attn_weights,
        args,
        cache_k,
        cache_v,
        dtype,
    )
    z = x + h1
    norm_z = rmsnorm(z, post_norm_weight, args.norm_eps, dtype)
    h2 = feed_forward(norm_z, *ff_weights, dtype)
    out = z + h2
    logger.debug(f"transformer_block output dtype: {out.dtype}")
    return out, cache_k, cache_v


def llama_init(model_path, args):
    logger.debug(f"llama_init input dtype: {args.dtype}")
    dtype = getattr(np, args.dtype)

    # Load and convert all weights to specified dtype immediately
    weights = load_parameters(model_path)
    weights = {k: v.astype(dtype) for k, v in weights.items()}
    logger.debug(f"All weights converted to {dtype}")

    tok_embedding = weights["model.embed_tokens.weight"]  # Already in correct dtype
    logger.debug(f"llama_init embedding dtype: {tok_embedding.dtype}")

    freqs_cos, freqs_sin = compute_cos_sin_cache(
        args.dim // args.n_heads, args.max_seq_len, dtype=dtype
    )
    layer_blocks = []
    for layer_id in range(args.n_layers):
        attn_weights = [
            weights[f"model.layers.{layer_id}.self_attn.q_proj.weight"],
            weights[f"model.layers.{layer_id}.self_attn.k_proj.weight"],
            weights[f"model.layers.{layer_id}.self_attn.v_proj.weight"],
            weights[f"model.layers.{layer_id}.self_attn.o_proj.weight"],
        ]
        ff_weights = [
            weights[f"model.layers.{layer_id}.mlp.up_proj.weight"],
            weights[f"model.layers.{layer_id}.mlp.gate_proj.weight"],
            weights[f"model.layers.{layer_id}.mlp.down_proj.weight"],
        ]
        in_norm = weights[f"model.layers.{layer_id}.input_layernorm.weight"]
        post_norm = weights[f"model.layers.{layer_id}.post_attention_layernorm.weight"]
        layer_blocks.append((attn_weights, ff_weights, in_norm, post_norm))
    norm_weight = weights["model.norm.weight"]
    lm_head_weight = weights["lm_head.weight"].T  # Already in correct dtype
    logger.debug(
        f"llama_init final weights dtypes: norm={norm_weight.dtype}, lm_head={lm_head_weight.dtype}"
    )
    del weights

    # Preallocate caches for all layers (list of np.arrays)
    caches_k = [
        np.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                args.n_kv_heads if args.n_kv_heads else args.n_heads,
                args.dim // args.n_heads,
            ),
            dtype=dtype,
        )
        for _ in range(args.n_layers)
    ]
    caches_v = [np.zeros_like(caches_k[0]) for _ in range(args.n_layers)]
    logger.debug(
        f"llama_init cache dtypes: k={caches_k[0].dtype}, v={caches_v[0].dtype}"
    )

    return {
        "args": args,
        "dtype": dtype,
        "tok_embedding": tok_embedding,
        "freqs_cos": freqs_cos,
        "freqs_sin": freqs_sin,
        "layer_blocks": layer_blocks,
        "norm_weight": norm_weight,
        "lm_head_weight": lm_head_weight,
        "caches_k": caches_k,
        "caches_v": caches_v,
    }


def llama_forward(model, input_ids, start_pos):
    args = model["args"]
    dtype = model["dtype"]
    logger.debug(
        f"llama_forward input dtypes: input_ids={input_ids.dtype}, dtype={dtype}"
    )

    _, seq_len = input_ids.shape
    h = model["tok_embedding"][input_ids]
    logger.debug(f"llama_forward embedding output dtype: {h.dtype}")

    freqs_cos = model["freqs_cos"][start_pos : start_pos + seq_len]
    freqs_sin = model["freqs_sin"][start_pos : start_pos + seq_len]

    mask = None
    if seq_len > 1:
        mask = np.full((seq_len, seq_len), float("-inf"), dtype=dtype)
        mask = np.triu(mask, k=1)
        zeros_shape = (seq_len, start_pos)
        mask = np.concatenate([np.zeros(zeros_shape, dtype=dtype), mask], axis=1)
        logger.debug(f"llama_forward mask dtype: {mask.dtype}")

    caches_k = model["caches_k"]
    caches_v = model["caches_v"]

    for i, block in enumerate(model["layer_blocks"]):
        h, caches_k[i], caches_v[i] = transformer_block(
            h,
            start_pos,
            mask,
            freqs_cos,
            freqs_sin,
            block,
            args,
            caches_k[i],
            caches_v[i],
            dtype,
        )

    h = rmsnorm(h, model["norm_weight"], args.norm_eps, dtype)
    logit = h[:, [-1], :] @ model["lm_head_weight"]
    logger.debug(f"llama_forward final output dtype: {logit.dtype}")
    return logit


def llama_generate(model, input_ids, max_new_tokens):
    batch_size, prompt_len = input_ids.shape
    current_len = prompt_len
    next_id = None  # Initialize next_id to avoid undefined variable error
    for i in range(max_new_tokens):
        current_pos = prompt_len + i
        if i == 0:
            current_input_ids = input_ids
            pos = 0
        else:
            current_input_ids = next_id
            pos = current_pos - 1
        logits = llama_forward(model, current_input_ids, pos)
        next_id = logits[:, -1, :].argmax(-1, keepdims=True).astype(np.int32)
        yield next_id
        current_len += 1
        if current_len >= model["args"].max_seq_len:
            break

    elapsed = time.time() - start
    print(
        f"\n\nToken count: {seq_len}, elapsed: {elapsed:.2f}s, {round(seq_len / elapsed)} tokens/s"
    )


# -- Main script
if __name__ == "__main__":
    args = ModelArgs()
    print(f"Using precision: {args.dtype}")
    tokenizer = Tokenizer("./tokenizer.model.np")
    model = llama_init("./stories15M.model.npz", args)

    if len(sys.argv) == 1:
        prompt = "Once upon a time"
    else:
        prompt = sys.argv[1]
    print(f"\n{prompt}", end="")
    input_ids = np.array([tokenizer.encode(prompt)])
    start = time.time()
    _, seq_len = input_ids.shape
    generated_tokens_count = 0
    for id_val in llama_generate(model, input_ids, args.max_new_tokens):
        seq_len += 1
        generated_tokens_count += 1
        output_id = id_val[0].tolist()
        if output_id[-1] in [tokenizer.eos_id, tokenizer.bos_id]:
            break
        print(tokenizer.decode(output_id), end="", flush=True)
