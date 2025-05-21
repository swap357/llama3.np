import numpy as np
import pytest

from config import ModelArgs
from llama3 import Attention, FeedForward, RMSNorm, TransformerBlock
from llama3 import apply_rotary_emb as apply_rotary_emb_typed
from llama3 import compute_cos_sin_cache as compute_cos_sin_cache_typed
from llama3 import repeat_kv as repeat_kv_typed
from llama3 import silu as silu_typed
from llama3 import softmax as softmax_typed
from llama3_simple import apply_rotary_emb as apply_rotary_emb_simple
from llama3_simple import attention as attention_simple
from llama3_simple import compute_cos_sin_cache as compute_cos_sin_cache_simple
from llama3_simple import feed_forward as feed_forward_simple
from llama3_simple import repeat_kv as repeat_kv_simple
from llama3_simple import rmsnorm as rmsnorm_simple
from llama3_simple import silu as silu_simple
from llama3_simple import softmax as softmax_simple
from llama3_simple import transformer_block as transformer_block_simple

# Set random seed for reproducibility
np.random.seed(42)


def test_softmax_equivalence():
    # Create random input
    x = np.random.randn(2, 3, 4).astype(np.float32)

    # Get outputs from both implementations
    out_typed = softmax_typed(x)
    out_simple = softmax_simple(x)

    # Check if outputs are equal
    np.testing.assert_allclose(out_typed, out_simple, rtol=1e-5, atol=1e-5)


def test_silu_equivalence():
    # Create random input
    x = np.random.randn(2, 3, 4).astype(np.float32)

    # Get outputs from both implementations
    out_typed = silu_typed(x)
    out_simple = silu_simple(x)

    # Check if outputs are equal
    np.testing.assert_allclose(out_typed, out_simple, rtol=1e-5, atol=1e-5)


def test_compute_cos_sin_cache_equivalence():
    # Test parameters
    head_dim = 64
    max_seq_len = 128
    base = 10000
    dtype = np.float32

    # Get outputs from both implementations
    cos_typed, sin_typed = compute_cos_sin_cache_typed(
        head_dim, max_seq_len, base, dtype
    )
    cos_simple, sin_simple = compute_cos_sin_cache_simple(
        head_dim, max_seq_len, base, dtype
    )

    # Check if outputs are equal
    np.testing.assert_allclose(cos_typed, cos_simple, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(sin_typed, sin_simple, rtol=1e-5, atol=1e-5)


def test_apply_rotary_emb_equivalence():
    # Create random inputs
    batch_size = 2
    seq_len = 4
    n_heads = 8
    head_dim = 64

    xq = np.random.randn(batch_size, seq_len, n_heads, head_dim).astype(np.float32)
    xk = np.random.randn(batch_size, seq_len, n_heads, head_dim).astype(np.float32)
    freqs_cos = np.random.randn(seq_len, head_dim // 2).astype(np.float32)
    freqs_sin = np.random.randn(seq_len, head_dim // 2).astype(np.float32)

    # Get outputs from both implementations
    xq_out_typed, xk_out_typed = apply_rotary_emb_typed(xq, xk, freqs_cos, freqs_sin)
    xq_out_simple, xk_out_simple = apply_rotary_emb_simple(xq, xk, freqs_cos, freqs_sin)

    # Check if outputs are equal
    np.testing.assert_allclose(xq_out_typed, xq_out_simple, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(xk_out_typed, xk_out_simple, rtol=1e-5, atol=1e-5)


def test_repeat_kv_equivalence():
    # Create random input
    batch_size = 2
    seq_len = 4
    n_kv_heads = 4
    head_dim = 64
    n_rep = 2

    x = np.random.randn(batch_size, seq_len, n_kv_heads, head_dim).astype(np.float32)

    # Get outputs from both implementations
    out_typed = repeat_kv_typed(x, n_rep)
    out_simple = repeat_kv_simple(x, n_rep)

    # Check if outputs are equal
    np.testing.assert_allclose(out_typed, out_simple, rtol=1e-5, atol=1e-5)


def test_feed_forward_equivalence():
    # Create random inputs
    batch_size = 2
    seq_len = 4
    dim = 64
    ffn_dim = 128
    dtype = np.float32

    x = np.random.randn(batch_size, seq_len, dim).astype(dtype)
    up_weight = np.random.randn(ffn_dim, dim).astype(dtype)
    gate_weight = np.random.randn(ffn_dim, dim).astype(dtype)
    down_weight = np.random.randn(dim, ffn_dim).astype(dtype)

    # Get outputs from both implementations
    ff_typed = FeedForward(up_weight, gate_weight, down_weight, dtype)
    out_typed = ff_typed(x)
    out_simple = feed_forward_simple(x, up_weight, gate_weight, down_weight, dtype)

    # Check if outputs are equal
    np.testing.assert_allclose(out_typed, out_simple, rtol=1e-5, atol=1e-5)


def test_rmsnorm_equivalence():
    # Create random inputs
    batch_size = 2
    seq_len = 4
    dim = 64
    dtype = np.float32
    eps = 1e-6

    x = np.random.randn(batch_size, seq_len, dim).astype(dtype)
    weight = np.random.randn(dim).astype(dtype)

    # Get outputs from both implementations
    norm_typed = RMSNorm(weight, eps, dtype)
    out_typed = norm_typed(x)
    out_simple = rmsnorm_simple(x, weight, eps, dtype)

    # Check if outputs are equal
    np.testing.assert_allclose(out_typed, out_simple, rtol=1e-5, atol=1e-5)


def test_attention_equivalence():
    # Create random inputs
    batch_size = 2
    seq_len = 4
    dim = 64
    n_heads = 8
    dtype = np.float32

    x = np.random.randn(batch_size, seq_len, dim).astype(dtype)
    q_weight = np.random.randn(dim, dim).astype(dtype)
    k_weight = np.random.randn(dim, dim).astype(dtype)
    v_weight = np.random.randn(dim, dim).astype(dtype)
    o_weight = np.random.randn(dim, dim).astype(dtype)

    # Create args
    args = ModelArgs()
    args.dim = dim
    args.n_heads = n_heads
    args.n_kv_heads = n_heads
    args.max_batch_size = batch_size
    args.max_seq_len = seq_len

    # Create attention mask
    mask = np.triu(np.full((seq_len, seq_len), float("-inf")), k=1)
    mask = np.concatenate([np.zeros((seq_len, 0)), mask], axis=1)

    # Create rotary embeddings
    freqs_cos, freqs_sin = compute_cos_sin_cache_simple(
        dim // n_heads, seq_len, dtype=dtype
    )

    # Get outputs from both implementations
    attn_typed = Attention(q_weight, k_weight, v_weight, o_weight, args, dtype)
    out_typed = attn_typed(x, 0, mask, freqs_cos, freqs_sin)

    cache_k = np.zeros((batch_size, seq_len, n_heads, dim // n_heads), dtype=dtype)
    cache_v = np.zeros((batch_size, seq_len, n_heads, dim // n_heads), dtype=dtype)
    out_simple, _, _ = attention_simple(
        x,
        0,
        mask,
        freqs_cos,
        freqs_sin,
        [q_weight, k_weight, v_weight, o_weight],
        args,
        cache_k,
        cache_v,
        dtype,
    )

    # Check if outputs are equal
    np.testing.assert_allclose(out_typed, out_simple, rtol=1e-5, atol=1e-5)


def test_transformer_block_equivalence():
    # Create random inputs
    batch_size = 2
    seq_len = 4
    dim = 64
    n_heads = 8
    dtype = np.float32

    x = np.random.randn(batch_size, seq_len, dim).astype(dtype)

    # Create weights dictionary
    weights = {
        "model.layers.0.self_attn.q_proj.weight": np.random.randn(dim, dim).astype(
            dtype
        ),
        "model.layers.0.self_attn.k_proj.weight": np.random.randn(dim, dim).astype(
            dtype
        ),
        "model.layers.0.self_attn.v_proj.weight": np.random.randn(dim, dim).astype(
            dtype
        ),
        "model.layers.0.self_attn.o_proj.weight": np.random.randn(dim, dim).astype(
            dtype
        ),
        "model.layers.0.mlp.up_proj.weight": np.random.randn(dim * 4, dim).astype(
            dtype
        ),
        "model.layers.0.mlp.gate_proj.weight": np.random.randn(dim * 4, dim).astype(
            dtype
        ),
        "model.layers.0.mlp.down_proj.weight": np.random.randn(dim, dim * 4).astype(
            dtype
        ),
        "model.layers.0.input_layernorm.weight": np.random.randn(dim).astype(dtype),
        "model.layers.0.post_attention_layernorm.weight": np.random.randn(dim).astype(
            dtype
        ),
    }

    # Create args
    args = ModelArgs()
    args.dim = dim
    args.n_heads = n_heads
    args.n_kv_heads = n_heads
    args.max_batch_size = batch_size
    args.max_seq_len = seq_len
    args.norm_eps = 1e-6

    # Create attention mask
    mask = np.triu(np.full((seq_len, seq_len), float("-inf")), k=1)
    mask = np.concatenate([np.zeros((seq_len, 0)), mask], axis=1)

    # Create rotary embeddings
    freqs_cos, freqs_sin = compute_cos_sin_cache_simple(
        dim // n_heads, seq_len, dtype=dtype
    )

    # Get outputs from both implementations
    block_typed = TransformerBlock(weights, 0, args)
    out_typed = block_typed(x, 0, mask, freqs_cos, freqs_sin)

    cache_k = np.zeros((batch_size, seq_len, n_heads, dim // n_heads), dtype=dtype)
    cache_v = np.zeros((batch_size, seq_len, n_heads, dim // n_heads), dtype=dtype)
    block_weights = [
        [
            weights["model.layers.0.self_attn.q_proj.weight"],
            weights["model.layers.0.self_attn.k_proj.weight"],
            weights["model.layers.0.self_attn.v_proj.weight"],
            weights["model.layers.0.self_attn.o_proj.weight"],
        ],
        [
            weights["model.layers.0.mlp.up_proj.weight"],
            weights["model.layers.0.mlp.gate_proj.weight"],
            weights["model.layers.0.mlp.down_proj.weight"],
        ],
        weights["model.layers.0.input_layernorm.weight"],
        weights["model.layers.0.post_attention_layernorm.weight"],
    ]
    out_simple, _, _ = transformer_block_simple(
        x, 0, mask, freqs_cos, freqs_sin, block_weights, args, cache_k, cache_v, dtype
    )

    # Check if outputs are equal
    np.testing.assert_allclose(out_typed, out_simple, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
