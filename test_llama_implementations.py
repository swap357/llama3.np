import numpy as np
import pytest
from numpy.testing import assert_allclose

import llama3 as llama_oop
import llama3_simple as llama_functional
from config import ModelArgs

# Set random seed for reproducibility
np.random.seed(42)

# Get default model configuration
config = ModelArgs()

# Test parameters using config values
BATCH_SIZE = config.max_batch_size
SEQ_LEN = 8  # Small enough for testing
DIM = config.dim
N_HEADS = config.n_heads
N_KV_HEADS = config.n_heads if config.n_kv_heads is None else config.n_kv_heads
HEAD_DIM = config.dim // config.n_heads
MAX_SEQ_LEN = config.max_seq_len
ATOL = 1e-4
RTOL = 2e-4


@pytest.fixture
def random_input():
    return np.random.randn(BATCH_SIZE, SEQ_LEN, DIM).astype(np.float32)


@pytest.fixture
def model_args():
    args = ModelArgs()
    args.dim = DIM
    args.n_heads = N_HEADS
    args.n_kv_heads = N_KV_HEADS
    args.max_seq_len = MAX_SEQ_LEN
    args.max_batch_size = BATCH_SIZE
    return args


def test_softmax():
    x = np.random.randn(BATCH_SIZE, N_HEADS, SEQ_LEN, SEQ_LEN).astype(np.float32)
    result_oop = llama_oop.softmax(x)
    result_functional = llama_functional.softmax(x)
    assert(result_oop == result_functional).all()


def test_silu():
    x = np.random.randn(BATCH_SIZE, SEQ_LEN, DIM).astype(np.float32)
    result_oop = llama_oop.silu(x)
    result_functional = llama_functional.silu(x)
    assert(result_oop == result_functional).all()


def test_compute_cos_sin_cache():
    result_oop_cos, result_oop_sin = llama_oop.compute_cos_sin_cache(
        HEAD_DIM, MAX_SEQ_LEN
    )
    result_functional_cos, result_functional_sin = (
        llama_functional.compute_cos_sin_cache(HEAD_DIM, MAX_SEQ_LEN)
    )
    assert_allclose(result_oop_cos, result_functional_cos, rtol=RTOL, atol=ATOL)
    assert_allclose(result_oop_sin, result_functional_sin, rtol=RTOL, atol=ATOL)


def test_apply_rotary_emb():
    # Prepare inputs
    xq = np.random.randn(BATCH_SIZE, SEQ_LEN, N_HEADS, HEAD_DIM).astype(np.float32)
    xk = np.random.randn(BATCH_SIZE, SEQ_LEN, N_KV_HEADS, HEAD_DIM).astype(np.float32)
    cos, sin = llama_functional.compute_cos_sin_cache(HEAD_DIM, MAX_SEQ_LEN)
    freqs_cos = cos[:SEQ_LEN]
    freqs_sin = sin[:SEQ_LEN]

    # Run both implementations
    result_oop_q, result_oop_k = llama_oop.apply_rotary_emb(
        xq, xk, freqs_cos, freqs_sin
    )
    result_functional_q, result_functional_k = llama_functional.apply_rotary_emb(
        xq, xk, freqs_cos, freqs_sin
    )

    assert(result_oop_q == result_functional_q).all()
    assert(result_oop_k == result_functional_k).all()

def test_rmsnorm(random_input, model_args):
    weight = np.random.randn(DIM).astype(np.float32)

    # Create RMSNorm instances
    rmsnorm_oop = llama_oop.RMSNorm(weight, model_args.norm_eps)

    # Test the implementations
    result_oop = rmsnorm_oop(random_input)
    result_functional = llama_functional.rmsnorm(
        random_input, weight, model_args.norm_eps
    )
    assert(result_oop == result_functional).all()


def test_full_model_forward():
    # Initialize both models with same random weights
    args = ModelArgs()
    model_oop = llama_oop.Llama("./stories15M.model.npz", args)
    model_functional = llama_functional.llama_init("./stories15M.model.npz", args)

    # Create random input with smaller sequence length
    input_ids = np.random.randint(0, 100, size=(1, 4), dtype=np.int32)
    start_pos = 0

    # Run full forward pass
    logits_oop = model_oop(input_ids, start_pos)
    logits_functional = llama_functional.llama_forward(
        model_functional, input_ids, start_pos
    )

    # Compare shapes first
    assert logits_oop.shape == logits_functional.shape == (1, 1, args.vocab_size)

    # Compare the actual values with detailed statistics
    print("\nDetailed logits comparison:")

    # Basic statistics
    diff = np.abs(logits_oop - logits_functional)
    rel_diff = np.abs(diff / (np.abs(logits_functional) + 1e-9))

    print(f"Shape: {logits_oop.shape}")
    print(f"Absolute differences:")
    print(f"  Max: {np.max(diff):.2e}")
    print(f"  Mean: {np.mean(diff):.2e}")
    print(f"  Median: {np.median(diff):.2e}")
    print(f"  Std: {np.std(diff):.2e}")
    print(f"\nRelative differences:")
    print(f"  Max: {np.max(rel_diff):.2e}")
    print(f"  Mean: {np.mean(rel_diff):.2e}")
    print(f"  Median: {np.median(rel_diff):.2e}")
    print(f"  Std: {np.std(rel_diff):.2e}")

    # Distribution of differences
    percentiles = [50, 75, 90, 95, 99, 99.9]
    print("\nPercentiles of absolute differences:")
    for p in percentiles:
        print(f"  {p}th: {np.percentile(diff, p):.2e}")

    # Count elements exceeding various thresholds
    thresholds = [1e-6, 1e-5, 1e-4, 1e-3]
    print("\nElements exceeding thresholds:")
    for t in thresholds:
        count = np.sum(diff > t)
        percent = count / diff.size * 100
        print(f"  > {t:.0e}: {count} elements ({percent:.1f}%)")

    # Compare top-k predictions
    k = 5
    top_k_oop = np.argsort(logits_oop[0, 0])[-k:][::-1]
    top_k_func = np.argsort(logits_functional[0, 0])[-k:][::-1]
    print(f"\nTop {k} predictions match: {np.array_equal(top_k_oop, top_k_func)}")
    if not np.array_equal(top_k_oop, top_k_func):
        print("OOP top-k:", top_k_oop)
        print("Functional top-k:", top_k_func)

    # Verify the differences don't affect the model's predictions
    assert np.array_equal(top_k_oop, top_k_func), (
        "Top-k predictions differ between implementations"
    )
    assert_allclose(logits_oop, logits_functional, rtol=RTOL, atol=ATOL)


if __name__ == "__main__":
    pytest.main([__file__])
