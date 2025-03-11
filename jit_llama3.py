import numpy as np
import time
import sys
import math
import cProfile
import pstats
from numba import njit, jit, config, typeof
from numba.core.registry import CPUDispatcher
import inspect
from config import ModelArgs
from tokenizer import Tokenizer
from utils import load_parameters

# Numba configuration
config.NUMBA_DEBUG_PRINT_AFTER_COMPILE = False  # Set to True for debugging

# Force float32 precision for all calculations
USE_FLOAT32 = True  # Set to False to use mixed precision

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

@njit
def softmax_jit(x_2d):
    """Optimized 2D softmax for Numba JIT"""
    # Manual implementation without using np.max with axis
    x_max = np.zeros(x_2d.shape[0])
    for i in range(x_2d.shape[0]):
        x_max[i] = x_2d[i].max()
    
    # Manual exp and normalization
    e_x = np.zeros_like(x_2d)
    for i in range(x_2d.shape[0]):
        for j in range(x_2d.shape[1]):
            e_x[i, j] = np.exp(x_2d[i, j] - x_max[i])
    
    # Manual sum for normalization
    sums = np.zeros(x_2d.shape[0])
    for i in range(x_2d.shape[0]):
        sums[i] = 0.0
        for j in range(x_2d.shape[1]):
            sums[i] += e_x[i, j]
    
    # Normalize
    for i in range(x_2d.shape[0]):
        for j in range(x_2d.shape[1]):
            e_x[i, j] /= sums[i]
    
    return e_x

def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))

@njit
def silu_jit(x):
    """SiLU activation function optimized for JIT, compatible with float32 or float64"""
    # This implementation will work with any floating point type
    result = np.zeros_like(x)
    for i in range(x.size):
        flat_idx = i
        val = x.flat[flat_idx]
        result.flat[flat_idx] = val * (1.0 / (1.0 + np.exp(-val)))
    return result

def compute_cos_sin_cache(head_dim, max_seq_len, base=10000):
    """Compute cosine and sine frequency cache for rotary embeddings"""
    dtype = np.float32 if USE_FLOAT32 else np.float64
    
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)] / head_dim))
    t = np.arange(max_seq_len, dtype=dtype)
    freqs = np.outer(t, inv_freq)
    
    return np.cos(freqs).astype(dtype), np.sin(freqs).astype(dtype)

# Don't use JIT for matrix multiplication - NumPy is better
def matmul_jit(a, b):
    """Matrix multiplication - uses NumPy's optimized implementation"""
    return a @ b

@njit
def layer_norm_jit(x, weight, eps):
    """JIT-optimized RMS norm calculation without numpy functions that don't work in nopython mode"""
    # Compute variance manually
    variance = np.zeros((x.shape[0], 1))
    for i in range(x.shape[0]):
        sum_squared = 0.0
        for j in range(x.shape[1]):
            sum_squared += x[i, j] * x[i, j]
        variance[i, 0] = sum_squared / x.shape[1]
    
    # Apply normalization
    result = np.zeros_like(x)
    for i in range(x.shape[0]):
        norm_factor = 1.0 / np.sqrt(variance[i, 0] + eps)
        for j in range(x.shape[1]):
            result[i, j] = x[i, j] * norm_factor * weight[j]
    
    return result

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
        # Try using JIT version when shape is compatible
        try:
            if x.ndim == 2 and x.dtype == np.float64:
                return layer_norm_jit(x, self.weight, self.eps)
            else:
                # Fall back to numpy version
                variance = np.mean(x**2, axis=-1, keepdims=True)
                x_norm = x / np.sqrt(variance + self.eps)
                return x_norm * self.weight
        except Exception as e:
            print(f"JIT Error in RMSNorm: {e}")
            # Fall back to numpy version
            variance = np.mean(x**2, axis=-1, keepdims=True)
            x_norm = x / np.sqrt(variance + self.eps)
            return x_norm * self.weight

class FeedForward:
    def __init__(self, up_weight, gate_weight, down_weight):
        self.up_weight = up_weight.T
        self.gate_weight = gate_weight.T
        self.down_weight = down_weight.T
        
    def __call__(self, x):
        # Get gate projection
        gate_proj = x @ self.gate_weight
        
        # Apply SiLU activation - use JIT for larger matrices
        if gate_proj.size > 32768:  # Use JIT for matrices larger than 32K elements
            try:
                swish = silu_jit(gate_proj)
            except:
                swish = silu(gate_proj)
        else:
            swish = silu(gate_proj)
        
        # Complete the feed-forward computation
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
        
        # Use float32 to match the model parameters
        self.cache_k = np.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), dtype=np.float32)
        self.cache_v = np.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim), dtype=np.float32)
        
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

def profile_function(func, *args, **kwargs):
    """Profile a function execution and return timing info"""
    iterations = 100
    
    # Warm up
    result = func(*args, **kwargs)
    
    # Time it
    start = time.time()
    for _ in range(iterations):
        func(*args, **kwargs)
    elapsed = time.time() - start
    
    return result, elapsed / iterations

def compare_jit_vs_standard():
    """Compare JIT vs standard implementations of critical functions"""
    print("\n==== JIT vs STANDARD PERFORMANCE ====")
    
    # Create test data
    test_data_1d = np.random.rand(1024).astype(np.float64)
    test_data_2d = np.random.rand(32, 1024).astype(np.float64)
    
    # Compare SiLU
    print("\nTesting SiLU:")
    _, standard_time = profile_function(silu, test_data_2d)
    _, jit_time = profile_function(silu_jit_2d, test_data_2d)
    speedup = standard_time / jit_time if jit_time > 0 else 0
    print(f"  Standard: {standard_time*1000:.4f}ms")
    print(f"  JIT:      {jit_time*1000:.4f}ms")
    print(f"  Speedup:  {speedup:.2f}x")
    
    # Test matrix multiplication
    print("\nTesting Matrix Multiplication:")
    A = np.random.rand(32, 64).astype(np.float64)
    B = np.random.rand(64, 32).astype(np.float64)
    _, standard_time = profile_function(lambda a, b: a @ b, A, B)
    _, jit_time = profile_function(matmul_jit, A, B)
    speedup = standard_time / jit_time if jit_time > 0 else 0
    print(f"  Standard: {standard_time*1000:.4f}ms")
    print(f"  JIT:      {jit_time*1000:.4f}ms")
    print(f"  Speedup:  {speedup:.2f}x")
    
    # Test Layer Norm
    print("\nTesting Layer Norm:")
    x = np.random.rand(32, 64).astype(np.float64)
    weight = np.random.rand(64).astype(np.float64)
    eps = 1e-5
    _, standard_time = profile_function(
        lambda x, w, e: x * w / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + e), 
        x, weight, eps
    )
    _, jit_time = profile_function(layer_norm_jit, x, weight, eps)
    speedup = standard_time / jit_time if jit_time > 0 else 0
    print(f"  Standard: {standard_time*1000:.4f}ms")
    print(f"  JIT:      {jit_time*1000:.4f}ms")
    print(f"  Speedup:  {speedup:.2f}x")

def check_jit_status():
    """Print JIT status for all functions in the module"""
    print("\n==== JIT COMPILATION STATUS ====")
    for name, obj in globals().items():
        if isinstance(obj, CPUDispatcher):
            print(f"✅ JIT Compiled: {name}")
            # Display compilation info
            if hasattr(obj, 'inspect_types'):
                print(f"  - Signature: {obj.signatures}")
    
    # Check methods of classes
    for cls_name, cls in [(n, c) for n, c in globals().items() if isinstance(c, type)]:
        print(f"\nClass: {cls_name}")
        for method_name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if hasattr(method, '_is_jitted'):
                print(f"  ✅ JIT method: {method_name}")
            else:
                print(f"  ❌ Non-JIT method: {method_name}")

def profile_operation(model, operation_name="matmul"):
    """Profile a specific operation in the model"""
    print(f"\n==== PROFILING {operation_name.upper()} OPERATION ====")
    
    if operation_name == "matmul":
        # Test matrix multiplication performance
        matrix_sizes = [(32, 64), (64, 128), (128, 256), (256, 512)]
        
        for size in matrix_sizes:
            m, n = size
            k = n  # Make compatible for matmul
            
            # Create test matrices
            a = np.random.randn(m, k).astype(np.float32)
            b = np.random.randn(k, n).astype(np.float32)
            
            # Time numpy version
            start = time.time()
            for _ in range(100):
                _ = a @ b
            numpy_time = time.time() - start
            
            # Time JIT version (if it works with float32)
            try:
                start = time.time()
                for _ in range(100):
                    _ = matmul_jit(a, b)
                jit_time = time.time() - start
                speedup = numpy_time / jit_time
            except Exception as e:
                jit_time = None
                speedup = None
                print(f"Error with JIT matmul: {e}")
            
            print(f"Size {size}: NumPy={numpy_time:.4f}s, JIT={'N/A' if jit_time is None else f'{jit_time:.4f}s'}, Speedup={'N/A' if speedup is None else f'{speedup:.2f}x'}")
    
    elif operation_name == "activation":
        # Test activation function performance
        sizes = [(1, 1024), (32, 1024), (128, 1024), (512, 1024)]
        
        for size in sizes:
            x = np.random.randn(*size).astype(np.float32)
            
            # Time numpy version
            start = time.time()
            for _ in range(100):
                _ = silu(x)
            numpy_time = time.time() - start
            
            # Time JIT version
            try:
                start = time.time()
                for _ in range(100):
                    _ = silu_jit(x)
                jit_time = time.time() - start
                speedup = numpy_time / jit_time
            except Exception as e:
                jit_time = None
                speedup = None
                print(f"Error with JIT silu: {e}")
            
            print(f"Size {size}: NumPy={numpy_time:.4f}s, JIT={'N/A' if jit_time is None else f'{jit_time:.4f}s'}, Speedup={'N/A' if speedup is None else f'{speedup:.2f}x'}")
    
    else:
        print(f"Unknown operation: {operation_name}")

def profile_bottlenecks(model, input_ids):
    """Profile key operations during a single forward pass"""
    # Extract a concrete layer for testing
    layer = model.layers[0]
    
    # Create a simple input with a batch size of 1 and sequence length of 16
    batch_size, seq_len = 1, 16
    if input_ids is not None:
        # Use the actual input for more realistic profiling
        batch_size, seq_len = input_ids.shape
        hidden = model.tok_embedding[input_ids]
    else:
        # Generate random input of the right shape and dtype
        hidden = np.random.randn(batch_size, seq_len, model.args.dim).astype(np.float32)
    
    start_pos = 0
    
    # Timers for each component
    timers = {}
    
    # Time layer normalization
    start = time.time()
    for _ in range(10):
        norm_x = layer.input_layernorm(hidden)
    timers["layernorm"] = (time.time() - start) / 10
    
    # Time attention QKV projection
    start = time.time()
    for _ in range(10):
        xq = hidden @ layer.attention.q_weight
        xk = hidden @ layer.attention.k_weight
        xv = hidden @ layer.attention.v_weight
    timers["qkv_proj"] = (time.time() - start) / 10
    
    # Prepare inputs for attention
    xq = hidden @ layer.attention.q_weight
    xq = xq.reshape(batch_size, seq_len, layer.attention.n_heads, layer.attention.head_dim)
    xk = hidden @ layer.attention.k_weight
    xk = xk.reshape(batch_size, seq_len, layer.attention.n_kv_heads, layer.attention.head_dim)
    
    # Time RoPE
    freqs_cos = model.freqs_cos[start_pos:start_pos + seq_len]
    freqs_sin = model.freqs_sin[start_pos:start_pos + seq_len]
    
    start = time.time()
    for _ in range(10):
        _, _ = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
    timers["rope"] = (time.time() - start) / 10
    
    # Profile overall model forward pass
    start = time.time()
    _ = model.forward(input_ids, 0)
    timers["forward_pass"] = time.time() - start
    
    # Print results
    print("\n==== BOTTLENECK ANALYSIS ====")
    for op, duration in sorted(timers.items(), key=lambda x: x[1], reverse=True):
        print(f"{op}: {duration*1000:.2f}ms")
        
    # Print percentages
    total = timers["forward_pass"]
    print("\n==== PERCENTAGE BREAKDOWN ====")
    for op, duration in sorted(timers.items(), key=lambda x: x[1], reverse=True):
        if op != "forward_pass":
            percentage = (duration / total) * 100
            print(f"{op}: {percentage:.1f}%")
    
    return timers

if __name__ == '__main__':
    # Initialize common components
    args = ModelArgs()
    tokenizer = Tokenizer("./tokenizer.model.np")
    
    # Handle different diagnostic modes
    if len(sys.argv) > 1:
        if sys.argv[1] == "--dtype-check":
            # Just check data types of model components
            model = JitLlama("./stories15M.model.npz", args)
            
            print("==== DATA TYPE INSPECTION ====")
            print(f"Token embedding dtype: {model.tok_embedding.dtype}")
            print(f"LM head dtype: {model.lm_head.dtype}")
            print(f"Freq cos dtype: {model.freqs_cos.dtype}")
            
            # Check first layer parameters
            layer0 = model.layers[0]
            print("\nLayer 0 parameters:")
            print(f"Attention q_weight dtype: {layer0.attention.q_weight.dtype}")
            print(f"Attention k_weight dtype: {layer0.attention.k_weight.dtype}")
            print(f"Feed-forward weights dtype: {layer0.feed_forward.up_weight.dtype}")
            
            # Check cache
            print("\nCache:")
            print(f"KV cache dtype: {layer0.attention.cache_k.dtype}")
            
            # Try to JIT a simple operation with model data
            print("\nJIT compatibility test:")
            try:
                test_input = np.ones((1, 32)).astype(np.float64)
                result = silu_jit(test_input)
                print("✅ JIT works on float64 test data")
            except Exception as e:
                print(f"❌ JIT failed on float64 test data: {e}")
                
            try:
                test_input = np.ones((1, 32)).astype(np.float32)
                result = silu_jit(test_input)
                print("✅ JIT works on float32 test data")
            except Exception as e:
                print(f"❌ JIT failed on float32 test data: {e}")
            
        elif sys.argv[1] == "--profile-bottlenecks":
            # Profile bottlenecks
            model = JitLlama("./stories15M.model.npz", args)
            prompt = "I have a dream" if len(sys.argv) <= 2 else sys.argv[2]
            print(f"Profiling with prompt: {prompt}")
            input_ids = np.array([tokenizer.encode(prompt)])
            profile_bottlenecks(model, input_ids)
            
        elif sys.argv[1] == "--profile-matmul":
            # Profile matrix multiplication
            model = JitLlama("./stories15M.model.npz", args)
            profile_operation(model, "matmul")
            
        elif sys.argv[1] == "--profile-activation":
            # Profile activation function
            model = JitLlama("./stories15M.model.npz", args)
            profile_operation(model, "activation")
            
        elif sys.argv[1] == "--compare":
            # Compare with simple_llama3
            prompt = "I have a dream" if len(sys.argv) <= 2 else sys.argv[2]
            max_tokens = 50  # Generate same number of tokens for fair comparison
            
            # Run our model
            print(f"\nRunning jit_llama3 with prompt: {prompt}")
            model = JitLlama("./stories15M.model.npz", args)
            
            print(f"\n{prompt}", end="")
            input_ids = np.array([tokenizer.encode(prompt)])
            
            start = time.time()
            token_count = input_ids.shape[1]
            generated_tokens = []
            for id in model.generate(input_ids, max_tokens):
                token_count += 1
                output_id = id[0, 0].item()
                if output_id in [tokenizer.eos_id, tokenizer.bos_id]:
                    break
                generated_tokens.append(output_id)
                print(tokenizer.decode([output_id]), end="")
                sys.stdout.flush()
                if token_count >= input_ids.shape[1] + max_tokens:
                    break
            
            elapsed = time.time() - start
            jit_tokens_per_sec = (token_count - input_ids.shape[1]) / elapsed
            print(f"\n\nJIT Llama: {token_count} tokens, {elapsed:.2f}s, {round(jit_tokens_per_sec)} tokens/s")
            
            # For comparison, print command to run on simple_llama3
            print(f"\nTo compare with simple_llama3, run:")
            print(f"python3 simple_llama3.py \"{prompt}\" | grep \"tokens/s\"")
            
        else:
            # Treat as normal prompt
            prompt = sys.argv[1]
            model = JitLlama("./stories15M.model.npz", args)
            
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
    
    else:
        # Regular model run with default prompt
        prompt = "I have a dream"
        model = JitLlama("./stories15M.model.npz", args)
        
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