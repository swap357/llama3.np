"""
Configuration for llama3np models.
"""


class ModelArgs:
    """
    Configuration for the Llama model.
    """
    def __init__(
            self,
            dim: int = 288,
            n_layers: int = 6,
            n_heads: int = 6,
            n_kv_heads: int = None,
            vocab_size: int = 32000,
            max_seq_len: int = 256,
            max_new_tokens: int = 200,
            norm_eps: float = 1e-6,
            max_batch_size: int = 32,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.max_new_tokens = max_new_tokens
        self.norm_eps = norm_eps
        self.max_batch_size = max_batch_size
        
    def __repr__(self) -> str:
        """Return string representation of model arguments"""
        attrs = [f"{key}={value}" for key, value in self.__dict__.items()]
        return f"ModelArgs({', '.join(attrs)})"