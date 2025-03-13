# llama3.np: NumPy Implementation of Llama3

[![Tests](https://github.com/swap357/llama3.np/actions/workflows/test_and_benchmark.yml/badge.svg)](https://github.com/swap357/llama3.np/actions/workflows/test_and_benchmark.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

<p align="center">
  <img src="/assets/llama3.np.webp" width="300" alt="llama3.np">
</p>

A simplified NumPy implementation of the Llama3 language model with performance optimizations. This project provides both original and optimized implementations for learning, experimentation, and performance comparison.

## 🚀 Performance Improvements

| Component | Speedup | Notes |
|-----------|---------|-------|
| Tokenizer | **~507x** | Dictionary-based lookup instead of list.index() |
| RoPE      | **~1.07-1.5x** | Direct indexing instead of reshape/split/stack |
| Inference | **~1.01x** | End-to-end token generation speedup |

## 📋 Requirements

- Python 3.8+
- NumPy

## 🔧 Installation

```bash
git clone https://github.com/swap357/llama3.np.git
cd llama3.np
pip install -e .
```

## 🏃‍♂️ Quick Start

Run the model with default settings:

```bash
python run_llama.py --prompt "Once upon a time"
```

Use the optimized implementation:

```bash
python run_llama.py --prompt "Once upon a time" --optimized
```

## 🔍 Benchmarking

Run comprehensive benchmarks:

```bash
python run_benchmarks.py --all
```

Or test specific components:

```bash
python run_benchmarks.py --tokenization --rope
python run_benchmarks.py --inference --max-tokens 50
```

## 📊 Key Findings

Our performance analysis uncovered several optimization opportunities:

1. **Tokenizer Optimization**: 
   - Original implementation used list.index() for lookups (O(n))
   - Optimized implementation uses dictionary lookup (O(1))
   - Result: Massive 507x speedup

2. **RoPE Implementation**:
   - Original implementation used complex reshape/split/stack operations
   - Optimized implementation uses direct indexing
   - Result: 7-50% speedup depending on batch size and sequence length

3. **End-to-End Performance**:
   - Tokenization is dramatically faster
   - Core inference speed shows modest 1% improvement
   - Clear separation of prefill vs. decode phases reveals different optimization needs

## 📂 Project Structure

```
llama3.np/
├── llama3np/                # Main package
│   ├── model/               # Model implementations
│   │   ├── base.py          # Original implementation
│   │   └── optimized.py     # Optimized implementation
│   ├── utils/               # Utilities
│   │   ├── config.py        # Model configuration
│   │   ├── loader.py        # Weight loading utilities
│   │   ├── tokenizer.py     # Original tokenizer
│   │   └── optimized_tokenizer.py # Optimized tokenizer
│   └── benchmark/           # Benchmarking tools
├── tests/                   # Test suite
├── run_llama.py             # CLI for text generation
├── run_benchmarks.py        # Benchmarking script
└── setup.py                 # Package installation
```

## References

Thank you to the creators of the following libraries and tools and their contributors:
- [llama2.c](https://github.com/karpathy/llama2.c) - @karpathy
- [llama.np](https://github.com/hscspring/llama.np) - @hscspring
- [modeling_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) - Hugging Face's Transformers

## License

MIT License