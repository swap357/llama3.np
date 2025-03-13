# llama3.np: NumPy Implementation of Llama3

[![Tests](https://github.com/swap357/llama3.np/actions/workflows/test_and_benchmark.yml/badge.svg)](https://github.com/swap357/llama3.np/actions/workflows/test_and_benchmark.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)


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
python scripts/run_llama.py --prompt "Once upon a time"
```

Use the optimized implementation:

```bash
python scripts/run_llama.py --prompt "Once upon a time" --optimized
```

Compare performance between versions:

```bash
python scripts/run_llama.py --prompt "Once upon a time" --compare
```

## 🔍 Benchmarking

Run comprehensive benchmarks:

```bash
python scripts/run_benchmarks.py --all
```

Or test specific components:

```bash
python scripts/run_benchmarks.py --tokenization --rope
python scripts/run_benchmarks.py --inference --max-tokens 50
```

Run direct component comparisons:

```bash
python -m llama3np.benchmark.direct --prompt "Once upon a time" --iterations 50
```

Run complete model comparison:

```bash
python -m llama3np.benchmark.llama --prompt "Once upon a time" --tokens 30
```

## 🔬 Analysis Tools

Analyze model bytecode:

```bash
python analysis/bytecode/analyze_bytecode.py --funcs "apply_rotary_emb,softmax"
```

Run comprehensive profiling:

```bash
python scripts/run_analysis.py --prompt "Once upon a time" --tokens 20
```

Profile specific inference phases:

```bash
python analysis/profiling/profile_inference.py --prompt "Hello" --max-tokens 10 --phases prefill
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
│       ├── components.py    # Component-level benchmarks
│       ├── end_to_end.py    # End-to-end benchmarks
│       ├── direct.py        # Direct component comparison
│       └── llama.py         # Full model comparison
├── scripts/                 # High-level scripts
│   ├── run_llama.py         # CLI for text generation
│   ├── run_benchmarks.py    # Benchmarking orchestration
│   └── run_analysis.py      # Analysis orchestration
├── analysis/                # Analysis tools
│   ├── bytecode/            # Bytecode analysis
│   ├── profiling/           # Performance profiling
│   └── types/               # Type annotation analysis
├── tests/                   # Test suite
└── setup.py                 # Package installation
```

## References

Thank you to the creators of the following libraries and tools and their contributors:
- [llama2.c](https://github.com/karpathy/llama2.c) - @karpathy
- [llama.np](https://github.com/hscspring/llama.np) - @hscspring
- [modeling_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) - Hugging Face's Transformers

## License

MIT License