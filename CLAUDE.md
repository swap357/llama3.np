# llama3.np Development Guide

## Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Run with default prompt
python llama3.py

# Run with custom prompt
python llama3.py "Your prompt text here"
```

## Code Style Guidelines
- **Imports**: Standard library first, then third-party (numpy), then local
- **Naming**: CamelCase for classes, snake_case for functions/variables
- **Type Annotations**: Use throughout with PEP 484-style typing
- **Arrays**: Use custom Array type with shape info in comments
- **Error Handling**: Minimal with assertions for validation
- **Performance**: Optimize NumPy operations, careful with memory management
- **Documentation**: Use type annotations over docstrings

## Project Structure
- `llama3.py`: Main model implementation
- `config.py`: Model configuration dataclasses
- `tokenizer.py`: BPE tokenizer
- `utils.py`: Utilities for parameter loading
- Model files: `stories15M.model.npz` and `tokenizer.model.np`