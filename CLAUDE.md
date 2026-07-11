# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the official code repository for "Build a Large Language Model (From Scratch)" by Sebastian Raschka. It implements a GPT-like LLM in PyTorch from scratch, covering the complete pipeline: tokenization, attention mechanisms, model architecture, pretraining, and finetuning.

## Development Commands

### Setup

```bash
# Install dependencies using uv (recommended)
uv sync --dev

# Or using pip
pip install -r requirements.txt

# Install bonus materials (optional)
uv pip install --group bonus
```

### Running Tests

```bash
# Run all package tests
pytest pkg/llms_from_scratch/tests/

# Run specific chapter tests
pytest ch04/01_main-chapter-code/tests.py
pytest ch05/01_main-chapter-code/tests.py
pytest ch06/01_main-chapter-code/tests.py

# Run notebook tests
pytest --nbval ch02/01_main-chapter-code/dataloader.ipynb
pytest --nbval ch03/01_main-chapter-code/multihead-attention.ipynb

# Run with ruff linting
pytest --ruff <path_to_test_file>
```

### Running Code

Main chapter code is organized as Jupyter notebooks (`*.ipynb`) and standalone Python scripts:

```bash
# Run a Python script
python ch05/01_main-chapter-code/gpt_train.py
python ch05/01_main-chapter-code/gpt_generate.py
```

## Code Architecture

### Directory Structure

- **ch01-ch07**: Chapter-by-chapter code implementations
  - `01_main-chapter-code/`: Primary code for each chapter
  - `02_bonus_*`, `03_bonus_*`: Additional/alternative implementations
  - `04_user_interface`, `06_user_interface`: Chainlit UI apps
- **appendix-A to appendix-E**: Supplementary materials (PyTorch intro, LoRA, etc.)
- **pkg/llms_from_scratch**: PyPI package version of the main chapter code
- **setup**: Environment setup guides and scripts
- **data**: Dataset files

### Core Components (ch04)

The GPT model implementation in `ch04/01_main-chapter-code/gpt.py` (and `pkg/llms_from_scratch/ch04.py`):

- `GPTDatasetV1`: Dataset class with sliding window tokenization
- `MultiHeadAttention`: Self-attention with causal masking
- `LayerNorm`, `GELU`, `FeedForward`: Standard transformer components
- `TransformerBlock`: Attention + FFN with residual connections
- `GPTModel`: Full model with token/position embeddings

### Training Pipeline (ch05)

Key training utilities in `ch05/01_main-chapter-code/gpt_train.py`:

- `calc_loss_batch`, `calc_loss_loader`: Loss computation
- `evaluate_model`: Validation evaluation
- `train_model_simple`: Main training loop

### Model Variants

The repository includes implementations of modern LLM architectures:

- **ch05/07_gpt_to_llama**: Llama 3.2 implementation with RoPE
- **ch05/11_qwen3**: Qwen3 (Dense and MoE variants)
- **ch05/12_gemma3**: Gemma 3 implementation
- **ch05/13_olmo3**: Olmo 3 implementation

### Bonus Implementations (ch04)

Advanced attention mechanisms and optimizations:

- **03_kv-cache**: KV cache for efficient generation
- **04_gqa**: Grouped-Query Attention
- **05_mla**: Multi-Head Latent Attention
- **06_swa**: Sliding Window Attention
- **07_moe**: Mixture of Experts
- **08_deltanet**: Gated DeltaNet

## Important Notes

- Main chapter code corresponds to the print book and should not be modified
- Bonus materials are in `02_bonus_*`, `03_bonus_*` subdirectories
- The `pkg/` directory contains the `llms_from_scratch` PyPI package
- Python 3.10-3.13 supported, PyTorch 2.2.2+ required
