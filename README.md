# Qwen3.5-0.8B Local Inference & Acceleration

This project aims to build a from-scratch implementation for loading and running inference on the open-source **Qwen3.5-0.8B** model. Our primary focus is to deeply understand the underlying model architecture, implement core inference mechanics, and explore major inference acceleration techniques.

## Model Information: Qwen3.5-0.8B
- **Release Date**: March 2026 (Alibaba Qwen3.5 Family)
- **Parameters**: ~0.8 Billion (Ultra-compact)
- **Architecture**: Hybrid Architecture (Gated Delta Networks and Gated Attention)
- **Context Length**: 256K context support
- **Capabilities**: Multimodal (Vision-Language), supports both "thinking" and "non-thinking" modes, covers 201 languages.

## Project Goals

### 1. From-Scratch Implementation
- Parse and load the raw model weights (e.g., Safetensors/PyTorch formats) locally.
- Reconstruct the specific Qwen3.5 hybrid architecture (Gated Delta Networks + Gated Attention) without relying on heavy high-level wrappers like Hugging Face `transformers` for the core forward pass.

### 2. Core Inference Mechanics
- Implement an autoregressive decoding loop.
- Successfully generate a continuous sequence of 10 tokens given a prompt.
- **KV Cache Implementation**: Build and manage Key-Value caching from the ground up to avoid redundant computations during the autoregressive generation phase.

### 3. Inference Acceleration (Core Focus)
Explore and implement mainstream inference acceleration schemes, which may include:
- **FlashAttention**: Integrate memory-efficient attention mechanisms to optimize the Gated Attention layers.
- **Quantization**: Implement INT8/INT4 weight-only quantization (e.g., GPTQ, AWQ) to reduce memory bandwidth bottlenecks.
- **Operator/Kernel Fusion**: Fuse operators (e.g., RMSNorm + Linear, RoPE application) to reduce memory access overhead and kernel launch latency.
- **Speculative Decoding** *(Optional exploration)*: Investigate draft-model or self-speculative approaches to speed up token generation.

## Roadmap
- [x] **Phase 1: Foundation**
  - [x] Download and inspect Qwen3.5-0.8B weights.
  - [x] Analyze `config.json` (Discovered Hybrid Architecture: Mamba + Transformer, 25% RoPE, GQA, `tie_word_embeddings`).
  - [x] Implement Tokenization & Embedding weight loading script (`01_tokenize_and_embed.py`).
  - [x] Document the overall Autoregressive Forward Pass pseudo-code (`02_pseudo_inference_flow.py`).
  - [x] Implement foundational `RMSNorm` (`03_rmsnorm.py`).
- [x] **Phase 2: Core Architecture**
  - [x] Implement `SwiGLU_MLP` (1024 -> 3584 -> 1024) (`07_mlp.py`).
  - [x] Implement `FullAttention` (GQA 8:2, partial 25% RoPE, Gated Attention) (`05_full_attention.py`).
  - [x] Implement `LinearAttention` (Gated Delta Network / Mamba variant) (`06_linear_attention.py`).
  - [x] Stitch layers into `Qwen3_5_Block` and `Qwen3_5_Model` (`08_generate.py`).
- [x] **Phase 3 & 4: Basic Generation & KV Cache (Merged)**
  - [x] Implement the naive autoregressive generation (Token-by-Token).
  - [x] Add Dual Cache management (KV for Full Attention + RNN/Conv State for Linear Attention) and verify correctness (`08_generate.py`).
- [ ] **Phase 5: Acceleration & Profiling**
  - [ ] Profile the baseline implementation.
  - [ ] Iteratively apply acceleration techniques (FlashAttention, Quantization, Fusion) and document speedups.

## Documentation
- [01_config_analysis.md](docs/01_config_analysis.md)
- [02_forward_pass_workflow.md](docs/02_forward_pass_workflow.md)
- [03_architectural_discoveries.md](docs/03_architectural_discoveries.md)
- [04_execution_suggestions.md](docs/04_execution_suggestions.md)
- [05_linear_attention_mechanics.md](docs/05_linear_attention_mechanics.md)
- [06_core_challenges_and_pitfalls.md](docs/06_core_challenges_and_pitfalls.md)

## Getting Started
*(Instructions for environment setup, model download, and execution will be added as the project progresses)*
