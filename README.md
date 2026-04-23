# Qwen3 / Qwen3.5 本地推理与加速研究

本项目从零手写 **Qwen3-0.6B** 和 **Qwen3.5-0.8B** 两个开源模型的本地推理引擎。核心目标是深入理解模型底层架构、实现完整的推理机制，并系统探索主流推理加速技术。

---

## 研究的模型架构

| 模型 | 架构类型 | 核心特征 |
|------|----------|----------|
| **Qwen3-0.6B** | 纯全注意力 Transformer | 100% 全维度 RoPE，GQA 16:8，28 层，`head_dim=128` |
| **Qwen3.5-0.8B** | Mamba + Transformer 混合架构 | 每 4 层中 3 层为 GLA（线性注意力）+ 1 层全注意力（GQA 8:2，25% RoPE），24 层，`head_dim=256` |

Qwen3.5 的混合架构是为超长上下文（256K）设计的激进稀疏化方案：线性注意力层的 KV Cache 是固定大小的状态矩阵（$O(1)$ 内存），彻底避免了传统 Transformer 的 $O(N)$ KV 爆炸问题。

---

## 项目目标

### 一、从零实现推理引擎
- 直接解析并加载 Safetensors 格式的原始权重，不依赖 Hugging Face `transformers` 做核心前向传播
- 手写复现 Qwen3.5 混合架构（Gated Delta Network + Gated Full Attention）的完整计算图

### 二、核心推理机制
- 实现自回归解码循环，成功从 Prompt 连续生成文本
- 从零构建并管理 KV Cache，避免自回归过程中的冗余计算

### 三、推理加速（核心重点，进行中）

| 方向 | 技术 | 状态 |
|------|------|------|
| 注意力加速 | FlashAttention / PyTorch SDPA | 📋 计划中 |
| 权重量化 | W8A16 / W4A16（GPTQ/AWQ） | 📋 计划中 |
| KV Cache 压缩 | INT8 / FP8 KV Cache 量化 | 📋 计划中 |
| 算子融合 | RMSNorm + Linear 融合 Kernel | 📋 计划中 |
| 静态计算图 | `torch.compile` Decode 优化 | 📋 计划中 |
| 投机解码 | Draft Model / Self-Speculative | 📋 可选探索 |
| 多硬件迁移 | Mac MPS → NVIDIA CUDA → AMD ROCm | 📋 计划中 |
| 并发推理 | Continuous Batching / Prefix Caching | 📋 计划中 |

---

## 进度追踪

### ✅ Phase 1：基础建设（2026-04-23）
- [x] 下载并检查 Qwen3.5-0.8B / Qwen3-0.6B 权重
- [x] 逆向分析 `config.json`，发现混合架构（Mamba + Transformer，25% RoPE，GQA，`tie_word_embeddings`）
- [x] 实现分词与 Embedding 权重加载（`01_tokenize_and_embed.py`）
- [x] 编写整体自回归前向传播伪代码（`02_pseudo_inference_flow.py`）
- [x] 实现 `RMSNorm`（含 bf16→f32 精度处理）（`03_rmsnorm.py`）
- [x] 实现局部旋转位置编码 `RoPE`（`partial_rotary_factor=0.25`）（`04_rope.py`）

### ✅ Phase 2：核心模块（2026-04-23）
- [x] 实现 `SwiGLU MLP`（1024→3584→1024，真实权重验证）（`07_mlp.py`）
- [x] 实现 `Full Attention`（GQA 8:2，Gated Attention，含 q/k norm）（`05_full_attention.py`）
- [x] 权重探查 Linear Attention 的内部算子形状（`06_linear_attention_inspector.py`）
- [x] 实现 `GLA Linear Attention`（Gated Delta Network，conv1d，状态机更新）（`06_linear_attention.py`）

### ✅ Phase 3 & 4：端到端生成与 KV Cache（2026-04-23）
- [x] 组装 Qwen3.5-0.8B 完整推理引擎，端到端自回归生成（`08_generate.py`）
- [x] **[里程碑]** 实现 Qwen3-0.6B 纯全注意力推理流水线，成功生成连贯中文（`09_qwen3_0_6b_generate.py`）
- [x] 修复 `q_norm`/`k_norm` 遗漏与 `kv_cache` 累积 Bug
- [x] **KV Cache Benchmark**：实测 $O(N)$ vs $O(N^2)$ 的性能差异

**Benchmark 结果**（Mac CPU，bfloat16，生成 20 tokens）：

| 指标 | 🟢 有 KV Cache | 🔴 无 KV Cache | 倍率 |
|------|--------------|--------------|------|
| Decode 总耗时 | **1.15 s** | 3.39 s | 约 **3x** |
| 单 Token 耗时 | **57.6 ms** | 169.5 ms（线性增长）| — |
| 生成速度 | **17.4 tokens/s** | 5.9 tokens/s | — |
| 峰值内存 | ~4659 MB | ~4654 MB | 基本持平（短序列）|

### 📋 Phase 5：推理加速（待开始）
- [ ] FlashAttention SDPA 替换（`F.scaled_dot_product_attention`）
- [ ] W8A16 手写权重量化，内存目标 ~2.4 GB
- [ ] INT8 KV Cache 量化
- [ ] `torch.compile` Decode 阶段优化
- [ ] Mac MPS 后端实验

### 📋 Phase 6：多硬件迁移（待开始）
- [ ] NVIDIA GPU 迁移 + CUDA Graph + `nsys` 分析
- [ ] AutoGPTQ W4 量化 + PPL 评估
- [ ] AMD ROCm 迁移验证

### 📋 Phase 7：并发推理系统（待开始）
- [ ] Continuous Batching 批推理实验
- [ ] Prefix Caching 共享前缀实验
- [ ] vLLM PagedAttention 源码阅读

> 📍 详细计划与实验设计见：**[docs/00_roadmap.md](docs/00_roadmap.md)**

---

## 文档索引

### 工程文档（`docs/`）

| 编号 | 文件 | 内容摘要 |
|------|------|----------|
| 00 | [00_roadmap.md](docs/00_roadmap.md) | 项目全局路线图，含已完成事件与未来计划 |
| 01 | [01_config_analysis.md](docs/01_config_analysis.md) | Qwen3.5-0.8B config.json 逐字段深度解析 |
| 02 | [02_forward_pass_workflow.md](docs/02_forward_pass_workflow.md) | 自回归推理完整流程（Prefill / Decode / KV Cache）|
| 03 | [03_architectural_discoveries.md](docs/03_architectural_discoveries.md) | 逆向发现：Gated Attention、Partial RoPE、GLA |
| 04 | [04_execution_suggestions.md](docs/04_execution_suggestions.md) | 项目启动时的技术路径规划与执行建议 |
| 05 | [05_linear_attention_mechanics.md](docs/05_linear_attention_mechanics.md) | GLA / Mamba 变体核心机制与工程挑战 |
| 06 | [06_core_challenges_and_pitfalls.md](docs/06_core_challenges_and_pitfalls.md) | 四大技术深坑（数值爆炸、精度漂移、Prefill 瓶颈、门控迷宫）|
| 07 | [07_kv_cache_benchmark_analysis.md](docs/07_kv_cache_benchmark_analysis.md) | Qwen3 vs Qwen3.5 架构演进 + KV Cache 量化 Benchmark |

### 基础概念（`docs/basic/`）

| 编号 | 文件 | 内容摘要 |
|------|------|----------|
| B01 | [basic/01_rmsnorm.md](docs/basic/01_rmsnorm.md) | RMSNorm 原理、两模型差异、精度敏感性与推理耗时分析 |
| B02 | [basic/02_rope.md](docs/basic/02_rope.md) | RoPE 原理、为何 Qwen3.5 只旋转 25% 维度、长文本外推能力 |
| B03 | [basic/03_activation_swiglu.md](docs/basic/03_activation_swiglu.md) | SiLU / SwiGLU 原理、门控机制、MLP 是推理最主要耗时来源 |

---

## 脚本索引

| 脚本 | 说明 |
|------|------|
| `01_tokenize_and_embed.py` | 分词与 Embedding 权重手工加载验证 |
| `02_pseudo_inference_flow.py` | 整体自回归推理伪代码（全局视野）|
| `03_rmsnorm.py` | RMSNorm 实现与真实权重验证 |
| `04_rope.py` | 局部旋转 RoPE 实现与正确性验证 |
| `05_full_attention.py` | Gated Full Attention 教学版（不含 causal mask）|
| `06_linear_attention_inspector.py` | 逆向探查 Linear Attention 权重形状 |
| `06_linear_attention.py` | GLA Linear Attention 教学版（conv1d 为占位实现）|
| `07_mlp.py` | SwiGLU MLP 实现与真实权重验证 |
| `08_generate.py` | Qwen3.5-0.8B 完整端到端推理引擎 |
| `09_qwen3_0_6b_generate.py` | **Qwen3-0.6B 完整推理引擎（推荐运行）**，含 KV Cache Benchmark |
| `scripts/inspect_mlp.py` | MLP 权重结构快速探查工具 |

---

## 快速开始

项目使用 `pyproject.toml` 管理依赖，推荐通过 `uv` 运行。

```bash
# 1. 确保已安装 uv（https://github.com/astral-sh/uv）
# 2. 运行 Qwen3-0.6B 推理引擎（默认开启 KV Cache）
uv run python 09_qwen3_0_6b_generate.py

# 3. 对比无 KV Cache 的性能（体验 O(N²) 的延迟雪崩）
uv run python 09_qwen3_0_6b_generate.py --no-kv-cache
```

**运行前提**：需要在本地 `Qwen3-0.6B/` 目录下放置模型权重文件（`model.safetensors`）。
