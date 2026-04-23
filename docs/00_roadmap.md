# localinfer 学习与工程路线图

> 本文档是 `localinfer` 项目的全局路线图，记录已完成的里程碑事件及未来的学习计划。
> 所有"已完成"章节均包含时间戳与具体产出；"计划中"章节包含实验设计与预期指标。

---

## 文档索引

| 编号 | 文件 | 内容 |
|------|------|------|
| 00 | [00_roadmap.md](00_roadmap.md) | 项目全局路线图，含已完成事件与未来计划 |
| 01 | [01_config_analysis.md](01_config_analysis.md) | Qwen3.5-0.8B config.json 逐字段解析，包含混合架构发现 |
| 02 | [02_forward_pass_workflow.md](02_forward_pass_workflow.md) | 自回归推理完整流程解析（Prefill / Decode / KV Cache 原理）|
| 03 | [03_architectural_discoveries.md](03_architectural_discoveries.md) | 逆向发现：Gated Attention、Partial RoPE、GLA |
| 04 | [04_execution_suggestions.md](04_execution_suggestions.md) | 项目启动时的技术路径规划与执行建议 |
| 05 | [05_linear_attention_mechanics.md](05_linear_attention_mechanics.md) | GLA / Mamba 变体核心机制、双重人格与工程挑战 |
| 06 | [06_core_challenges_and_pitfalls.md](06_core_challenges_and_pitfalls.md) | 四大技术深坑（数值爆炸、精度漂移、Prefill 瓶颈、门控迷宫）|
| 07 | [07_kv_cache_benchmark_analysis.md](07_kv_cache_benchmark_analysis.md) | Qwen3 vs Qwen3.5 架构演进 + KV Cache Benchmark 报告 |

### basic/ — 基础概念专题

> 针对项目中涉及的每个核心概念独立成文，包含两模型差异对比与推理影响分析。

| 编号 | 文件 | 内容 |
|------|------|------|
| B01 | [basic/01_rmsnorm.md](basic/01_rmsnorm.md) | RMSNorm 原理、Qwen3 vs Qwen3.5 差异、精度敏感性、推理耗时分析 |
| B02 | [basic/02_rope.md](basic/02_rope.md) | RoPE 原理、为何 Qwen3.5 只旋转 25% 维度、长文本外推能力 |
| B03 | [basic/03_activation_swiglu.md](basic/03_activation_swiglu.md) | SiLU / SwiGLU 原理、门控机制、两模型 MLP 维度对比与推理耗时 |


---

## 已完成工作（Phase 1–4）

### ✅ Phase 1：基础建设
> **时间**：2026-04-23（项目启动）

| 产出 | 脚本 | 关键发现 |
|------|------|----------|
| 分词与词嵌入加载 | `01_tokenize_and_embed.py` | 手工 safetensors mmap 查表，避免全量加载 |
| 整体推理流程伪代码 | `02_pseudo_inference_flow.py` | 建立 24 层混合架构全局视野 |
| RMSNorm 实现与验证 | `03_rmsnorm.py` | bf16→f32→bf16 精度转换，`rsqrt` 硬件加速 |
| RoPE 局部旋转实现 | `04_rope.py` | 发现 `partial_rotary_factor=0.25`，仅前 64 维旋转 |
| config.json 逆向分析 | — | 确认混合架构（3 Linear + 1 Full 每 4 层） |

---

### ✅ Phase 2：核心模块实现
> **时间**：2026-04-23

| 产出 | 脚本 | 关键发现 |
|------|------|----------|
| Full Attention（Gated）教学版 | `05_full_attention.py` | 发现 `q_proj` 是 Q+Gate 合并投影（4096 维），`attn_output_gate=true` |
| Linear Attention 权重探查 | `06_linear_attention_inspector.py` | 逐字段打印 `in_proj_qkv/z/a/b`、`conv1d`、`A_log`、`dt_bias` 的 shape |
| Linear Attention 教学版 | `06_linear_attention.py` | 展示 GLA 单步 decode 的状态机更新逻辑（conv1d 为占位实现）|
| SwiGLU MLP 实现 | `07_mlp.py` | 1024→3584→门控相乘→1024，三矩阵真实权重加载验证 |

---

### ✅ Phase 3：端到端生成（Qwen3.5-0.8B）
> **时间**：2026-04-23

| 产出 | 脚本 | 关键问题 |
|------|------|----------|
| Qwen3.5 完整推理引擎 | `08_generate.py` | 修复：补充 Q 缩放因子（`1/sqrt(head_dim)`）、l2norm、f32 精度转换 |
| 已知问题：精度漂移 | — | 纯 Python 实现 GLA 存在累积误差，需 Triton kernel 级数值对齐才可生成连贯中文 |
| 已知问题：串行 Prefill | — | Mamba 无法并行扫描，token-by-token 喂入导致冷启动慢 |

> 详见 [06_core_challenges_and_pitfalls.md](06_core_challenges_and_pitfalls.md)

---

### ✅ Phase 4：Qwen3-0.6B 纯全注意力基准 + KV Cache Benchmark
> **时间**：2026-04-23

| 产出 | 脚本 | 说明 |
|------|------|------|
| Qwen3-0.6B 推理引擎 | `09_qwen3_0_6b_generate.py` | 纯 Full Attention，28 层，GQA 16:8，完整 causal mask |
| KV Cache 对比 Benchmark | `09` 的 `--no-kv-cache` 参数 | CPU bfloat16，生成 20 tokens |
| 性能 profiling 模块 | — | 逐 token 耗时 / 峰值内存 / throughput |

**Benchmark 结果**（Mac CPU，bfloat16，prompt: "人工智能的未来是"，生成 20 tokens）：

| 指标 | 有 KV Cache | 无 KV Cache | 结论 |
|------|-------------|-------------|------|
| Decode 总耗时 | **1.15 s** | 3.39 s | 约 **3x 提速** |
| 单 Token 平均耗时 | **57.6 ms** | 169.5 ms | 延迟显著降低 |
| 生成速度 | **17.4 tokens/s** | 5.9 tokens/s | — |
| 峰值内存 | ~4659 MB | ~4654 MB | 短序列下基本持平（模型权重主导） |

> 无缓存模式的逐 token 延迟呈线性雪崩（第 20 个 token 达 205ms），与 O(N²) 理论完全吻合。
> 详见 [07_kv_cache_benchmark_analysis.md](07_kv_cache_benchmark_analysis.md)

---

## 计划中工作

### 📋 Phase 5：推理加速（当前重点）

#### 5-A：FlashAttention 替换（优先级：最高）
> **目标**：用 `F.scaled_dot_product_attention` 替换 `09` 里的手写 matmul + softmax

```python
# 修改 09_qwen3_0_6b_generate.py 的 FullAttentionBlock.forward()
attn_output = F.scaled_dot_product_attention(
    q, k_rep, v_rep,
    is_causal=(seq_len > 1)  # decode 时 seq_len=1，无需 causal mask
)
```

**预期验证**：
- CPU 下加速不明显（MPS 和 CUDA 才能发挥 FlashAttention 效果）
- 为后续 GPU 实验建立正确的接口基础

---

#### 5-B：权重量化（W8A16）
> **新脚本**：`10_quantization_w8a16.py`

实现步骤：
1. 对 `Qwen3Model` 所有 `nn.Linear` 层进行 per-channel INT8 量化
2. 存储 `weight_int8` + `scale` 替代原始 `bfloat16` 权重
3. 推理时反量化：`weight_fp = weight_int8.to(bf16) * scale`

**预期指标**：
- 峰值内存：4659 MB → ~2400 MB（理论 50% 压缩）
- 速度变化：CPU 可能略降（反量化开销），GPU 上应提速（带宽减半）
- 生成质量：逐层激活 MSE 对比原始模型

---

#### 5-C：KV Cache 量化（INT8）
> **新脚本**：`11_kv_cache_quant.py`

在现有 `kv_cache = (k, v)` 结构上叠加量化：
- 存储时：`k_int8 = (k / scale).round().clamp(-128, 127).to(int8)`
- 读取时：`k = k_int8.to(bf16) * scale`

**意义**：长文本场景（32K+ token）下，KV Cache 将超越权重成为内存主角，此优化是长文本推理的关键。

---

#### 5-D：torch.compile 静态图优化
> 修改 `09`，在 decode 阶段（输入 shape 固定为 `[1,1]`）启用

```python
model_compiled = torch.compile(model.forward_step, mode="reduce-overhead")
```

**预期加速**：decode 阶段 20~40%（消除 Python 层 overhead）

---

#### 5-E：Speculative Decoding（可选探索）
基于 Qwen3-0.6B 作为 Draft Model，人工构造简单的 n-gram 候选，体验"并行验证"的加速机制。

---

### 📋 Phase 6：多硬件迁移

#### 阶段次序

```
Mac CPU（当前）
  ↓ 补充 Apple MPS 实验
Mac MPS（Metal Performance Shaders）
  ↓ 租云实例（Lambda Labs / RunPod，~$0.6/h）
NVIDIA A10 / A100
  ↓ 切换 ROCm PyTorch 构建
AMD MI250 / ROCm
```

#### 每个平台的实验清单

**Mac MPS**（下一步）：
```python
device = torch.device("mps")
# 注意：MPS bfloat16 支持有限，可能需要 float16
```
- 对比：CPU vs MPS 的 decode 速度

**NVIDIA GPU**：
```python
# Step 1：基础迁移
device = torch.device("cuda")

# Step 2：Flash Attention SDPA（A10 上自动使用 Flash Attention 内核）
attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)

# Step 3：CUDA Graph（消除 kernel launch overhead）
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    logits, caches = model.forward_step(...)
# 后续 decode 步骤：g.replay()

# Step 4：性能分析
# torch.profiler → nsys → ncu（逐步深入）
```

**AMD ROCm**：
```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
# 代码层面基本零修改（PyTorch 通过 HIP 桥接）
# 性能分析：rocm-smi / rocprof
```

#### 标准化 Benchmark 输出格式

每个平台实验结束后，输出统一格式便于横向对比：
```
平台:   Mac CPU M-series / NVIDIA A10 / AMD MI250
精度:   bfloat16 / float16 / int8
TTFT:   xx ms（首 token 延迟）
TBT:    xx ms（token 间平均延迟）
Speed:  xx tokens/s
RAM:    xx MB
```

---

### 📋 Phase 7：并发推理系统

| 知识点 | 实验脚本 | 核心验证问题 |
|--------|----------|-------------|
| Static vs Dynamic Batching | `12_batch_inference.py` | padding 浪费了多少算力？ |
| Prefix Caching | `13_prefix_cache.py` | 共享 system prompt 能省多少 TTFT？ |
| Chunked Prefill | 修改 `09` | 长 prompt 如何避免 decode 请求被饿死？ |
| Tensor Parallelism（模拟） | `14_tensor_parallel_sim.py` | AllReduce 通信占推理时间的比例？ |

**配套阅读**：
- vLLM `block_manager.py` + `scheduler.py`（PagedAttention + Continuous Batching）
- SGLang `radix_cache.py`（RadixTree 前缀缓存）

---

## 整体时间规划

```
[Month 1] Phase 5-A/B/C（推理加速基础）
  Week 1：FlashAttention SDPA 替换 + MPS 实验
  Week 2：W8A16 手写量化，对比内存与速度
  Week 3：INT8 KV Cache 量化，评估长文本场景
  Week 4：torch.compile decode 阶段实验

[Month 2] Phase 6（硬件迁移）
  Week 1-2：NVIDIA 云实例，CUDA Graph + nsys 入门
  Week 3：AutoGPTQ W4 量化（配合 GPU 才能体现带宽优势）
  Week 4：AMD ROCm 迁移验证

[Month 3] Phase 7（并发系统）
  Week 1-2：批推理实验，理解 throughput vs latency tradeoff
  Week 3：vLLM 源码阅读，PagedAttention 核心逻辑
  Week 4：Prefix Caching 实验
```

---

## 里程碑自测清单

- [x] 手写 RMSNorm / RoPE / SwiGLU MLP，从 safetensors 加载真实权重验证
- [x] 实现 Gated Full Attention（含 q/k norm，含 KV Cache 积累）
- [x] 实现 GLA Linear Attention（Delta Rule 核心更新）
- [x] 端到端自回归生成（Qwen3.5-0.8B，受数值精度限制）
- [x] Qwen3-0.6B 纯全注意力推理，生成连贯中文文本
- [x] KV Cache vs 无 Cache 量化 Benchmark（~3x 加速，O(N) vs O(N²) 实测）
- [ ] FlashAttention SDPA 替换，测量加速比
- [ ] W8A16 权重量化，内存降至 ~2.4GB
- [ ] INT8 KV Cache 量化，为长文本场景做准备
- [ ] Mac MPS 后端实验
- [ ] NVIDIA GPU 迁移 + CUDA Graph
- [ ] torch.profiler / nsys 性能分析
- [ ] AutoGPTQ W4 量化 + PPL 评估
- [ ] AMD ROCm 迁移验证
- [ ] 批推理 + Prefix Caching 实验
- [ ] 阅读 vLLM PagedAttention 实现
