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

#### ✅ 5-A：FlashAttention 替换（已完成）
> **目标**：用 `F.scaled_dot_product_attention` 替换手写 matmul + softmax  
> **完成**：已内置到 `10_backend_benchmark.py` 的 `FullAttentionBlock`

```python
# 10_backend_benchmark.py — FullAttentionBlock.forward()
attn_out = F.scaled_dot_product_attention(
    q, k_rep, v_rep,
    is_causal=(S > 1)   # decode 时 S=1，无需 causal mask
)
```

**实测结果（Mac CPU arm，bfloat16，decode 10 tokens）**：
- TBT: **28ms**（vs 旧版 57ms），约 **2x 提速**（SDPA 在 CPU 上也有优化）
- CUDA 上将自动调度 FlashAttention-2 kernel，预期再提速 3~5x

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

#### ✅ 5-D：torch.compile + 静态 KV Cache（已完成）
> **目标**：在 Mac MPS 上验证图编译对 decode 阶段的收益  
> **完成**：新增 `11_mps_compile_kv_benchmark.py`，用预分配静态 KV Cache 固定图形状

```python
compiled_runner = torch.compile(
    StaticKVForward(model, max_cache_len=64, device="mps"),
    mode="reduce-overhead",
)
```

**实测结果（Mac MPS，float16，max_cache_len=64）**：
- eager static KV：TBT **25-26ms**，约 **39 tokens/s**
- compile static KV：TBT **12-13ms**，约 **75-81 tokens/s**
- 首次编译 warmup：**26.50s**，适合长会话/常驻服务摊销

**文档**：
- 测试结果：[metal_benchmark.md](metal_benchmark.md)
- 原理说明：[basic/04_torch_compile_static_kv.md](basic/04_torch_compile_static_kv.md)

---

#### 5-E：Mac MPS 后续优化实验（下一步）

> **目标**：继续在 macOS 上学习“动态图/静态图、KV Cache、dtype、prompt 长度、量化”之间的性能权衡。

**推荐实验矩阵**：

| 实验 | 脚本/命令 | 学习目标 |
|------|----------|----------|
| max_cache_len 扫描 | `11_mps_compile_kv_benchmark.py --max-cache-len 32/64/128` | 固定图长度越大，编译图越稳定，但 attention 计算越浪费 |
| 动态 KV vs 静态 KV eager | `--skip-compile --include-dynamic-kv` | 对比真实历史长度 vs 固定 cache 长度的 eager 成本 |
| float16 vs float32 compile | `--dtype float16/float32` | 验证 MPS 不同 dtype 的 kernel 成熟度和内存差异 |
| prompt 长度扫描 | 构造 4/32/128/512 tokens prompt | 分离 prefill 和 decode 的瓶颈 |
| SDPA vs 手写 attention | 保留手写 matmul + softmax 版本 | 理解 PyTorch SDPA 在 MPS 上的收益 |
| block/static KV 折中 | 设计 block cache，例如 64/128 token 一块 | 学习接近 PagedAttention 的 cache 管理思路 |

**建议优先顺序**：
1. `max_cache_len=32/64/128` 扫描，观察 compile warmup、TBT、内存变化。
2. `dynamic KV eager` vs `static KV eager` vs `static KV compile` 三者横向对比。
3. 用较长 prompt 测 prefill/decode 分界。
4. 再考虑 block KV Cache，避免一次性固定到很大的 `max_cache_len`。

---

#### 5-F：MLX 生态学习与对比（下一步）

> **目标**：了解 Apple Silicon 原生推理生态，与 PyTorch/MPS 手写推理引擎形成对照。

**MLX 适合学习的点**：
- Apple 官方机器学习框架，围绕 Unified Memory 设计。
- `mlx-lm` 提供本地 LLM 推理工具链。
- MLX community 有大量 4-bit / 8-bit 量化模型，适合 macOS 本地推理。
- 与当前 PyTorch/MPS 项目互补：PyTorch 手写实现适合理解推理内部，MLX 适合理解 Apple 原生高性能推理路径。

**建议新增实验脚本**：`12_mlx_benchmark.py`

**推荐实验矩阵**：

| 实验 | 对比对象 | 学习目标 |
|------|----------|----------|
| MLX fp16 生成速度 | PyTorch MPS dynamic KV eager | 框架级推理路径对比 |
| MLX 4bit 生成速度 | PyTorch MPS float16 | 量化对内存带宽和速度的影响 |
| MLX prompt 长度扫描 | PyTorch MPS prompt 长度扫描 | 比较 prefill/decode 行为 |
| MLX 内存占用 | PyTorch MPS Mem(MB) | Apple Unified Memory 下量化收益 |

**建议先跑现成模型**：
```bash
pip install mlx-lm

mlx_lm.generate \
  --model mlx-community/Qwen3-0.6B-4bit \
  --prompt "人工智能的未来是" \
  --max-tokens 100
```

> 模型名称需要以 MLX community 实际可用模型为准。如果没有 Qwen3-0.6B，可先用 Qwen2.5/其他小模型跑通 MLX benchmark 流程。

**最终推荐横向表**：

| 路线 | 配置 | 重点指标 |
|------|------|----------|
| PyTorch MPS | dynamic KV eager | 自然手写推理 baseline |
| PyTorch MPS | static KV eager | 固定图形状成本 |
| PyTorch MPS | static KV compile | 图编译收益 |
| MLX | fp16 / 4bit | Apple 原生推理生态与量化收益 |

---

#### 5-G：Speculative Decoding（可选探索）
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

**Mac Metal (MPS) — 加速点全景**：

> MPS 最大优势：**Unified Memory**，CPU/GPU 零拷贝共享物理内存（M4 Pro 带宽 ~273 GB/s）。

| 优先级 | 技术 | 说明 | 注意事项 |
|--------|------|------|---------|
| ⭐⭐⭐ | **基础迁移 `device="mps"`** | `10_backend_benchmark.py` 已支持，立刻可测 | **bfloat16 → 自动降级 float16** |
| ⭐⭐⭐ | **SDPA on MPS** | PyTorch ≥2.1 MPS 后端支持 SDPA，走 Metal shader | 非 FlashAttention，是 Metal kernel 近似 |
| ⭐⭐ | **float16 精度** | 统一内存高带宽，矩阵乘法受益明显 | bf16 在部分 MPS 版本不稳定 |
| ✅ | **torch.compile + 静态 KV** | 已实测 TBT 25-26ms → 12-13ms | warmup ~26.5s，需长会话摊销 |
| ⭐⭐ | **max_cache_len 扫描** | 32/64/128 固定 cache 长度对比 | 固定长度越大，计算浪费越多 |
| ⭐⭐ | **动态 KV vs 静态 KV** | 对比 eager 自然实现与 compile 友好实现 | 理解工程 trade-off |
| ⭐⭐ | **W8A16 手写量化** | 统一内存下带宽节省效果比 CUDA 更显著 | MPS 无 bitsandbytes，需手写 |
| ⭐⭐ | **MLX / mlx-lm** | Apple 原生推理生态，适合 4bit/8bit 模型 | 建议单独脚本 `12_mlx_benchmark.py` |
| ⭐ | **Metal Performance Shaders (MPSGraph)** | Apple 官方 C++/ObjC API，可绕过 PyTorch 写 Metal kernel | 需要 Swift/ObjC，学习成本高 |
| ❌ | **FlashAttention-2 官方包** | flash-attn 库不支持 MPS | 只能靠 PyTorch SDPA 近似 |
| ❌ | **bitsandbytes** | 不支持 MPS | 只能手写量化 |

**推荐实验顺序**（当前 Mac 上可立即执行）：
```bash
# Step 1：基础迁移，对比 CPU 基线
python 10_backend_benchmark.py --backend mps --dtype float16 --decode-steps 50

# Step 2：精度对比（查看 float32 是否更稳定）
python 10_backend_benchmark.py --backend mps --dtype float32 --decode-steps 50

# Step 3：静态 KV + torch.compile 对比
python 11_mps_compile_kv_benchmark.py --dtype float16 --decode-lengths 1 5 10 20 50 --max-cache-len 64

# Step 4：MLX 生态对比（待新增 12_mlx_benchmark.py）
mlx_lm.generate --model mlx-community/Qwen3-0.6B-4bit --prompt "人工智能的未来是" --max-tokens 100

# Step 5：性能分析 — Xcode Instruments → Metal System Trace
```


**NVIDIA GPU — CUDA 加速点全景**：

| 优先级 | 技术 | 作用位置 | 预期收益 | 实现难度 |
|--------|------|---------|---------|--------|
| ⭐⭐⭐ | **FlashAttention 2**（SDPA 已内置） | Attention QKV 计算 | Prefill 10x+，Decode 3~5x | ✅ 已完成 |
| ⭐⭐⭐ | **CUDA Graph** | Decode loop 每步 | 消除 kernel launch，30~50% | 中 |
| ⭐⭐⭐ | **device="cuda" 基础迁移** | 全模型 | GPU 并行，预计 10~30x vs CPU | 低 |
| ⭐⭐ | **torch.compile** | forward_step | Python overhead 减少 20~30% | 低 |
| ⭐⭐ | **W8A16 量化** | 全部 Linear 层 | 显存减半，带宽瓶颈消除 | 中 |
| ⭐⭐ | **INT8 KV Cache** | 长文本 KV 存储 | 显存 50%，>4K token 关键 | 中 |
| ⭐ | **Speculative Decoding** | 生成循环 | 吞吐 2~3x（batch=1 时效果有限） | 高 |
| ⭐ | **AutoGPTQ W4 量化** | Linear 层 | 显存再减半（需要 GPU 才划算）| 高 |

```python
# Step 1：基础迁移（直接用 10_backend_benchmark.py）
python 10_backend_benchmark.py --backend cuda --decode-steps 50

# Step 2：CUDA Graph（消除 kernel launch overhead）
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    logits, caches = model.forward_step(current_token, caches, pos)
# 后续 decode：g.replay()  # 零 Python overhead

# Step 3：torch.compile
model_compiled = torch.compile(model.forward_step, mode="reduce-overhead")

# Step 4：性能分析工具链
# torch.profiler  →  nsys（系统级）  →  ncu（kernel 级）
```

**推荐实验顺序**（在云端 A10/A100 上）：
1. `--backend cuda`（基础迁移，验证 GPU 加速基线）
2. 加入 `torch.compile`，对比 Decode TBT
3. CUDA Graph 手动实验，对比 kernel launch overhead
4. W8A16 量化，对比显存与速度

---

**AMD ROCm — 加速点全景**：

> ROCm 核心优势：PyTorch 代码层**零修改**，通过 HIP 桥接 CUDA API。

```bash
# 安装 ROCm 版 PyTorch（ROCm 6.0，对应 MI200/MI300/RX 7900 系列）
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

# 验证（ROCm 下 torch.cuda.is_available() 仍返回 True）
python -c "import torch; print(torch.version.hip)"  # 确认是 HIP 构建
```

| 优先级 | 技术 | 说明 | 与 CUDA 的差异 |
|--------|------|------|--------------|
| ⭐⭐⭐ | **基础迁移 `device="cuda"`** | ROCm 将 `cuda` 设备映射到 AMD GPU，代码无需改 | 仅安装包不同 |
| ⭐⭐⭐ | **SDPA / flash_attn_rocm** | SDPA 在 ROCm 走 `flash_attn_rocm` 后端 | 部分 GPU 内核支持度不如 A100 |
| ⭐⭐ | **torch.compile on ROCm** | HIP 编译，效果与 CUDA 相当 | 编译时间更长 |
| ⭐⭐ | **W8A16 手写量化** | 反量化逻辑完全可移植 | 无差异 |
| ⭐⭐ | **bitsandbytes-rocm** | 支持 INT8/NF4，需安装 ROCm 版 | `pip install bitsandbytes-rocm` |
| ⭐ | **Triton on ROCm** | `triton-rocm` 已成熟，可写自定义 kernel | Triton API 相同，后端走 HIP |
| ⭐ | **性能分析** | `rocm-smi`（显存/功耗）+ `rocprof`（kernel trace） | 类比 `nvidia-smi` + `nsys` |

**推荐实验顺序**（在 AMD GPU 机器上）：
1. 安装 ROCm PyTorch → 直接运行 `python 10_backend_benchmark.py --backend cuda`
2. 用 `rocm-smi` 监控显存占用
3. `rocprof --stats python 10_backend_benchmark.py ...` 查看 kernel 热点
4. 若 SDPA 有问题，回退到手写 matmul + softmax 对比

---

**四平台横向对比总览**：

| 维度 | CPU (当前) | Mac MPS | AMD ROCm | NVIDIA CUDA |
|------|-----------|---------|----------|------------|
| 立刻可测 | ✅ | ✅ 现在就行 | ❌ 需 AMD 硬件 | ❌ 需云实例 |
| 代码修改量 | 基准 | 几乎零 | 零 | 零 |
| FlashAttention | SDPA ✅ | SDPA Metal 近似 | flash_attn_rocm ✅ | FA2 完整 ✅ |
| 量化库支持 | 手写 | 手写 | bitsandbytes-rocm | bitsandbytes / AutoGPTQ |
| 自定义 kernel | ❌ | Metal shader (难) | Triton-ROCm ✅ | Triton / CUDA C++ ✅ |
| 性能分析工具 | — | Instruments.app | rocprof / rocm-smi | nsys / ncu |
| 统一内存优势 | — | ✅ 核心优势 | ❌ | ❌ |

#### 标准化 Benchmark 输出格式

每个平台实验结束后，输出统一格式便于横向对比：
```
平台:   Mac CPU M-series / Mac MPS / NVIDIA A10 / AMD MI250
精度:   bfloat16 / float16 / int8
TTFT:   xx ms（首 token 延迟）
TBT:    xx ms（token 间平均延迟）
Speed:  xx tokens/s
RAM/VRAM: xx MB
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
[Month 1] Phase 5-A/B/C/D/F（推理加速基础 + macOS 深挖）
  Week 1：FlashAttention SDPA 替换 + MPS 实验
  Week 2：torch.compile + 静态 KV Cache，验证 decode 图编译收益
  Week 3：max_cache_len / prompt 长度 / dtype 扫描，理解 MPS 上的 trade-off
  Week 4：MLX / mlx-lm 基线，对比 PyTorch MPS 与 Apple 原生推理生态

[Month 2] Phase 6（硬件迁移）
  Week 1：W8A16 手写量化，对比内存与速度
  Week 2：INT8 KV Cache 量化，评估长文本场景
  Week 3：NVIDIA 云实例，CUDA Graph + nsys 入门
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
- [x] FlashAttention SDPA 替换（已内置到 `10_backend_benchmark.py`，CPU 实测约 2x）
- [x] Mac MPS 后端实验（CPU/MPS、float16/float32 对比）
- [x] torch.compile + 静态 KV Cache（MPS decode TBT 25-26ms → 12-13ms）
- [ ] max_cache_len 扫描（32/64/128），评估静态 KV 固定长度成本
- [ ] 动态 KV eager vs 静态 KV eager vs 静态 KV compile 三方对比
- [ ] MLX / mlx-lm 本地推理 baseline（新增 `12_mlx_benchmark.py`）
- [ ] W8A16 权重量化，内存降至 ~2.4GB
- [ ] INT8 KV Cache 量化，为长文本场景做准备
- [ ] NVIDIA GPU 迁移 + CUDA Graph（`python 10_backend_benchmark.py --backend cuda`）
- [ ] torch.profiler / nsys 性能分析
- [ ] AutoGPTQ W4 量化 + PPL 评估
- [ ] AMD ROCm 迁移验证
- [ ] 批推理 + Prefix Caching 实验
- [ ] 阅读 vLLM PagedAttention 实现
