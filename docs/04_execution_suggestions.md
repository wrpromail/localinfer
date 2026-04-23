# Qwen3.5-0.8B 本地推理执行建议

基于您的核心目标（从零实现前向传播、连续生成 10 个 Token、手写 KVCache，以及探索推理加速方案），我为您梳理了以下的执行建议和技术路径：

## 1. 语言与框架选择建议
由于您的目标包含“尝试主要的推理加速方案”（如 FlashAttention、算子融合等），我强烈建议使用 **PyTorch** 作为基础框架。
- **前期验证阶段**：使用纯 PyTorch 的 Tensor 操作来实现架构，验证正确性（这一步非常适合用 Python 快速跑通）。
- **后期加速阶段**：使用 **Triton** 或 **CUDA** 来编写自定义算子（Kernel），并集成到 PyTorch 中。这样既能享受 Python 的灵活性，又能深入探索底层加速。

## 2. 阶段一：解析与加载 (Weight Loading)
- **不要直接读取全部权重到内存**：现代模型都是以 `safetensors` 格式分发。建议使用 `safetensors` 库的 `safe_open` 和 `framework="pt"`，这底层是基于内存映射 (mmap) 的，能极大减小内存峰值。
- **解析 `config.json`**：在动手写代码前，先详细研究官方权重的 `config.json`，确认关键超参，例如：是否使用了 GQA（Grouped-Query Attention）、RoPE 的基数、RMSNorm 的 epsilon 等。

## 3. 阶段二：朴素前向传播 (Naive Forward Pass)
先不要考虑 KVCache，写一个只能一次性处理全部输入，然后输出下一个 Token 的“朴素版”模型。
**核心实现模块：**
1. **Embedding**: Token Embedding。
2. **RMSNorm**: 现代大模型标配的层归一化。
3. **RoPE (旋转位置编码)**: 弄清楚 RoPE 是如何应用在 Query 和 Key 上的。
4. **Attention**: 根据 `config.json` 实现 MHA（多头）或 GQA（分组查询注意力）。如果是 GQA，需要实现 Key/Value 头的 broadcast（广播）。
5. **MLP (SwiGLU)**: 实现带门控的激活函数机制。

*验证方式*：将您手写的输出 Logits 与 HuggingFace 官方 `transformers` 库的前向传播 Logits 进行对比，确保绝对误差（如 `torch.max(torch.abs(out1 - out2))`）在 $10^{-4}$ 以内。

## 4. 阶段三：引入 KVCache 与自回归 (Autoregressive Generation)
在确认朴素版前向无误后，重构代码。
- **数据结构**：在每一层的 Attention 模块中，预分配一个足够大的 Tensor（如 `[batch_size, num_kv_heads, max_seq_len, head_dim]`）作为缓存池，而不是每次动态 `concat`。
- **更新逻辑**：每次只传入当前步的单个 token，计算出 `q, k, v` 后，将 `k, v` 写入到预分配的 Tensor 的指定位置。然后使用所有的缓存进行 Attention 计算。
- **解码循环**：写一个 `while` 或 `for` 循环，将当前步选出的 token id 重新输入给模型，循环 10 次。

## 5. 阶段四：推理加速方案探索 (核心重头戏)
当正确连续生成 10 个 token 后，真正的挑战和乐趣开始了。大模型推理特别是 Batch Size = 1 的自回归生成（Decode 阶段），是一个典型的**访存密集型（Memory-bound）**任务。

我建议您按以下顺序探索：
1. **权重量化 (Weight-Only Quantization)**
   - *为什么*：瓶颈在于将权重从显存搬运到计算单元。如果权重用 INT8 甚至 INT4 表示，显存读取带宽需求直接减半甚至减少 75%。
   - *怎么做*：您可以尝试手写一个最简单的对称线性量化算法（如类似 GPTQ 或 AWQ 的简化版），将 Linear 层的权重转为 INT8。在计算时反量化回 FP16（即 W8A16 计算）。
2. **算子融合 (Operator Fusion)**
   - *为什么*：每次调用 PyTorch 的小算子（比如做完 RoPE 后再和 Q 乘，或者 SwiGLU 里的两个线性层和 Silu）都会引发一次显存的读写和 Kernel 启动开销。
   - *怎么做*：使用 **Triton** 编写自定义 Kernel，例如把 `RMSNorm + Linear` 融合成一个 Kernel，或者把 `Silu + 逐元素乘法` 融合。
3. **针对 Decode 阶段的 Attention 加速**
   - *为什么*：标准的 FlashAttention 主要是针对长序列的 Prefill（预填充）阶段加速。对于每次只生成 1 个 token 的 Decode 阶段，`Query` 长度为 1，此时标准的 FlashAttention 并不能完全发挥作用。
   - *怎么做*：您可以去了解并尝试实现 **FlashDecoding** 的核心思想（沿 Sequence 维度切分 KV 并行计算，最后做一次 reduction），这能极大优化长上下文下的解码速度。

## 下一步行动
建议您先创建一个 `scripts/` 目录，我们首先写一个脚本去 HuggingFace (或 ModelScope) 下载 `qwen3.5-0.8B` 的权重和 `config.json`，您觉得如何？
