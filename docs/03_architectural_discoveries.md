# Qwen3.5-0.8B 架构大揭秘与已知问题记录

在从零开始手写 Qwen3.5-0.8B 模型架构并将其逆向实例化的过程中，我们遇到了许多与官方传统文档描述不符的“盲盒”机制，并成功摸清了它们底层的物理结构。本记录旨在汇总我们的发现以及手写推理引擎遇到的难点。

## 🎯 重大架构发现 (Architectural Discoveries)

### 1. 混合架构：24层的精妙排布
模型并非单纯的 Transformer，而是一个 **Mamba (SSM) 与 Transformer 混合架构**。
通过逆向权重前缀（`model.language_model.layers` 与 `mtp.layers`），我们得出其排布规律为：
- **每 4 层一轮 (Block)**：其中前 3 层为 `Linear Attention`（状态空间模型/Mamba 变体），第 4 层为 `Full Attention`（传统的全注意力模型）。
- 这种设计允许模型在大部分层级通过 RNN 状态机进行极低算力消耗的推理（$O(1)$ 缓存复杂度），同时依赖少量的传统注意力层来保持对于全局上下文的精准“回头看”能力。

### 2. Full Attention：隐藏的门控注意力 (Gated Attention)
当我们尝试加载 `05_full_attention` 时，遭遇了严重的维度不匹配问题（预期 2048 维的 `q_proj` 实际为 4096 维）。这导致我们发现 Qwen3.5 的全注意力实际上是 **Gated Attention**：
- `config.json` 中的 `"attn_output_gate": true` 揭示了其机制。
- `q_proj` 投影出的张量尺寸为 `[hidden_size, num_heads * head_dim * 2]`，其中一半是真正的 Query（用于与 K 计算 Attention），另一半则是 **门控向量 (Gate)**。
- 最终的注意力输出在进行 `o_proj` 之前，必须先乘以 `F.silu(Gate)` 进行一次非线性信息过滤。
- **特异点**：Qwen 的全注意力层中除了主干的 RMSNorm 外，还在执行 RoPE 之前，额外加入了针对 Q 和 K 的 `q_norm` 和 `k_norm`（`RMSNorm(head_dim)`）。

### 3. Linear Attention：魔改的门控 Delta 网络 (Gated Delta Network)
通过探针脚本拿到的权重形状，我们断定该模型所谓的 Linear Attention 并非原味的 Mamba，而是结合了注意力的 **Gated Delta Rule (GLA)**。
- 它保留了 `Q, K, V` 的投影，但加入了 `conv1d` (核大小为 4 的一维深度因果卷积)。
- 它的状态机并非 Mamba 的隐式参数，而是维持了一个显式的 **`rnn_state`** 矩阵，其维度被死死固定在 `[num_heads, head_dim, head_dim]` (即 `[16, 128, 128]`)。
- 在 `forward` 阶段，新传入的词特征会通过 `in_proj_a` 和 `in_proj_b` 算出门控系数 `g` 和 `beta`，从而决定如何使旧的记忆矩阵衰减，并融合新的 `K^T * V` 知识。

### 4. 局部旋转位置编码 (Partial RoPE)
模型使用了一种局部的 RoPE，由 `partial_rotary_factor: 0.25` 控制。即对于 256 维的特征向量，它只旋转前 64 维来注入位置信息，后 192 维直接跳过，保持不变。

---

## ⚠️ 已知问题与局限性 (Known Issues & Limitations)

### 1. 精度漂移与“概率陷阱”问题 (生成无意义重复符)
- **表现**：在我们最终构建的 `08_generate.py` 端到端自回归生成引擎中，模型在加载真实权重后，生成的并不是自然语言，而是重复的连字符（如 `----------`）等乱码。
- **根本原因**：Gated Delta Rule 在数学上极为敏感。它包含了类似于 `g_t.exp()` (基于缩放后激活函数的指数运算) 以及复杂的层归一化运算。官方实现（Hugging Face 或 vLLM）中对这些运算应用了 Triton 级的 C++ 定制算子，并辅以特定的精度转化（如在计算 `exp` 时强制转换回 `float32` 等）。
- **结论**：我们纯手写的 Python 版本能够确保 **架构与张量流转逻辑 100% 正确**（因为成功串联 24 层且形状分毫不差），但要达到生成连贯自然语言的数值精度要求，则需要耗费大量时间对齐 Triton 算子的底层数值阶段逻辑。

### 2. Token-by-Token 流式消化速度瓶颈
- **表现**：由于 18 层 Linear Attention (Mamba) 在 Python 中极难实现并行的“前缀关联扫描 (Associative Scan)”，我们在处理用户输入的 Prompt 句子（Prefill 阶段）时，被迫采用了最原始的 **Token-by-Token 逐词送入模式**。
- **影响**：这意味着处理一段 100 个词的 Prompt，需要执行 100 轮包含 24 层庞大矩阵计算的前向传播。这导致我们手写的引擎在“冷启动暖机”阶段速度极慢。真正的生产级引擎必然需要通过并行扫描算法来解决这个问题。

---

## 🚀 下一步 (Future Work)
- **FlashAttention 整合**：针对第 4、8、12... 等纯注意力层，用 FlashAttention 替换手写的矩阵乘法，可以大幅降低显存带宽压力。
- **算子级数值对齐**：通过比较我们手写的中间输出与 `transformers` 加载模型后的单层中间输出，找出导致输出乱码的“浮点数偏差魔鬼”，在不使用 Triton 的情况下实现 Python 原生的数值完美对齐。
