# 激活函数：SiLU / SwiGLU 原理与两模型的使用

> 关联代码：`07_mlp.py` / `08_generate.py` / `09_qwen3_0_6b_generate.py`

---

## 1. 什么是激活函数？为什么需要它？

### 核心问题：线性模型的表达力上限

如果神经网络只有线性层（矩阵乘法），无论堆多少层，整体仍然等价于一个单层线性变换：

$$W_n \cdots W_2 W_1 x = W_{\text{fused}} x$$

**激活函数的作用**：在线性层之间插入**非线性变换**，打破线性组合的限制，使网络能够拟合任意复杂的函数（通用近似定理）。

没有激活函数，MLP 无法区分"猫"和"狗"；有了激活函数，才能在高维空间中划出复杂的决策边界。

---

## 2. 两个模型使用的激活函数

### Qwen3-0.6B 和 Qwen3.5-0.8B 使用完全相同的方案：**SwiGLU**

```python
# config.json 中明确标注
"hidden_act": "silu"  # SiLU 是 SwiGLU 的核心激活组件
```

两个模型在激活函数的选择上**没有任何区别**，差异只在 MLP 的维度参数：

| 模型 | `hidden_size` | `intermediate_size` | 扩张比 |
|------|--------------|---------------------|--------|
| Qwen3-0.6B | 1024 | **3072** | 3.0× |
| Qwen3.5-0.8B | 1024 | **3584** | 3.5× |

---

## 3. SiLU 激活函数

SiLU（Sigmoid Linear Unit）是 SwiGLU 的核心组件，也称为 **Swish** 激活：

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

### 与经典激活函数的对比

| 激活函数 | 公式 | 特点 |
|----------|------|------|
| ReLU | $\max(0, x)$ | 简单快速，但负区间梯度为 0（"死亡 ReLU"）|
| GELU | $x \cdot \Phi(x)$ | 平滑，效果好，但计算含高斯分布近似 |
| **SiLU** | $x \cdot \sigma(x)$ | 平滑、无上界、负区间保留少量梯度，无需查表 |

```
    SiLU(x)
      │
   1 ─┤          ╱────────────
      │        ╱
   0 ─┤──────╱
      │    ╱ ← 负区间保留少量负值（约 -0.28 处最小）
  -0.3┤  ╱
      └──────────────────────── x
        -3  -2  -1   0   1   2
```

**关键特性**：
- **平滑可微**：处处可导，梯度下降时不会卡在不可导点
- **非单调**：在负区间有轻微下凸（约 x=-1.28 处最小值 ~-0.28），保留了少量负值信息
- **无上界**：正区间线性增长，不像 Sigmoid 那样饱和，缓解梯度消失

---

## 4. SwiGLU：门控线性单元

SwiGLU = **Swish** + **GLU（Gated Linear Unit）**

### 从标准 MLP 到 SwiGLU

**标准 MLP（如早期 Transformer）**：
$$\text{MLP}(x) = \text{ReLU}(xW_1 + b_1) W_2 + b_2$$

**SwiGLU MLP（Qwen3/Qwen3.5 使用）**：
$$\text{SwiGLU}(x) = \big(\text{SiLU}(xW_{\text{gate}}) \odot xW_{\text{up}}\big) W_{\text{down}}$$

其中 $\odot$ 是逐元素乘法（Hadamard product）。

### 三个矩阵的分工

```
输入 x [1, 1024]
    │
    ├──→ gate_proj [1024→3072/3584] → SiLU(·) ──┐
    │                                              │ 逐元素相乘
    └──→ up_proj   [1024→3072/3584] ─────────────┘
                                                   │
                                              [1, 3072/3584]
                                                   │
                                             down_proj [3072/3584→1024]
                                                   │
                                              输出 [1, 1024]
```

**gate_proj（门控投影）**：经过 SiLU 激活后，产生一个值域在 `[-0.28, +∞)` 的"门控信号"，决定哪些特征应该被强调、哪些应该被抑制。

**up_proj（上升投影）**：将特征无激活地投影到高维空间，保留原始线性信息。

**逐元素乘法**：门控信号像一个动态滤波器，对高维特征进行**选择性激活**：值接近 0 的维度被抑制，值大的维度被保留。

**down_proj（下降投影）**：将过滤后的高维特征压缩回主干道维度。

### 代码实现

```python
# 07_mlp.py / 08_generate.py / 09_qwen3_0_6b_generate.py 共用相同结构
class QwenSwiGLU_MLP(nn.Module):
    def forward(self, x):
        gate = self.gate_proj(x)          # [B, seq, intermediate_size]
        up   = self.up_proj(x)            # [B, seq, intermediate_size]

        activated_gate = F.silu(gate)     # 门控激活

        fused = activated_gate * up       # 逐元素相乘（核心！）

        return self.down_proj(fused)      # 压缩回 hidden_size
```

---

## 5. SwiGLU 的特点：为什么比 ReLU 好？

### 参数效率
SwiGLU 有三个矩阵（gate / up / down），而传统双层 MLP 只有两个。为了参数量对等，现代大模型将 `intermediate_size` 调整为约 `8/3 × hidden_size`（而非 4×）：

- Qwen3-0.6B：1024 × 3.0 = 3072 ≈ `8/3 × 1024 = 2730`（取了更保守的整数）
- Qwen3.5-0.8B：1024 × 3.5 = 3584

### 门控机制的语义意义
MLP 层是模型"记忆知识"的地方（研究表明 FFN 层存储了大量事实性知识）。门控机制使得：
- 模型可以根据当前输入**动态决定激活哪些"知识神经元"**
- 而不是简单地用 ReLU 做硬截断（要么全激活，要么完全抑制）

这种"软门控"在 LLM 上的实验结果显示，相比 ReLU/GELU 能提升约 **0.5~1 ppl** 的语言模型质量。

---

## 6. 激活函数对推理性能的影响

### 计算量分析

对于 decode 阶段（单 token，`[1, 1024]` 输入）：

| 操作 | 计算量 | 说明 |
|------|--------|------|
| gate_proj | 1024 × 3072 ≈ **3.1M** FLOP | 矩阵乘法，主要耗时 |
| up_proj | 1024 × 3072 ≈ **3.1M** FLOP | 矩阵乘法，主要耗时 |
| SiLU（逐元素）| 3072 ≈ **3K** FLOP | 可忽略 |
| 逐元素相乘 | 3072 ≈ **3K** FLOP | 可忽略 |
| down_proj | 3072 × 1024 ≈ **3.1M** FLOP | 矩阵乘法，主要耗时 |

**结论**：激活函数本身（SiLU 的 `sigmoid` 计算）几乎不耗时。MLP 的耗时**完全来自三次矩阵乘法**，SwiGLU 带来的额外代价是比标准 FFN 多了一次矩阵乘法（gate_proj）。

### 是否是推理瓶颈？

**是的**——MLP 整体是推理中最重要的耗时来源之一：
- Decode 阶段：MLP 通常占总时间的 **40~60%**（比 Attention 更重！）
- 原因：三次大型矩阵乘法涉及大量 HBM 数据搬运（权重读取）
- 这是**权重量化（W8A16/W4A16）的最高价值目标**：量化 gate/up/down 权重，使带宽需求减半

### SwiGLU 的融合优化潜力

```python
# 未优化（两次 HBM 读写）
gate = self.gate_proj(x)   # HBM 读 gate_proj 权重
up   = self.up_proj(x)     # HBM 读 up_proj 权重
out  = F.silu(gate) * up   # 额外的 element-wise 操作

# 融合优化（Triton kernel）：gate_proj + silu + element-wise 合并为一个 kernel
# 将中间结果保留在 SRAM 中，避免写回 HBM
# liger-kernel 中的 LigerSiLUMulFunction 即为此优化
```

---

## 7. 两模型 MLP 参数量对比

| 模型 | gate_proj | up_proj | down_proj | MLP 总参数 |
|------|-----------|---------|-----------|------------|
| Qwen3-0.6B (×28层) | 1024×3072 | 1024×3072 | 3072×1024 | 28 × 3 × 3.1M ≈ **261M** |
| Qwen3.5-0.8B (×24层) | 1024×3584 | 1024×3584 | 3584×1024 | 24 × 3 × 3.6M ≈ **262M** |

MLP 参数量约占两个模型总参数量的 **45~50%**，是参数最密集的模块，也是量化收益最大的部分。
