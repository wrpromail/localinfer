# RMSNorm：归一化层原理与推理特性

> 关联代码：`03_rmsnorm.py` / `08_generate.py` / `09_qwen3_0_6b_generate.py`

---

## 1. 用途：为什么需要归一化层？

### 核心问题：数值不稳定性
大模型有数十层堆叠的矩阵乘法。每经过一层，特征值的量级（尺度）就会被放大或缩小。经过 24~28 层后，若不加干预，数值会爆炸（趋向无穷）或消失（趋向零），梯度也会随之失控。

**归一化层的作用**：在每一层的关键位置，将特征向量的数值尺度"拉平"到合理区间，使深层网络能够稳定训练和推理。

### LayerNorm vs RMSNorm

| 特性 | LayerNorm（BERT 时代） | RMSNorm（现代大模型） |
|------|----------------------|----------------------|
| 计算均值 | ✅ 需要 | ❌ 省略 |
| 计算方差 | ✅ 需要 | ✅ 需要（均方） |
| 可学习参数 | γ（缩放）+ β（偏移） | γ（缩放）仅此一个 |
| 计算量 | 较高 | 约快 **15~20%** |
| 精度 | 略高 | 在大模型中实测无明显差异 |

**RMSNorm 的数学公式**：

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \varepsilon}} \cdot \gamma$$

省略均值计算的理由：实验发现，在 Transformer 中均值的"重新中心化"作用可以被残差连接（Residual Add）隐式替代，省掉它并不影响模型质量。

---

## 2. Qwen3 vs Qwen3.5 的 RMSNorm 差异

两个模型在 RMSNorm 的**数学公式和代码实现上完全一致**，差异在于使用位置和数量：

### Qwen3-0.6B（28 层，纯 Full Attention）

每层包含 **2 个** 标准 RMSNorm：
```
input_layernorm          # Attention 前，归一化主干道
post_attention_layernorm # MLP 前，归一化 Attention 输出后的残差
```

额外的 **q_norm / k_norm**（每层 Full Attention 内部）：
```
q_norm(head_dim=128)  # 对 Q 矩阵按 head 维度归一化
k_norm(head_dim=128)  # 对 K 矩阵按 head 维度归一化
```
这两个 norm 作用于 RoPE **之前**，防止 Q/K 内积在高维下数值爆炸。

### Qwen3.5-0.8B（24 层，混合架构）

每层的 RMSNorm 格局更复杂：

**Full Attention 层（每 4 层一个）**：同 Qwen3，含 input_norm + post_attn_norm + q_norm + k_norm

**Linear Attention 层（每 4 层中的 3 个）**：
```
input_layernorm          # 主干道前归一化
post_attention_layernorm # MLP 前归一化
linear_attn.norm         # 额外！对 GLA 的 head 输出归一化（针对 head_dim=128）
```
`linear_attn.norm` 是 Mamba 变体特有的，作用在 RNN 状态机的输出上，防止状态矩阵的值域在跨步更新中漂移。

**总参数量对比**：

| 模型 | 每层 RMSNorm 数 | 总 RMSNorm 参数量 |
|------|----------------|-------------------|
| Qwen3-0.6B | 4个（2主 + q_norm + k_norm）| 28 × 4 × 1024 × 2B ≈ **229 KB** |
| Qwen3.5-0.8B | 5~6个（含 linear_attn.norm）| 24 × 5 × 各不同dim ≈ **~280 KB** |

RMSNorm 参数量极小，占总参数量不足 **0.05%**。

---

## 3. 精度敏感性：是否需要高精度？

**结论：高度敏感，必须用 float32 计算，但参数本身可以低精度存储。**

### 为什么计算时必须 float32

```python
# 03_rmsnorm.py 中的关键代码
def forward(self, hidden_states):
    input_dtype = hidden_states.dtype          # 记录原始精度（bfloat16）
    hidden_states = hidden_states.to(torch.float32)  # ← 必须升精度！
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)  # ← 计算完再降回去
```

**原因**：均方计算（`.pow(2).mean()`）是一个累加操作。在 bfloat16（精度约 3 位有效小数）下，对 1024 个元素平方累加时，尾数截断误差会叠加放大，导致 `variance` 结果显著偏差。float32 提供约 7 位有效小数，可以保证数值稳定。

### 精度敏感场景对比

| 场景 | 是否敏感 | 说明 |
|------|----------|------|
| 权重 `γ` 存储精度 | 不敏感 | bfloat16 存储完全够用 |
| 方差计算 | **高度敏感** | 必须 float32，否则数值偏差 |
| 深层模型（28层以上）| **更敏感** | 误差随层数指数积累 |
| Mamba/GLA 的 `linear_attn.norm` | **极度敏感** | 状态机跨步连乘，误差放大效应最强 |

---

## 4. 推理耗时：RMSNorm 是瓶颈吗？

**结论：单独看很快，但累积影响显著，是算子融合的重要优化目标。**

### 量化分析

对于 Qwen3-0.6B，每个 RMSNorm 的计算量：
- 输入：`[1, 1024]`（decode 阶段单 token）
- 操作：平方 + 均值 + rsqrt + 乘法 = 约 **4096 次浮点运算**
- 对比一次 `q_proj` 矩阵乘法：`[1, 1024] × [2048, 1024]` = 约 **2M 次浮点运算**

单次 RMSNorm 的计算量约为一次线性投影的 **1/500**。但：

- 每层有 4 个 RMSNorm，28 层共 **112 次**
- 每次 RMSNorm 都需要 bf16 → f32 → bf16 的**类型转换**，这是 HBM 带宽的消耗
- 在 CPU 上，类型转换的内存搬运开销甚至超过计算本身

### 在 `torch.profiler` 中的典型表现

在 decode 阶段，RMSNorm 系列算子通常占总推理时间的 **5~10%**。这看似不多，但因为它频繁读写 HBM，是**算子融合（Kernel Fusion）最高价值的目标**之一：

```
优化前：Linear → [HBM写] → RMSNorm → [HBM读写] → Linear
优化后：Linear + RMSNorm 融合为一个 kernel → 消除中间 HBM 读写
```

这正是 liger-kernel 和 TensorRT-LLM 中 `fused_add_rmsnorm` 算子存在的原因。

---

## 5. 代码速查

```python
# localinfer 中的标准实现（03_rmsnorm.py / 08 / 09 共用相同逻辑）
class QwenRMSNorm(nn.Module):
    def __init__(self, hidden_size=1024, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # γ，唯一可学习参数
        self.variance_epsilon = eps                           # 防止除零

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)      # 升精度计算
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)   # 降回原精度
```
