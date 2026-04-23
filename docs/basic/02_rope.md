# RoPE：旋转位置编码原理与 Qwen 的局部旋转设计

> 关联代码：`04_rope.py` / `08_generate.py` / `09_qwen3_0_6b_generate.py`

---

## 1. 用途：Transformer 为什么需要位置编码？

### 核心问题：Attention 天生没有位置感

标准的 Self-Attention 计算：

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

这个公式中，**词的顺序信息完全不存在**。无论把"我爱你"还是"你爱我"输入，只要词的集合相同，Attention 矩阵的值就完全一样。

**位置编码的作用**：将每个 token 的位置信息（它是序列中的第几个词）"注入"到 Q 和 K 向量中，使得 Attention 能感知到词序。

### 位置编码的演进

| 方案 | 代表模型 | 原理 | 缺点 |
|------|----------|------|------|
| 绝对位置编码（APE） | BERT、GPT-2 | 给每个位置一个固定向量直接相加 | 无法外推到训练时未见过的长度 |
| 相对位置编码（RPE） | T5 | 在 Attention 中加入相对距离偏置 | 实现复杂，推理时有额外开销 |
| **RoPE** | LLaMA、Qwen、Mistral | 用旋转矩阵将位置编码融入 Q/K 的内积中 | 近乎无缺点，成为主流 |

---

## 2. RoPE 的核心思想：用旋转代替相加

### 数学直觉

RoPE 的核心洞察：**两个向量的内积，可以通过对它们各自旋转相同角度来编码相对距离。**

对于位置 $m$ 的 Query 向量 $q$ 和位置 $n$ 的 Key 向量 $k$，RoPE 做的事情是：

$$q_m' = R(\theta_m) \cdot q_m, \quad k_n' = R(\theta_n) \cdot k_n$$

其中 $R(\theta)$ 是一个旋转矩阵，旋转角度与位置有关。

旋转后两者的内积：
$$\langle q_m', k_n' \rangle = \langle R(\theta_m) q_m, R(\theta_n) k_n \rangle = f(m - n)$$

**结论**：内积只与位置差 $(m - n)$ 有关，天然编码了相对位置，而不是绝对位置。这使得 RoPE 在长序列上的外推能力优于绝对位置编码。

### 实现方式：复数旋转

在实现时，将 head_dim 维向量两两配对，每对 $(x_{2i}, x_{2i+1})$ 看作复平面上的一个点，旋转角度为：

$$\theta_i = \frac{1}{\text{base}^{2i/d}}$$

其中 `base` 决定了频率范围（Qwen3 用 `1,000,000`，Qwen3.5 用 `10,000,000`）。

实际代码中用"旋转半部"操作替代矩阵乘法：
```python
# 04_rope.py
def _rotate_half(self, x):
    # [x1, x2, x3, x4] → [-x3, -x4, x1, x2]
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

# 旋转公式：(x * cos θ) + (rotate_half(x) * sin θ)
q_rot_out = (q_rot * cos) + (self._rotate_half(q_rot) * sin)
```

---

## 3. 为什么 Qwen3.5 只旋转 25% 的维度？

### Qwen3 vs Qwen3.5 的 RoPE 对比

| 模型 | `partial_rotary_factor` | 旋转维度 | base（基础频率）|
|------|------------------------|----------|----------------|
| Qwen3-0.6B | **1.0**（全维度） | 128 / 128 | 1,000,000 |
| Qwen3.5-0.8B（Full Attn 层）| **0.25**（局部） | 64 / 256 | 10,000,000 |

### 为什么要局部旋转？—— 三个动机

#### 动机一：head_dim 已经被"扩维"了
Qwen3.5 的 Full Attention 中，`head_dim = 256`，远大于 Qwen3 的 128。这是因为 Gated Attention 需要更大的投影空间来存放门控特征。

但 RoPE 的旋转是基于**频率编码**的，维度越高，高频部分的信息越冗余——后 75% 的维度在实验中对位置信息的贡献已经可以忽略。全部旋转反而浪费计算。

#### 动机二：计算效率
仅旋转前 25%（64 维），跳过后 75%（192 维）：
- 计算量：从 `O(head_dim)` 降为 `O(0.25 × head_dim)`
- 对 8 个 Q 头 + 2 个 K 头 × 256 维：每步 decode 节省约 **75% 的 RoPE 计算**

#### 动机三：混合架构的特殊需求
Qwen3.5 的 Linear Attention 层（18 层）完全**不使用 RoPE**——状态空间模型（Mamba/GLA）用 conv1d 的局部感知替代了位置编码。只有 Full Attention 层（6 层）才需要 RoPE。

因此全注意力层的 RoPE 承担的"全局位置感知"压力已经大幅降低（只有 25% 的层数在用），进一步削减 RoPE 的维度是合理的工程权衡。

### 代码对照

```python
# Qwen3-0.6B（09_qwen3_0_6b_generate.py）：全维度旋转
class QwenRoPE:
    def __init__(self, head_dim=128, base=1000000.0):
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        # rotary_dim = head_dim = 128，全部旋转

    def apply_rope(self, q, k, position_ids):
        q_out = (q * cos) + (self._rotate_half(q) * sin)  # 整个 q 都参与旋转
        k_out = (k * cos) + (self._rotate_half(k) * sin)
        return q_out, k_out


# Qwen3.5-0.8B（08_generate.py）：25% 局部旋转
class QwenRoPE:
    def __init__(self, head_dim=256, partial_rotary_factor=0.25, base=10000000.0):
        self.rotary_dim = int(head_dim * partial_rotary_factor)  # 只旋转 64 维

    def apply_rope(self, q, k, position_ids):
        q_rot  = q[..., :self.rotary_dim]   # 前 64 维参与旋转
        q_pass = q[..., self.rotary_dim:]   # 后 192 维原封不动

        q_rot_out = (q_rot * cos) + (self._rotate_half(q_rot) * sin)
        return torch.cat((q_rot_out, q_pass), dim=-1)  # 拼回去
```

---

## 4. RoPE 对推理的性能影响

### 计算量
RoPE 的计算是**逐元素操作**（element-wise），不涉及矩阵乘法：
- 每次 decode：`cos/sin 查表 + 乘法 + 拼接`
- 计算量极小，约为一次 q_proj 的 **1/100**

### 是否是推理瓶颈？
**不是主要瓶颈**，但有间接影响：
- cos/sin 缓存（`cos_cached` / `sin_cached`）预先计算并驻留内存，decode 时仅查表，速度极快
- 在极长序列（>32K）下，position_id 索引的内存访问模式可能带来轻微的缓存未命中

### 外推能力（长文本推理的关键）
RoPE 的频率编码天然支持**长度外推**，但实际效果取决于 `base` 的选择：

| 模型 | base | 训练序列长度 | 外推能力 |
|------|------|------------|----------|
| Qwen3-0.6B | 1,000,000 | 32K | 可外推到 ~64K |
| Qwen3.5-0.8B | 10,000,000 | 256K | 可外推到 ~512K |

base 越大，低频维度的变化越慢，模型能"记住"更远的相对距离。这是 Qwen3.5 使用 `base=10,000,000` 的根本原因。

---

## 5. 关键验证实验（来自 04_rope.py）

```python
# 验证局部旋转是否正确：后 192 维应完全不变
diff = torch.abs(q_out[..., 64:] - q_dummy[..., 64:]).sum().item()
print(f"后192维度的误差总和: {diff}")
# 预期输出：0.0（完全一致）
```

这个验证确保了 `q_pass` 部分在 `torch.cat` 前后没有被误修改，是实现正确性的重要保障。
