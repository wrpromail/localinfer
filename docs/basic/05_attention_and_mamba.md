下面给你一条**从数学原理到 macOS 实践**的学习路线，目标是理解：

1. Transformer 里的**全注意力 full attention / self-attention**
2. Mamba 里的**状态空间模型 State Space Model, SSM**
3. 它们为什么复杂度不同、记忆方式不同、适合的序列长度不同
4. 如何在 Mac 上用 Python/PyTorch 做小实验

---

## 0. 先建立一个总览

你可以先把二者感性地区分为：

| 模型机制 | 核心思想 | 序列建模方式 | 复杂度直觉 |
|---|---|---|---|
| Self-Attention | 每个 token 和所有 token 两两交互 | 显式比较所有位置 | 通常 \(O(n^2)\) |
| SSM / Mamba | 用一个隐藏状态沿序列递推 | 像动态系统一样逐步更新状态 | 通常接近 \(O(n)\) |

Transformer 的注意力更像是：

> “我现在这个词，要回头看整段话里所有词，并算每个词的重要程度。”

Mamba / SSM 更像是：

> “我维护一个状态，读一个 token 更新一次状态，这个状态压缩了之前的信息。”

---

# 1. 数学基础准备

建议你先补下面几块，不需要学到非常深，但要能读公式。

## 1.1 线性代数

重点掌握：

- 向量、矩阵乘法
- 点积
- 矩阵转置
- 特征值和特征向量的直觉
- 矩阵指数 \(e^{A}\)
- 对角化

尤其要熟悉：

\[
QK^\top
\]

这种形式，因为 attention 里面会大量出现。

---

## 1.2 概率与 softmax

Attention 中会用 softmax 把相似度变成权重：

\[
\operatorname{softmax}(x_i)
=
\frac{e^{x_i}}{\sum_j e^{x_j}}
\]

你要理解：

- 为什么 softmax 输出非负
- 为什么所有权重和为 1
- 为什么大的分数会被强调
- 温度参数如何影响分布尖锐程度

---

## 1.3 微积分与优化

深度学习里的参数都是通过梯度下降学习的，所以至少要理解：

- 导数
- 偏导数
- 链式法则
- 梯度下降
- 反向传播的基本直觉

---

## 1.4 动态系统与差分方程

这是理解 SSM/Mamba 的关键。

最简单的一阶递推：

\[
h_t = a h_{t-1} + b x_t
\]

其中：

- \(x_t\)：当前输入
- \(h_t\)：当前隐藏状态
- \(a\)：保留多少历史信息
- \(b\)：吸收多少当前输入

这就是 SSM 的核心雏形。

---

# 2. 学习 Self-Attention 的数学原理

## 2.1 从 Query、Key、Value 开始

给定输入序列：

\[
X \in \mathbb{R}^{n \times d}
\]

其中：

- \(n\)：序列长度
- \(d\)：每个 token 的维度

Transformer 会学习三个矩阵：

\[
W_Q, W_K, W_V
\]

然后得到：

\[
Q = XW_Q
\]

\[
K = XW_K
\]

\[
V = XW_V
\]

直觉上：

- Query：我想找什么信息
- Key：我有什么信息可被匹配
- Value：真正要被取走的信息内容

---

## 2.2 注意力分数

计算每个 token 和其他 token 的相似度：

\[
S = QK^\top
\]

如果：

\[
Q \in \mathbb{R}^{n \times d_k}
\]

\[
K \in \mathbb{R}^{n \times d_k}
\]

那么：

\[
S \in \mathbb{R}^{n \times n}
\]

这就是为什么 self-attention 是 \(O(n^2)\) 的核心原因：  
每个位置都要和每个位置比较。

---

## 2.3 缩放点积注意力

实际公式是：

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V
\]

为什么除以 \(\sqrt{d_k}\)？

因为当维度 \(d_k\) 较大时，点积数值容易变大，softmax 会变得过于尖锐，导致梯度不稳定。

---

## 2.4 Multi-Head Attention

多头注意力就是并行做多组注意力：

\[
\operatorname{head}_i
=
\operatorname{Attention}
(
XW_i^Q,
XW_i^K,
XW_i^V
)
\]

然后拼接：

\[
\operatorname{MultiHead}(X)
=
\operatorname{Concat}(\operatorname{head}_1,\dots,\operatorname{head}_h)W^O
\]

直觉上：

> 不同的 head 可以学习不同类型的关系，比如语法关系、指代关系、局部关系、长距离依赖等。

---

# 3. 学习 SSM / Mamba 的数学原理

Mamba 的根基是状态空间模型。

## 3.1 连续时间 SSM

经典连续时间状态空间模型：

\[
\frac{dh(t)}{dt}
=
Ah(t) + Bx(t)
\]

\[
y(t)
=
Ch(t) + Dx(t)
\]

其中：

- \(x(t)\)：输入
- \(h(t)\)：隐藏状态
- \(y(t)\)：输出
- \(A\)：状态转移矩阵
- \(B\)：输入投影矩阵
- \(C\)：输出投影矩阵
- \(D\)：跳跃连接，表示输入直接影响输出

可以理解为：

> 输入 \(x(t)\) 驱动一个动态系统，这个系统的内部状态 \(h(t)\) 随时间演化，并产生输出 \(y(t)\)。

---

## 3.2 离散化

语言模型处理的是离散 token，所以要把连续系统变成离散系统：

\[
h_t = \overline{A}h_{t-1} + \overline{B}x_t
\]

\[
y_t = Ch_t + Dx_t
\]

其中：

\[
\overline{A} = e^{\Delta A}
\]

\[
\overline{B}
=
(\Delta A)^{-1}(e^{\Delta A}-I)\Delta B
\]

如果你暂时看不懂这个公式，不要紧。先记住：

> 连续动态系统通过步长 \(\Delta\) 变成了离散递推系统。

---

## 3.3 SSM 的卷积形式

递推展开：

\[
h_t = \overline{A}h_{t-1} + \overline{B}x_t
\]

可以得到：

\[
y_t =
C\overline{B}x_t
+
C\overline{A}\overline{B}x_{t-1}
+
C\overline{A}^2\overline{B}x_{t-2}
+
\cdots
\]

也就是：

\[
y_t = \sum_{i=0}^{t} K_i x_{t-i}
\]

其中：

\[
K_i = C\overline{A}^i\overline{B}
\]

这说明 SSM 可以看成一种特殊卷积。

---

## 3.4 Mamba 的关键：选择性 SSM

传统 SSM 的参数 \(A,B,C\) 对所有输入固定。

Mamba 的重要想法是让部分参数依赖当前输入：

\[
B_t = f_B(x_t)
\]

\[
C_t = f_C(x_t)
\]

\[
\Delta_t = f_\Delta(x_t)
\]

于是递推变成：

\[
h_t =
\overline{A}_t h_{t-1}
+
\overline{B}_t x_t
\]

\[
y_t =
C_t h_t
\]

这样模型可以根据当前 token 动态决定：

- 记住什么
- 忘掉什么
- 输出什么

这就是 Mamba 里 “selective” 的含义。

---

# 4. 推荐学习顺序

我建议按这个路线：

## 阶段一：从零实现 Attention

目标：完全理解 Transformer attention。

任务：

1. 用 NumPy 实现 softmax
2. 用 NumPy 实现 scaled dot-product attention
3. 用 PyTorch 实现一个单头 attention
4. 可视化 attention matrix
5. 训练一个小模型做字符级语言建模

---

## 阶段二：理解简单 RNN 和状态递推

目标：理解隐藏状态如何压缩历史。

任务：

1. 实现简单 RNN：

\[
h_t = \tanh(W_h h_{t-1} + W_x x_t)
\]

2. 实现线性递推：

\[
h_t = ah_{t-1} + bx_t
\]

3. 观察不同 \(a\) 对记忆长度的影响。

---

## 阶段三：实现简化 SSM

目标：理解 SSM 是如何做序列建模的。

任务：

1. 实现：

\[
h_t = Ah_{t-1} + Bx_t
\]

\[
y_t = Ch_t
\]

2. 改变 \(A\) 的特征值，看模型是否稳定。
3. 展开卷积核 \(K_i = CA^iB\)。
4. 比较递推形式和卷积形式是否等价。

---

## 阶段四：实现简化 Mamba

目标：理解 selective SSM。

任务：

1. 让 \(B_t\) 依赖输入：

\[
B_t = W_B x_t
\]

2. 让 \(C_t\) 依赖输入：

\[
C_t = W_C x_t
\]

3. 让遗忘因子依赖输入：

\[
a_t = \sigma(W_a x_t)
\]

4. 实现：

\[
h_t = a_t h_{t-1} + B_t x_t
\]

\[
y_t = C_t h_t
\]

这不是完整 Mamba，但足够帮助你理解核心思想。

---

# 5. macOS 实践环境搭建

如果你是 Apple Silicon，比如 M1/M2/M3/M4 Mac，可以这样配置。

## 5.1 安装 Miniforge

推荐用 Miniforge，而不是 Anaconda，因为对 Apple Silicon 支持更好。

```bash
brew install miniforge
```

创建环境：

```bash
conda create -n llm-math python=3.11 -y
conda activate llm-math
```

---

## 5.2 安装 PyTorch

```bash
pip install torch torchvision torchaudio
```

测试 MPS 是否可用：

```python
import torch

print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())
```

如果输出都是 `True`，说明可以使用 Apple GPU。

选择设备：

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)
```

---

## 5.3 安装常用工具

```bash
pip install numpy matplotlib einops tqdm jupyter ipykernel
```

启动 notebook：

```bash
jupyter notebook
```

---

# 6. 第一个实践：手写 Attention

新建 `attention_demo.py`：

```python
import torch
import torch.nn.functional as F
import math

torch.manual_seed(0)

# sequence length n, embedding dim d
n = 4
d = 8
d_k = 8

X = torch.randn(n, d)

W_Q = torch.randn(d, d_k)
W_K = torch.randn(d, d_k)
W_V = torch.randn(d, d_k)

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

scores = Q @ K.T / math.sqrt(d_k)
weights = F.softmax(scores, dim=-1)
Y = weights @ V

print("X shape:", X.shape)
print("Q shape:", Q.shape)
print("scores shape:", scores.shape)
print("attention weights:")
print(weights)
print("output Y:")
print(Y)
```

运行：

```bash
python attention_demo.py
```

你要重点观察：

```python
scores.shape
```

它是：

```python
[n, n]
```

这就是 \(O(n^2)\) 的来源。

---

# 7. 第二个实践：可视化 Attention Matrix

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

torch.manual_seed(0)

n = 8
d = 16

X = torch.randn(n, d)
W_Q = torch.randn(d, d)
W_K = torch.randn(d, d)
W_V = torch.randn(d, d)

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

scores = Q @ K.T / math.sqrt(d)
weights = F.softmax(scores, dim=-1)

plt.imshow(weights.detach().numpy(), cmap="viridis")
plt.colorbar()
plt.title("Attention Matrix")
plt.xlabel("Key position")
plt.ylabel("Query position")
plt.show()
```

你会看到一个 \(n \times n\) 矩阵。  
第 \(i\) 行表示第 \(i\) 个 token 关注所有 token 的权重。

---

# 8. 第三个实践：线性状态空间递推

新建 `simple_ssm.py`：

```python
import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)

T = 100

x = torch.zeros(T)
x[10] = 1.0  # impulse input

def run_system(a, b=1.0, c=1.0):
    h = torch.tensor(0.0)
    ys = []

    for t in range(T):
        h = a * h + b * x[t]
        y = c * h
        ys.append(y.item())

    return ys

for a in [0.2, 0.5, 0.8, 0.95]:
    y = run_system(a)
    plt.plot(y, label=f"a={a}")

plt.legend()
plt.title("Memory decay for different a")
plt.xlabel("time")
plt.ylabel("output")
plt.show()
```

观察：

- \(a=0.2\)：记忆很快消失
- \(a=0.95\)：记忆持续很久
- 如果 \(|a| > 1\)：系统可能爆炸

这就是 SSM 稳定性的直觉。

---

# 9. 第四个实践：递推形式和卷积形式等价

```python
import torch

T = 10

a = 0.8
b = 1.5
c = 2.0

x = torch.randn(T)

# recurrent form
h = 0.0
ys_rec = []

for t in range(T):
    h = a * h + b * x[t]
    y = c * h
    ys_rec.append(y)

ys_rec = torch.stack(ys_rec)

# convolution form
kernel = torch.tensor([c * (a ** i) * b for i in range(T)])

ys_conv = []

for t in range(T):
    y = 0.0
    for i in range(t + 1):
        y = y + kernel[i] * x[t - i]
    ys_conv.append(y)

ys_conv = torch.stack(ys_conv)

print("recurrent:", ys_rec)
print("convolution:", ys_conv)
print("max diff:", (ys_rec - ys_conv).abs().max())
```

如果 `max diff` 很小，说明二者等价。

---

# 10. 第五个实践：简化版 Selective SSM

这个例子非常接近 Mamba 的核心直觉。

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

class TinySelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()

        self.to_a = nn.Linear(d_model, d_state)
        self.to_b = nn.Linear(d_model, d_state)
        self.to_c = nn.Linear(d_model, d_state)

    def forward(self, x):
        """
        x: [batch, seq_len, d_model]
        """
        batch, seq_len, d_model = x.shape

        h = torch.zeros(batch, self.to_a.out_features, device=x.device)
        ys = []

        for t in range(seq_len):
            x_t = x[:, t, :]

            a_t = torch.sigmoid(self.to_a(x_t))
            b_t = self.to_b(x_t)
            c_t = self.to_c(x_t)

            h = a_t * h + b_t
            y_t = c_t * h

            ys.append(y_t.unsqueeze(1))

        return torch.cat(ys, dim=1)


model = TinySelectiveSSM(d_model=16, d_state=32)

x = torch.randn(2, 10, 16)
y = model(x)

print(y.shape)
```

这个模型做了三件重要的事：

\[
a_t = \sigma(W_a x_t)
\]

\[
b_t = W_b x_t
\]

\[
c_t = W_c x_t
\]

\[
h_t = a_t \odot h_{t-1} + b_t
\]

\[
y_t = c_t \odot h_t
\]

它表达了：

> 每个 token 可以动态决定保留多少过去信息，以及写入多少新信息。

---

# 11. 建议阅读顺序

## Attention 方向

建议顺序：

1. 《The Illustrated Transformer》
2. 《Attention Is All You Need》
3. Jay Alammar 的 Transformer 图解文章
4. Harvard NLP Annotated Transformer
5. nanoGPT 源码

重点不是一开始就读论文，而是：

> 先手写 attention，再读论文。

---

## SSM / Mamba 方向

建议顺序：

1. 先学一维线性递推：

\[
h_t = ah_{t-1} + bx_t
\]

2. 再学状态空间模型：

\[
h_t = Ah_{t-1} + Bx_t
\]

3. 再看 S4 的思想。
4. 再看 Mamba 论文。
5. 最后看 Mamba 官方代码。

对于 Mamba，不建议一上来直接啃完整论文。它涉及：

- continuous-time SSM
- discretization
- HiPPO
- S4
- parallel scan
- hardware-aware algorithm
- selective scan

更好的方式是：

> 先实现简化 selective SSM，再逐步接近论文。

---

# 12. 一个推荐的 4 周学习计划

## 第 1 周：Attention

目标：

- 理解 \(Q,K,V\)
- 理解 attention matrix
- 理解为什么是 \(O(n^2)\)

任务：

- NumPy 实现 softmax
- PyTorch 实现 scaled dot-product attention
- 可视化 attention matrix
- 读 Illustrated Transformer

---

## 第 2 周：Transformer Block

目标：

- 理解 multi-head attention
- 理解 residual connection
- 理解 layer normalization
- 理解 feed-forward network

任务：

- 用 PyTorch 写一个 mini Transformer block
- 跑一个字符级语言模型
- 看 nanoGPT 中的 attention 实现

---

## 第 3 周：SSM

目标：

- 理解递推系统
- 理解稳定性
- 理解 SSM 和卷积的关系

任务：

- 实现一维 SSM
- 实现矩阵形式 SSM
- 展开卷积核
- 可视化不同 \(A\) 的记忆曲线

---

## 第 4 周：Mamba

目标：

- 理解 selective SSM
- 理解输入依赖的 \(B_t,C_t,\Delta_t\)
- 理解为什么适合长序列

任务：

- 实现 TinySelectiveSSM
- 用它做简单序列分类
- 读 Mamba 论文摘要、方法部分
- 对比 Transformer 和 Mamba 的速度、显存

---

# 13. 最小项目建议

如果你想做一个真正有成就感的小项目，可以做：

## 项目 A：Attention vs SSM 的记忆实验

构造一个任务：

输入序列：

```text
A x x x x x x x B
```

要求模型判断前面是否出现过 `A`。

比较：

1. 小 Transformer
2. 小 RNN
3. 小 SSM
4. TinySelectiveSSM

观察不同序列长度下的准确率。

---

## 项目 B：字符级语言模型

数据集可以用很小的文本，比如：

- 唐诗
- 英文童话
- 你自己的笔记
- Shakespeare tiny dataset

分别实现：

1. mini GPT
2. simple SSM LM
3. TinySelectiveSSM LM

比较：

- loss
- 生成效果
- 训练速度
- 显存占用

---

# 14. 你可以这样开始

如果你想要最稳的路线，我建议今天先做三件事：

```bash
brew install miniforge
conda create -n llm-math python=3.11 -y
conda activate llm-math
pip install torch numpy matplotlib einops tqdm jupyter
```

然后按顺序运行：

1. `attention_demo.py`
2. attention matrix 可视化
3. `simple_ssm.py`
4. 递推和卷积等价实验
5. `TinySelectiveSSM`

---

## 一句话总结

学习路线可以是：

> 先用矩阵乘法理解 full attention，再用差分方程理解 SSM，最后用输入依赖的动态递推理解 Mamba。

如果你的目标是在 macOS 上边学边做，我建议不要一开始直接读 Mamba 源码，而是先手写：

\[
\operatorname{Attention}(Q,K,V)
\]

和

\[
h_t = a_t h_{t-1} + b_t x_t
\]

这两个最小模型。  
理解了这两个公式，Transformer 和 Mamba 的核心差异就会非常清楚。