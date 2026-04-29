# torch.compile 与静态 KV Cache

> 本文解释为什么 LLM decode 阶段的 KV Cache 实现会影响 `torch.compile` 的图编译效果，以及为什么预分配静态 KV Cache 更适合编译优化。

---

## 1. eager 模式为什么有额外开销

PyTorch 默认是 eager execution：Python 每执行到一行张量代码，就立刻调用一次对应算子。以 decode 阶段为例，每生成一个 token 都要经过 28 层 Transformer，每层包含：

- RMSNorm
- Q/K/V Linear
- RoPE
- KV Cache 更新
- GQA repeat
- SDPA Attention
- O projection
- MLP
- residual add

这些算子本身可以在 MPS/CUDA 上执行，但算子之间的调度仍由 Python 逐步发起。当 decode 每步只处理 1 个 token 时，单个算子的计算量并不大，Python 调度、动态图判断、张量 shape 检查、kernel launch/dispatch 的相对占比就会上升。

`torch.compile` 的目标就是把这一段 Python 级别的动态图执行路径捕获成一个可复用的计算图，减少每个 token 重复付出的调度成本。

---

## 2. torch.compile 做了什么

调用：

```python
compiled_runner = torch.compile(
    runner,
    mode="reduce-overhead",
    dynamic=False,
)
```

大致会经历以下阶段：

### 2.1 TorchDynamo 捕获 Python 执行路径

第一次运行 `runner(...)` 时，Dynamo 观察 Python bytecode 和实际张量输入，将可追踪的 PyTorch tensor 运算捕获成 FX graph。

同时，Dynamo 会生成 guard，例如输入 dtype、device、rank、部分 shape、模块属性是否变化。后续调用如果满足 guard，就能复用已经捕获和编译的图。

### 2.2 形成稳定的 FX Graph

在静态 KV Cache benchmark 中，关键输入包括：

- `token_ids`
- `key_cache`
- `value_cache`
- `cache_position`

如果这些张量形状稳定，后续 decode step 就能复用同一张图。如果形状变化，guard 可能失效，PyTorch 可能重新捕获和重新编译。

### 2.3 TorchInductor 做图级优化和 lowering

Inductor 会对图中的算子做优化，例如删除冗余操作、融合可融合的 pointwise 操作、规划中间张量。

在 MPS 上，并不是所有算子都会变成一个巨大的自定义 Metal kernel；很多重型算子仍会调用底层 MPS/Metal 优化 kernel。但即使不能完全融合，图编译仍然能减少 Python 逐算子调度成本，并让一部分操作按更稳定的图执行。

### 2.4 缓存编译结果

warmup 第一次运行最慢，因为它承担了图捕获、guard 生成、lowering、kernel/graph 准备等成本。后续输入只要满足 guard，就直接复用编译结果。

因此，`torch.compile` 的收益不是让模型参数更少，也不是降低 attention 理论复杂度，而是减少 decode 小步执行中的 Python/调度开销，并让可融合的局部图更高效。

---

## 3. 为什么动态 KV Cache 不适合图编译

常见的 KV Cache 写法会在每步 decode 时追加新的 K/V：

```python
k = torch.cat([past_k, k], dim=2)
v = torch.cat([past_v, v], dim=2)
```

这会导致每一步的 K/V shape 都不同：

```text
step 1: [batch, kv_heads, 1, head_dim]
step 2: [batch, kv_heads, 2, head_dim]
step 3: [batch, kv_heads, 3, head_dim]
...
```

对 eager 来说这很自然，因为 Python 每步动态执行即可。对 `torch.compile` 来说，这会带来两个问题：

- guard 容易失效：图通常会对 shape 生成约束，shape 变了就可能触发新图。
- 编译成本无法摊销：如果不同 token 长度触发不同图，decode 阶段可能边生成边编译，实际速度会非常差。

这就是直接 compile 动态 KV 路径时 warmup 很慢、行为不稳定的根本原因。

---

## 4. 静态预分配 KV Cache 如何让图稳定

静态 KV Cache 使用固定容量：

```python
shape = (NUM_LAYERS, BATCH_SIZE, NUM_KV_HEADS, max_cache_len, HEAD_DIM)
key_cache = torch.empty(shape, dtype=dtype, device=device)
value_cache = torch.empty(shape, dtype=dtype, device=device)
```

每一步不再 `cat`，而是写入固定位置：

```python
key_cache[layer_idx].index_copy_(2, cache_position, k)
value_cache[layer_idx].index_copy_(2, cache_position, v)
```

然后 attention 始终面对固定长度的 K/V：

```python
k_rep = key_cache[layer_idx].repeat_interleave(groups, dim=1)
v_rep = value_cache[layer_idx].repeat_interleave(groups, dim=1)
allowed_positions = (self.cache_positions <= cache_position[0]).view(1, 1, 1, self.max_cache_len)

attn_out = F.scaled_dot_product_attention(
    q,
    k_rep,
    v_rep,
    attn_mask=allowed_positions,
    is_causal=False,
)
```

核心变化是：

- `key_cache/value_cache` 的 shape 固定。
- `q/k/v` 当前 token 的 shape 固定。
- attention mask 的 shape 固定。
- 只有 `cache_position` 的数值变化，表示当前允许看到哪些位置。

这样 compile 看到的是同一个结构：固定大小 cache、单 token decode、固定形状 mask。不同 decode 长度只是外层 Python 循环跑多少次，而每次调用的 compiled graph 是同一类输入。

---

## 5. 为什么静态 KV eager 会比动态 KV eager 慢

静态 KV Cache 是为编译友好设计的，它牺牲了一点 eager 性能：

- 动态 KV 每步只 attention 到真实历史长度，例如第 5 步看 5 个位置。
- 静态 KV 每步都传入固定 `max_cache_len`，例如 64 个位置，再用 mask 屏蔽未来位置。

所以在 eager 模式下，静态 KV 做了更多固定长度的 attention 工作，TBT 会比原始动态 KV 高一些。这不是错误，而是为了换取图稳定性。

真正对比 compile 时，应比较：

- 静态 KV + eager
- 静态 KV + compile

这样变量才只剩下是否图编译。

---

## 6. 为什么 warmup 很慢但后续 decode 变快

在 MPS float16 实测中：

- compile wrapper 创建耗时：约 0.55s
- 第一次编译 warmup：约 26.50s
- 编译后 decode TBT：约 12-13ms

warmup 慢，是因为第一次运行承担了图捕获和编译准备成本。后续 decode 变快，是因为：

- 不再逐层逐算子通过 Python 动态调度完整路径。
- 固定 shape 让 guard 稳定，不需要每步重新编译。
- 一些小算子和张量操作可以被图级优化处理。
- MPS 上的底层 kernel 调用路径更稳定，调度开销更低。

因此 `torch.compile` 适合：

- 长 decode
- 多轮生成
- 常驻服务
- 同一模型和 cache 规格反复使用

不适合：

- 只生成很少 token 就退出
- prompt/cache shape 经常大幅变化
- 每次请求都重新启动进程并重新编译

---

## 7. 为什么测试要使用相同 prompt

测试 compile/eager 时，不应换不同 prompt 来“避免 KV Cache 影响”。KV Cache 本来就是 decode 的核心状态，应该被纳入测试。

公平做法是：

- 使用相同 prompt，保证输入 token 一致。
- 每个 run 重新初始化 KV Cache，避免前一次生成污染下一次。
- 保持相同 `max_cache_len`，保证 compile/eager 的 attention 工作量一致。

换不同 prompt 会改变 token 数、语义路径、生成 token 和 attention 内容，反而会引入额外变量。
