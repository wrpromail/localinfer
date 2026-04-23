# Qwen3.5-0.8B config.json 字段解析与分析

`config.json` 是模型架构的蓝图。经过分析，我们可以发现 Qwen3.5-0.8B **并不是一个传统的纯 Transformer 模型**，而是一个融合了状态空间模型（SSM/Mamba类似机制）和传统 Attention 的**混合架构**，并且内置了多 Token 预测（MTP）和多模态 RoPE（mRoPE）的支持。

以下是 `config.json` 中各个核心字段的具体含义：

## 1. 顶层全局配置
- `architectures`: `["Qwen3_5ForConditionalGeneration"]`
  - HF `transformers` 库用来实例化模型的类名。
- `model_type`: `"qwen3_5"`
  - 模型的架构标识符。
- `tie_word_embeddings`: `true`
  - 表示输入词表（Embedding）的权重矩阵和输出预测（LM Head）的权重矩阵是共享的，这能大幅减少参数量。
- `image_token_id`, `video_token_id`, `vision_end_token_id`, `vision_start_token_id`
  - 用于处理多模态（图像、视频）的特殊控制 Token ID。

## 2. 文本模型配置 (`text_config`)
这是我们手写推理代码最需要关注的部分。

### 基础结构与维度
- `hidden_size`: `1024`
  - 模型的隐层维度（也就是每一个 Token 的特征维度大小）。
- `intermediate_size`: `3584`
  - MLP（多层感知机 / FFN）中的中间扩展维度。
- `num_hidden_layers`: `24`
  - 模型的总层数（Layer 数量）。
- `vocab_size`: `248320`
  - 词表大小，Qwen 的词表非常大，包含了多语言和大量的特殊 Token。
- `rms_norm_eps`: `1e-06`
  - RMSNorm 层计算方差时为了防止除零错误加的一个极小值。
- `dtype`: `"bfloat16"`
  - 模型的默认权重类型，建议加载时以此类型（`torch.bfloat16`）加载以防精度溢出。
- `hidden_act`: `"silu"`
  - 激活函数，在大模型中通常配合 Gated MLP 组成 SwiGLU。

### 混合架构层定义 (Hybrid Architecture)
这个模型使用了混合层。这解释了之前我们查到的 "Gated Delta Networks and Gated Attention"。
- `layer_types`: 长度为 24 的列表，每层定义如下：
  - `["linear_attention", "linear_attention", "linear_attention", "full_attention", ...]`
  - 每 3 层“线性注意力”（Linear Attention），接 1 层“全注意力”（Full Attention）。
- `full_attention_interval`: `4`
  - 验证了上面的分布，每隔 4 层有一个 Full Attention。
- `attn_output_gate`: `true`
  - 表示注意力层的输出还会经过一个门控（Gate）网络。

### 全注意力层参数 (Full Attention)
在 `layer_types` 等于 `full_attention` 的层：
- `num_attention_heads`: `8` (Query 头的数量)
- `num_key_value_heads`: `2` (Key/Value 头的数量)
  - Q:KV = 8:2 = 4。这说明它使用了 **GQA (Grouped-Query Attention)** 机制。
- `head_dim`: `256`
  - 每一个注意力头的维度。注意：`8 * 256 = 2048`，而 `hidden_size` 是 1024。通常 `num_heads * head_dim = hidden_size`，这里的特殊比例可能是混合架构特意设计的扩维机制。

### 线性注意力层参数 (Linear Attention)
这些层主要为了长文本的线性复杂度推理而设计。
- `linear_num_key_heads`: `16`
- `linear_num_value_heads`: `16`
- `linear_key_head_dim`: `128`
- `linear_value_head_dim`: `128`
- `linear_conv_kernel_dim`: `4`
  - 存在一维卷积，这非常类似于 Mamba 架构的底层设计。
- `mamba_ssm_dtype`: `"float32"`
  - 明确提到了 SSM（State Space Model），说明这里的“线性注意力”底层实质上是基于状态空间模型（类似于 Mamba）。计算时这部分的隐状态需要保持 `float32` 精度。

### 旋转位置编码参数 (`rope_parameters`)
Qwen3.5 为了支持视觉，采用了更复杂的多模态 RoPE。
- `rope_theta`: `10000000` (RoPE 的基础频率基数，很大，为了支持 256K 极长上下文)
- `partial_rotary_factor`: `0.25`
  - **重要坑点**：不是对 `head_dim` 所有的维度都做旋转，只旋转 25%！以 `head_dim=256` 为例，只有前 64 个维度参与 RoPE 计算。
- `mrope_section`: `[11, 11, 10]`
  - 因为只旋转 64 个维度（复数角度对应 32 对），这里将其拆分成了 11, 11, 10 三部分，用来分别编码 3D 视觉（时间、高、宽）的位置信息。对于纯文本，通常需要将这三个维度展平为 1D 序列的位置。

### 其他特殊机制
- `mtp_num_hidden_layers`: `1`
  - **MTP (Multi-Token Prediction)**：代表模型支持多 Token 预测（类似深度求索 DeepSeek 的机制），训练时不仅预测下一个词，还预测下几个词。有助于做投机解码（Speculative Decoding）。

## 3. 视觉模型配置 (`vision_config`)
- 因为它是多模态模型，这里定义了一个基于 ViT (Vision Transformer) 的图像编码器。
- `depth`: `12` (层数)
- `hidden_size`: `768`
- `patch_size`: `16`, `temporal_patch_size`: `2`, `spatial_merge_size`: `2` (负责视频帧和图像块的切分与聚合)。

---

## 💡 对我们手写代码的启发与挑战：
1. **不是简单的 Transformer**：我们不能抄标准的 LLaMA/Qwen2 架构。我们需要写**两套 Block**：
   - 一套是 `FullAttentionBlock`（标准的 GQA Attention）。
   - 一套是 `LinearAttentionBlock`（带有 1D 卷积和状态空间的 Mamba 变体）。
2. **RoPE 只应用部分维度**：这点写代码时极容易写错（`partial_rotary_factor=0.25`）。
3. **KV Cache 有两种**：全注意力层需要存过去的 Key 和 Value；而线性层（SSM）的 KV Cache 逻辑是不同的，它们只需要存 SSM 的隐状态！
