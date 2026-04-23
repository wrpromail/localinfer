"""
Qwen3.5-0.8B 整体自回归推理前向传播 (伪代码)
这份代码是为了展示数据在整个模型中流转的全景结构。
"""
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def forward(self, x):
        # 具体的数学计算后续实现，作用是平滑数据
        return normalized_x

class SwiGLU_MLP(nn.Module):
    def forward(self, x):
        # 1. 兵分两路放大到 intermediate_size
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        # 2. 激活与门控
        activated_gate = torch.nn.functional.silu(gate)
        # 3. 融合并压缩回 hidden_size
        return self.down_proj(activated_gate * up)

class FullAttention(nn.Module):
    def forward(self, x, kv_cache):
        # 传统的 QKV 乘法注意力机制，用于精准长文本特征提取
        # 包含：生成 QKV、应用 RoPE、更新 Cache、计算 Attention Score
        return attention_output

class LinearAttention(nn.Module):
    def forward(self, x, state_cache):
        # Mamba/SSM 变体线性注意力，极其省内存和计算量
        # 包含：1D卷积处理、更新状态机 (State Space)
        return linear_output


class Qwen3_5_Block(nn.Module):
    """这代表了 24 层中的某一层"""
    def __init__(self, layer_idx, layer_type):
        super().__init__()
        self.input_layernorm = RMSNorm()
        self.post_attention_layernorm = RMSNorm()
        self.mlp = SwiGLU_MLP()
        
        # 就像您理解的，每一层最大的区别，仅仅在于到底挂载了哪种注意力模块！
        if layer_type == "full_attention":
            self.attention = FullAttention()
        else:
            self.attention = LinearAttention()

    def forward(self, x, cache):
        # ==========================================
        # 1. Attention 模块处理
        # ==========================================
        residual = x
        x_normed = self.input_layernorm(x)
        
        # 根据本层的类型，动态调用对应的注意力机制
        attn_output = self.attention(x_normed, cache)
        
        x = residual + attn_output  # 第一次残差

        # ==========================================
        # 2. MLP 模块处理 (每一层都长得一模一样)
        # ==========================================
        residual = x
        x_normed = self.post_attention_layernorm(x)
        mlp_output = self.mlp(x_normed)
        x = residual + mlp_output  # 第二次残差

        return x


class Qwen3_5_Model(nn.Module):
    """这是整个大模型的外壳"""
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(248320, 1024)
        
        # 按照 config.json，搭建 24 个混合的 Layer
        # 每 3 个 linear_attention 接 1 个 full_attention
        self.layers = nn.ModuleList([
            Qwen3_5_Block(layer_idx=i, layer_type="linear_attention" if (i+1)%4 != 0 else "full_attention")
            for i in range(24)
        ])
        
        # 最终大门出口的 RMSNorm
        self.norm = RMSNorm()

    def forward(self, input_ids, caches):
        # 1. 大门入口：查表转为 1024 维张量 (我们刚才用脚本验证过这一步)
        x = self.embed_tokens(input_ids)

        # 2. 依次穿过 24 层
        for i, layer in enumerate(self.layers):
            # 将主干数据 x 传进每一层，并更新缓存
            x = layer(x, caches[i])

        # 3. 大门出口：最后一次归一化
        x = self.norm(x)

        # 4. 计算词表概率：因为 tie_word_embeddings = True
        # 所以输出时没有单独的 lm_head，而是直接再次用 embed_tokens.weight 矩阵乘回去！
        logits = torch.matmul(x, self.embed_tokens.weight.T)
        
        return logits

# =====================================================================
# 下面是自回归循环 (Autoregressive Loop) 的调用过程
# =====================================================================
def generate_tokens(model, prompt_ids, generate_steps=10):
    current_ids = prompt_ids
    caches = [None] * 24  # 存放每一层的 KV 缓存或状态机缓存
    
    generated_tokens = []
    
    for step in range(generate_steps):
        # 将当前的 IDs 和缓存塞入模型，拿回下一个词的所有概率
        logits = model(current_ids, caches)
        
        # 只取出最后一个词的概率分布，找到概率最大的那个词的 ID
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
        generated_tokens.append(next_token_id)
        
        # 【核心技巧】：进入下一轮前，我们不需要再把前面所有的词塞进去了！
        # 只要把新生成的这个词传给模型即可，因为前面的记忆都在 caches 里。
        current_ids = next_token_id.unsqueeze(0)
        
    return generated_tokens
