import os
import torch
import torch.nn as nn
from safetensors.torch import safe_open

from importlib.machinery import SourceFileLoader
rmsnorm_module = SourceFileLoader("rmsnorm", "03_rmsnorm.py").load_module()
QwenRMSNorm = rmsnorm_module.QwenRMSNorm

class QwenLinearAttentionBlock(nn.Module):
    """
    Qwen3.5 的 Gated Linear Attention (GLA) / Mamba 变体
    这里展示的是它在自回归生成（Decode 阶段，每次 1 个词）时的真实物理运行机理！
    """
    def __init__(self, hidden_size=1024, num_heads=16, head_dim=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim  # 2048
        
        # 1. 主投影矩阵
        self.in_proj_qkv = nn.Linear(hidden_size, self.inner_dim * 3, bias=False)
        self.in_proj_z = nn.Linear(hidden_size, self.inner_dim, bias=False)
        
        # 2. 局部一维卷积 (深度可分离卷积)
        # 用来提取当前词和过去 3 个词的局部特征 (kernel_size=4)
        self.conv1d = nn.Conv1d(
            in_channels=self.inner_dim * 3, 
            out_channels=self.inner_dim * 3, 
            kernel_size=4, 
            groups=self.inner_dim * 3, # 深度可分离
            padding=3 # 因果填充
        )
        
        # 3. 动态衰减因子/状态更新门控 (Data-dependent decay)
        self.in_proj_a = nn.Linear(hidden_size, num_heads, bias=False)
        self.in_proj_b = nn.Linear(hidden_size, num_heads, bias=False)
        self.A_log = nn.Parameter(torch.empty(num_heads))
        self.dt_bias = nn.Parameter(torch.empty(num_heads))
        
        # 4. 头归一化 (针对 128 维度，所有头共享一套权重！)
        self.norm = QwenRMSNorm(self.head_dim)
        
        # 5. 输出投影
        self.out_proj = nn.Linear(self.inner_dim, hidden_size, bias=False)

    def forward_decode_step(self, x, rnn_state, conv_state):
        """
        单步 Decode 逻辑（极其重要：这里没有任何 Attention 的 NxN 矩阵乘法！）
        x: [1, 1024]
        rnn_state: [1, 16, 128, 128] (固定大小的记忆矩阵)
        conv_state: [1, 6144, 3] (存储过去 3 步的输入特征)
        """
        # 1. QKV 投影
        qkv = self.in_proj_qkv(x) # [1, 6144]
        
        # 2. 1D 卷积更新局部特征 (伪代码逻辑，我们用简单的相加代替真实的卷积核相乘)
        # 实际是在做：当前 qkv 和 conv_state 里的历史进行加权
        # 更新 conv_state (丢掉最老的，塞入最新的)
        qkv_conv = qkv # 假设这是被 conv1d 处理过后的 qkv
        
        # 3. 切分 Q, K, V
        # 形状全都是 [1, 16, 128]
        q, k, v = qkv_conv.split(self.inner_dim, dim=-1)
        q = q.view(1, self.num_heads, self.head_dim)
        k = k.view(1, self.num_heads, self.head_dim)
        v = v.view(1, self.num_heads, self.head_dim)
        
        # 4. 计算动态衰减率 (Decay)
        # 用输入特征 x 来决定我们要忘记多少历史包袱！
        decay_a = self.in_proj_a(x) # [1, 16]
        # (实际数学上这里会和 A_log, dt_bias 经过指数或者 softplus 激活)
        decay_rate = torch.sigmoid(decay_a).unsqueeze(-1).unsqueeze(-1) # 简化表示
        
        # 5. 🌟 最核心的 RNN 状态机更新 🌟
        # h_t = h_{t-1} * decay + K^T * V
        # k.unsqueeze(-1) @ v.unsqueeze(-2) 是计算当前词的知识矩阵 [16, 128, 128]
        current_knowledge = torch.matmul(k.unsqueeze(-1), v.unsqueeze(-2))
        
        # 记忆矩阵更新！老记忆衰减，新知识加入！
        rnn_state = rnn_state * decay_rate + current_knowledge
        
        # 6. 生成当前的注意力输出 (Q 和 记忆矩阵 相乘)
        # [1, 16, 1, 128] @ [1, 16, 128, 128] -> [1, 16, 1, 128]
        y = torch.matmul(q.unsqueeze(-2), rnn_state).squeeze(-2)
        # 7. 头级别归一化与门控输出 (Gated)
        # 注意 y 目前是 [1, 16, 128]，norm 作用于最后的 128 维
        y = self.norm(y)
        y = y.view(1, self.inner_dim) # 合并回 [1, 2048]
        z = self.in_proj_z(x) # [1, 2048]
        y = y * torch.nn.functional.silu(z)
        
        # 8. 降维回归主干道
        out = self.out_proj(y) # [1, 1024]
        
        return out, rnn_state, conv_state

def main():
    print("=== 第六步：实现 Mamba (Gated Linear Attention) 的单步推理模型 ===")
    model_dir = "/Users/wangrui/localinfer/Qwen3.5-0.8B"
    safetensor_path = os.path.join(model_dir, "model.safetensors-00001-of-00001.safetensors")
    
    # 初始化我们的块
    block = QwenLinearAttentionBlock(hidden_size=1024, num_heads=16, head_dim=128)
    
    layer_idx = 0
    print(f"正在读取模型 Layer {layer_idx} (线性层) 的大量奇特权重...")
    
    with safe_open(safetensor_path, framework="pt", device="cpu") as f:
        block.in_proj_qkv.weight.data = f.get_tensor(f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_qkv.weight")
        block.in_proj_z.weight.data = f.get_tensor(f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_z.weight")
        block.in_proj_a.weight.data = f.get_tensor(f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_a.weight")
        block.in_proj_b.weight.data = f.get_tensor(f"model.language_model.layers.{layer_idx}.linear_attn.in_proj_b.weight")
        
        # conv1d 的权重需要 squeeze 适配一下 PyTorch Conv1d 初始化
        conv_w = f.get_tensor(f"model.language_model.layers.{layer_idx}.linear_attn.conv1d.weight")
        block.conv1d.weight.data = conv_w
        
        block.norm.weight.data = f.get_tensor(f"model.language_model.layers.{layer_idx}.linear_attn.norm.weight")
        block.out_proj.weight.data = f.get_tensor(f"model.language_model.layers.{layer_idx}.linear_attn.out_proj.weight")
        
    block = block.to(torch.bfloat16)
    print("✅ 权重加载完毕！")
    
    # === 模拟生成阶段 (Decode) ===
    # 输入仅仅是当前生成的 1 个词
    dummy_x = torch.randn(1, 1024, dtype=torch.bfloat16)
    
    # 【重点】：我们不需要维护庞大的 KV Cache！只需要固定大小的状态机矩阵！
    # rnn_state 固定大小为: [Batch, Heads, HeadDim, HeadDim] = [1, 16, 128, 128]
    rnn_state = torch.zeros(1, 16, 128, 128, dtype=torch.bfloat16)
    conv_state = None # 模拟卷积缓存
    
    print(f"\n[推演开始] 前方正在遭遇极强的数据风暴...")
    print(f"主干道输入: {list(dummy_x.shape)}")
    print(f"我们的状态机记忆矩阵大小: {list(rnn_state.shape)} (它是恒定不变的！)")
    
    out, new_rnn_state, new_conv_state = block.forward_decode_step(dummy_x, rnn_state, conv_state)
    
    print(f"\n[推演完成]")
    print(f"主干道输出: {list(out.shape)}")
    print(f"更新后的记忆矩阵大小: {list(new_rnn_state.shape)}")
    print("🎉 Mamba 核心机制通关！它的算力不仅小，而且不管上下文有 10万 还是 100万，记忆矩阵永远只有这么大！这就叫『无痛长文本』！")

if __name__ == "__main__":
    main()
