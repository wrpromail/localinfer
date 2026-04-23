import os
import torch
import torch.nn as nn
from safetensors.torch import safe_open

class QwenSwiGLU_MLP(nn.Module):
    """
    Qwen3.5 使用的 SwiGLU 多层感知机 (MLP)
    这是模型真正用来“死记硬背”海量知识的地方。
    与传统 ReLU MLP 的不同在于，它使用了两路兵马 (up 和 gate)，并且引入了硅基生命最爱的 SiLU 激活函数。
    """
    def __init__(self, hidden_size=1024, intermediate_size=3584):
        super().__init__()
        # 门控投影：用来决定我们要“激活”多少知识
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # 上升投影：用来将特征放大到极其宽广的高维空间寻找组合
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # 下降投影：将激活后的庞大知识浓缩回主干道的大小
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        # 1. 兵分两路：同时进行 Gate 和 Up 投影
        # 将 1024 维瞬间膨胀到 3584 维
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # 2. 对 Gate 应用 SiLU 激活函数 (本质上是 x * sigmoid(x))
        # 这一步赋予了神经网络非线性拟合能力
        activated_gate = torch.nn.functional.silu(gate)
        
        # 3. GLU (Gated Linear Unit) 核心操作：逐元素相乘
        # 用门控去挑选 Up 投影出来的海量高维特征
        fused_hidden = activated_gate * up
        
        # 4. 压缩回原始维度
        # 将 3584 维压缩回 1024 维，重新回到主干道
        return self.down_proj(fused_hidden)

def main():
    print("=== 第七步：实现并测试 SwiGLU MLP 模块 ===")
    model_dir = "/Users/wangrui/localinfer/Qwen3.5-0.8B"
    safetensor_path = os.path.join(model_dir, "model.safetensors-00001-of-00001.safetensors")
    
    # 实例化我们的 MLP 模块
    mlp = QwenSwiGLU_MLP(hidden_size=1024, intermediate_size=3584)
    
    layer_idx = 0
    print(f"正在抽取 Layer {layer_idx} 中海量的参数记忆 (MLP weights)...")
    
    with safe_open(safetensor_path, framework="pt", device="cpu") as f:
        mlp.gate_proj.weight.data = f.get_tensor(f"model.language_model.layers.{layer_idx}.mlp.gate_proj.weight")
        mlp.up_proj.weight.data = f.get_tensor(f"model.language_model.layers.{layer_idx}.mlp.up_proj.weight")
        mlp.down_proj.weight.data = f.get_tensor(f"model.language_model.layers.{layer_idx}.mlp.down_proj.weight")
        
    mlp = mlp.to(torch.bfloat16)
    print("✅ 权重替换完毕！")
    print(f"门控矩阵大小: {list(mlp.gate_proj.weight.shape)}")
    print(f"压缩矩阵大小: {list(mlp.down_proj.weight.shape)}\n")
    
    # 测试前向传播
    # 模拟 [Batch=1, SeqLen=3, Hidden=1024]
    dummy_input = torch.randn(1, 3, 1024, dtype=torch.bfloat16)
    print(f"主干道输入张量: {list(dummy_input.shape)}")
    
    output = mlp(dummy_input)
    print(f"MLP 激活提取结果: {list(output.shape)}")
    print("\n🎉 测试成功！数据完美地经历了 1024 -> 3584 -> 门控相乘 -> 1024 的膨胀与压缩之旅！")

if __name__ == "__main__":
    main()
