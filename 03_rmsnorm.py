import os
import torch
import torch.nn as nn
from safetensors.torch import safe_open

class QwenRMSNorm(nn.Module):
    """
    均方根归一化 (Root Mean Square Normalization)
    作用：把特征值拉平，防止随着网络深度的增加发生数值爆炸或消失。
    这比传统的 LayerNorm 更快，因为它省去了计算均值 (Mean) 的步骤，直接计算均方根。
    """
    def __init__(self, hidden_size=1024, eps=1e-6):
        super().__init__()
        # 可学习的权重 gamma，它的长度正好等于 hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # 1. 精度转换：大模型推理时，累加平方操作极易溢出，必须强制转换为 float32 计算
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        
        # 2. 计算均方 (Mean Square)
        # pow(2) 求平方，mean(-1) 在特征维度上求平均，keepdim=True 保持形状以便广播
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        
        # 3. 归一化公式：x * (1 / sqrt(variance + epsilon))
        # rsqrt 也就是 1/sqrt，在底层有专门的硬件加速指令，比 / sqrt() 快
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        
        # 4. 转回原本的低精度 (如 bfloat16)，并乘以可学习的特定权重 gamma
        return self.weight * hidden_states.to(input_dtype)


def main():
    print("=== 第三步：实现与测试 RMSNorm ===")
    model_dir = "/Users/wangrui/localinfer/Qwen3.5-0.8B"
    safetensor_path = os.path.join(model_dir, "model.safetensors-00001-of-00001.safetensors")
    
    # 实例化我们手写的 RMSNorm (配置来自 config.json)
    rmsnorm = QwenRMSNorm(hidden_size=1024, eps=1e-6)
    
    print("正在加载真实的 model.language_model.norm.weight ...")
    with safe_open(safetensor_path, framework="pt", device="cpu") as f:
        # 我们这里先用大门出口的那一层 norm.weight 进行测试
        real_weight = f.get_tensor("model.language_model.norm.weight")
        
    # 把从 safetensors 里读出来的真实权重，注入给我们的模型
    rmsnorm.weight.data = real_weight
    
    # 保持张量精度为大模型标配的 bfloat16
    rmsnorm = rmsnorm.to(torch.bfloat16)
    
    print(f"✅ RMSNorm 模块加载完成！内部权重形状: {list(rmsnorm.weight.shape)}\n")
    
    # 模拟我们在 01 脚本中跑出来的词嵌入张量 (3 个 token)
    # Shape: [Batch=1, SeqLen=3, Hidden=1024]
    # 我们故意搞一些很大的随机数来测试归一化效果
    dummy_input = torch.randn(1, 3, 1024, dtype=torch.bfloat16) * 100.0
    
    print(f"输入张量形状: {list(dummy_input.shape)}")
    print(f"归一化前，第一个 Token 的前 3 个特征值: {dummy_input[0, 0, :3].tolist()}")
    
    # 前向传播！
    output = rmsnorm(dummy_input)
    
    print(f"\n输出张量形状: {list(output.shape)}")
    print(f"归一化后，第一个 Token 的前 3 个特征值: {output[0, 0, :3].tolist()}")
    print("\n🎉 测试成功！原本上百的数值，被完美平滑到了合理的区间，并融入了 Qwen 训练出来的权重特征。")

if __name__ == "__main__":
    main()
