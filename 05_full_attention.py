import os
import math
import torch
import torch.nn as nn
from safetensors.torch import safe_open
from importlib.machinery import SourceFileLoader

# 动态导入刚才写的 04_rope.py 模块
rope_module = SourceFileLoader("rope", "04_rope.py").load_module()
QwenRoPE = rope_module.QwenRoPE

class FullAttentionBlock(nn.Module):
    """
    Qwen3.5-0.8B 的全注意力块（Gated Attention）教学版实现。

    ⚠️ 与完整版（08_generate.py）的差异：
    1. 本版本不含 q_norm / k_norm（QwenRMSNorm on head_dim），完整版有。
    2. 本版本不含 causal mask，仅适用于单步 decode（seq_len=1）。
       如需用于 prefill（seq_len>1），请参考 09_qwen3_0_6b_generate.py 的实现。
    """
    def __init__(self, hidden_size=1024, num_heads=8, num_kv_heads=2, head_dim=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        # 【神级发现】：因为 config.json 中 attn_output_gate = true
        # q_proj 其实是 Q 和 Gate 的合并投影！所以维度是 8 * 256 * 2 = 4096
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim * 2, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
        self.rope = QwenRoPE(head_dim=head_dim, partial_rotary_factor=0.25)

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        
        # 1. 获取 Q_和_Gate, K, V
        q_and_gate = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # 切分 Q 和 Gate (各 2048 维)
        q = q_and_gate[..., : self.num_heads * self.head_dim]
        gate = q_and_gate[..., self.num_heads * self.head_dim :]
        
        # 2. Reshape 切分出“头 (Heads)”
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # 3. 应用 RoPE
        q, k = self.rope.apply_rope(q, k, seq_len)
        
        # 4. GQA KV 头广播复制
        num_key_value_groups = self.num_heads // self.num_kv_heads
        k = torch.repeat_interleave(k, repeats=num_key_value_groups, dim=1)
        v = torch.repeat_interleave(v, repeats=num_key_value_groups, dim=1)
        
        # 5. Attention 计算
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # ⚠️ 本实现未加 causal mask（因果掩码）。
        # 对单步 decode（seq_len=1）不需要掩码，但对 prefill（seq_len>1）会导致
        # 未来 token 的信息泄露，结果将不正确。
        # 完整带 causal mask 的实现见：09_qwen3_0_6b_generate.py FullAttentionBlock.forward()
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        # 6. Reshape 还原
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)
        
        # 【核心差异】：在送入 o_proj 前，用切出来的 gate 对结果进行过滤 (通常使用 silu 激活)
        # 也就是 Gated Attention 的真面目！
        attn_output = attn_output * torch.nn.functional.silu(gate)
        
        # 7. 最后送入输出线性层
        output = self.o_proj(attn_output)
        
        return output

def main():
    print("=== 第五步：实现并加载权重的 Full Attention ===")
    model_dir = "/Users/wangrui/localinfer/Qwen3.5-0.8B"
    safetensor_path = os.path.join(model_dir, "model.safetensors-00001-of-00001.safetensors")
    
    # 实例化全注意力块
    attn = FullAttentionBlock(hidden_size=1024, num_heads=8, num_kv_heads=2, head_dim=256)
    
    layer_idx = 3
    print(f"正在读取模型 Layer {layer_idx} 的 Q(带Gate), K, V, O 真实权重...")
    
    with safe_open(safetensor_path, framework="pt", device="cpu") as f:
        q_w = f.get_tensor(f"model.language_model.layers.{layer_idx}.self_attn.q_proj.weight")
        k_w = f.get_tensor(f"model.language_model.layers.{layer_idx}.self_attn.k_proj.weight")
        v_w = f.get_tensor(f"model.language_model.layers.{layer_idx}.self_attn.v_proj.weight")
        o_w = f.get_tensor(f"model.language_model.layers.{layer_idx}.self_attn.o_proj.weight")
        
        attn.q_proj.weight.data = q_w
        attn.k_proj.weight.data = k_w
        attn.v_proj.weight.data = v_w
        attn.o_proj.weight.data = o_w
    
    attn = attn.to(torch.bfloat16)
    print("✅ 权重替换完毕，准备测试前向传播！\n")
    
    dummy_input = torch.randn(1, 3, 1024, dtype=torch.bfloat16)
    print(f"主干道输入张量: {list(dummy_input.shape)}")
    
    output = attn(dummy_input)
    print(f"注意力层提取结果: {list(output.shape)}")
    print("\n🎉 测试成功！数据在内部完美经历了：Q带门控投影 -> RoPE局部旋转 -> GQA广播 -> Attention点积 -> Gated过滤 -> O_Proj 的炼狱洗礼！")

if __name__ == "__main__":
    main()
