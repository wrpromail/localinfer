import torch

class QwenRoPE:
    def __init__(self, head_dim=256, partial_rotary_factor=0.25, base=10000000.0, max_seq_len=8192):
        self.head_dim = head_dim
        # Qwen3.5 的大坑：只旋转部分维度 (256 * 0.25 = 64)
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        
        # 1. 计算反频 (Inverse Frequency)
        # 形状: [rotary_dim // 2] 即 [32]
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim))
        
        # 2. 生成位置序列并计算角度矩阵 (Outer product)
        # t 形状: [max_seq_len]
        t = torch.arange(max_seq_len, dtype=torch.float32)
        # freqs 形状: [max_seq_len, rotary_dim // 2]
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        
        # 3. 复制两份以匹配完整的 rotary_dim
        # 形状变成 [max_seq_len, rotary_dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # 4. 提前算出全序列的 cos 和 sin，并调整形状以便广播 [1, 1, seq_len, rotary_dim]
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def _rotate_half(self, x):
        """旋转向量的一半：将 [x1, x2, x3, x4] 变成 [-x3, -x4, x1, x2]"""
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope(self, q, k, seq_len):
        # q, k 形状通常为: [batch, num_heads, seq_len, head_dim]
        
        # 【核心差异切片】：仅切出前 rotary_dim (64维) 的部分进行旋转！
        q_rot = q[..., :self.rotary_dim]
        q_pass = q[..., self.rotary_dim:]
        
        k_rot = k[..., :self.rotary_dim]
        k_pass = k[..., self.rotary_dim:]
        
        # 取出当前长度的 cos 和 sin
        cos = self.cos_cached[:, :, :seq_len, :].to(q.dtype)
        sin = self.sin_cached[:, :, :seq_len, :].to(q.dtype)
        
        # 经典的复数乘法旋转公式：(x * cos) + (rotate_half(x) * sin)
        q_rot_out = (q_rot * cos) + (self._rotate_half(q_rot) * sin)
        k_rot_out = (k_rot * cos) + (self._rotate_half(k_rot) * sin)
        
        # 最后，把旋转过的 25% 和原封不动的 75% 像拼图一样拼回去
        q_out = torch.cat((q_rot_out, q_pass), dim=-1)
        k_out = torch.cat((k_rot_out, k_pass), dim=-1)
        
        return q_out, k_out

if __name__ == "__main__":
    print("=== 第四步：测试 Qwen 专属的局部维度旋转 RoPE ===")
    # 模拟 [Batch=1, Num_Heads=1, SeqLen=3, HeadDim=256] 的 Query 和 Key
    q_dummy = torch.randn(1, 1, 3, 256)
    k_dummy = torch.randn(1, 1, 3, 256)
    
    # 初始化
    rope = QwenRoPE(head_dim=256, partial_rotary_factor=0.25)
    
    # 应用旋转
    q_out, k_out = rope.apply_rope(q_dummy, k_dummy, seq_len=3)
    
    print(f"原始 Q 形状: {list(q_dummy.shape)}")
    print(f"旋转后 Q 形状: {list(q_out.shape)}")
    print("验证未参与旋转的后 75% 维度（索引 64 之后）是否真的被原封不动地保留了：")
    diff = torch.abs(q_out[..., 64:] - q_dummy[..., 64:]).sum().item()
    print(f"  后192维度的误差总和: {diff} (如果为 0 则代表完全一致！)")
