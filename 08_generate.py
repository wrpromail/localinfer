import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import safe_open
from transformers import AutoTokenizer

# --- 1. 基础组件 ---

class QwenRMSNorm(nn.Module):
    def __init__(self, hidden_size=1024, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class QwenSwiGLU_MLP(nn.Module):
    def __init__(self, hidden_size=1024, intermediate_size=3584):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class QwenRoPE:
    def __init__(self, head_dim=256, partial_rotary_factor=0.25, base=10000000.0, max_seq_len=8192):
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def _rotate_half(self, x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    def apply_rope(self, q, k, position_ids):
        q_rot = q[..., :self.rotary_dim]
        q_pass = q[..., self.rotary_dim:]
        k_rot = k[..., :self.rotary_dim]
        k_pass = k[..., self.rotary_dim:]
        
        cos = self.cos_cached[:, :, position_ids, :].squeeze(2).to(q.dtype) # [1, 1, seq, dim] -> [1, 1, 1, dim]
        sin = self.sin_cached[:, :, position_ids, :].squeeze(2).to(q.dtype)
        
        q_rot_out = (q_rot * cos) + (self._rotate_half(q_rot) * sin)
        k_rot_out = (k_rot * cos) + (self._rotate_half(k_rot) * sin)
        
        return torch.cat((q_rot_out, q_pass), dim=-1), torch.cat((k_rot_out, k_pass), dim=-1)

# --- 2. 核心架构层 ---

class FullAttentionBlock(nn.Module):
    def __init__(self, hidden_size=1024, num_heads=8, num_kv_heads=2, head_dim=256):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim * 2, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
        self.q_norm = QwenRMSNorm(head_dim)
        self.k_norm = QwenRMSNorm(head_dim)
        self.rope = QwenRoPE(head_dim=head_dim, partial_rotary_factor=0.25)

    def forward(self, hidden_states, kv_cache=None, position_ids=None):
        batch_size, seq_len, _ = hidden_states.shape
        q_and_gate = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        q = q_and_gate[..., : self.num_heads * self.head_dim]
        gate = q_and_gate[..., self.num_heads * self.head_dim :]
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        q, k = self.rope.apply_rope(q, k, position_ids)
        
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2) if past_k is not None else k
            v = torch.cat([past_v, v], dim=2) if past_v is not None else v
            kv_cache = (k, v)
            
        num_key_value_groups = self.num_heads // self.num_kv_heads
        k_rep = torch.repeat_interleave(k, repeats=num_key_value_groups, dim=1)
        v_rep = torch.repeat_interleave(v, repeats=num_key_value_groups, dim=1)
        
        scores = torch.matmul(q, k_rep.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v_rep)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = attn_output * F.silu(gate)
        return self.o_proj(attn_output), kv_cache

def l2norm(x, dim=-1, eps=1e-6):
    return x / x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)

class QwenLinearAttentionBlock(nn.Module):
    def __init__(self, hidden_size=1024, num_heads=16, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        
        self.in_proj_qkv = nn.Linear(hidden_size, self.inner_dim * 3, bias=False)
        self.in_proj_z = nn.Linear(hidden_size, self.inner_dim, bias=False)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.inner_dim * 3, out_channels=self.inner_dim * 3, 
            kernel_size=4, groups=self.inner_dim * 3, padding=3
        )
        
        self.in_proj_a = nn.Linear(hidden_size, num_heads, bias=False)
        self.in_proj_b = nn.Linear(hidden_size, num_heads, bias=False)
        self.A_log = nn.Parameter(torch.empty(num_heads))
        self.dt_bias = nn.Parameter(torch.empty(num_heads))
        
        self.norm = QwenRMSNorm(head_dim)
        self.out_proj = nn.Linear(self.inner_dim, hidden_size, bias=False)

    def forward(self, hidden_states, rnn_state=None, conv_state=None):
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.in_proj_qkv(hidden_states)
        
        if conv_state is None:
            conv_state = torch.zeros(batch_size, self.inner_dim * 3, 3, dtype=qkv.dtype, device=qkv.device)
        
        x_in = qkv.transpose(1, 2)
        conv_input = torch.cat([conv_state, x_in], dim=2)
        conv_state = conv_input[:, :, 1:]
        
        qkv_conv = F.conv1d(conv_input, self.conv1d.weight, self.conv1d.bias, groups=self.inner_dim * 3)
        qkv_conv = F.silu(qkv_conv).transpose(1, 2)
        
        q, k, v = qkv_conv.split(self.inner_dim, dim=-1)
        q = q.view(batch_size, 1, self.num_heads, self.head_dim)
        k = k.view(batch_size, 1, self.num_heads, self.head_dim)
        v = v.view(batch_size, 1, self.num_heads, self.head_dim)
        
        z = self.in_proj_z(hidden_states)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        
        q = l2norm(q, dim=-1, eps=1e-6).to(torch.float32)
        k = l2norm(k, dim=-1, eps=1e-6).to(torch.float32)
        v, beta, g = v.to(torch.float32), beta.to(torch.float32), g.to(torch.float32)
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        beta = beta.unsqueeze(1).transpose(1, 2)
        g = g.unsqueeze(1).transpose(1, 2)
        
        # 【关键修正】：必须对 Q 进行缩放，否则点积会引发数值爆炸！
        scale = 1.0 / math.sqrt(q.shape[-1])
        q = q * scale
        
        if rnn_state is None:
            rnn_state = torch.zeros(batch_size, self.num_heads, self.head_dim, self.head_dim, dtype=torch.float32, device=q.device)
            
        q_t, k_t, v_t = q[:, :, 0], k[:, :, 0], v[:, :, 0]
        g_t = g[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, 0].unsqueeze(-1)
        
        # ★ GLA (Gated Linear Attention) Delta Rule 核心更新
        # 完整数学推导：
        # 1. rnn_state = rnn_state * exp(g_t)   => 旧记忆按当前词的“遗忘门”指数衰减
        # 2. kv_mem = (rnn_state * k_t).sum(-2) => 用 K_t 从旧状态中检索已有预期 V
        # 3. delta = (v_t - kv_mem) * beta_t    => 计算“预期偏差”，beta 控制融入比例
        # 4. rnn_state += k_t * delta            => 将修正后的新知识写入记忆矩阵
        # 数学上等价于在线最小二乘更新（online least-squares update）
        rnn_state = rnn_state * g_t
        kv_mem = (rnn_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        rnn_state = rnn_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out = (rnn_state * q_t.unsqueeze(-1)).sum(dim=-2)
        
        core_attn_out = core_attn_out.to(hidden_states.dtype).transpose(1, 2).contiguous()
        core_attn_out = self.norm(core_attn_out).view(batch_size, 1, self.inner_dim)
        core_attn_out = core_attn_out * F.silu(z)
        
        return self.out_proj(core_attn_out), rnn_state, conv_state

# --- 3. 层与模型组装 ---

class Qwen3_5_Block(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = QwenRMSNorm(1024)
        self.post_attention_layernorm = QwenRMSNorm(1024)
        self.mlp = QwenSwiGLU_MLP(1024, 3584)
        
        self.is_full_attn = ((layer_idx + 1) % 4 == 0)
        if self.is_full_attn:
            self.attention = FullAttentionBlock(1024, 8, 2, 256)
        else:
            self.attention = QwenLinearAttentionBlock(1024, 16, 128)

    def forward(self, x, cache, position_ids=None):
        residual = x
        x_normed = self.input_layernorm(x)
        
        if self.is_full_attn:
            attn_out, new_cache = self.attention(x_normed, cache, position_ids)
        else:
            rnn_state, conv_state = cache if cache is not None else (None, None)
            attn_out, rnn_state, conv_state = self.attention(x_normed, rnn_state, conv_state)
            new_cache = (rnn_state, conv_state)
            
        x = residual + attn_out
        
        residual = x
        x_normed = self.post_attention_layernorm(x)
        mlp_out = self.mlp(x_normed)
        x = residual + mlp_out
        
        return x, new_cache

class Qwen3_5_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(248320, 1024)
        self.layers = nn.ModuleList([Qwen3_5_Block(i) for i in range(24)])
        self.norm = QwenRMSNorm(1024)
        
    def forward_step(self, token_ids, caches, position_ids):
        x = self.embed_tokens(token_ids)
        new_caches = []
        for i, layer in enumerate(self.layers):
            x, new_cache = layer(x, caches[i], position_ids)
            new_caches.append(new_cache)
        x = self.norm(x)
        logits = torch.matmul(x, self.embed_tokens.weight.T)
        return logits, new_caches

# --- 4. 权重加载与执行生成 ---

def load_weights(model, safetensor_path):
    print("正在逆向加载 1.7GB 的完整模型权重...")
    with safe_open(safetensor_path, framework="pt", device="cpu") as f:
        model.embed_tokens.weight.data = f.get_tensor("model.language_model.embed_tokens.weight")
        model.norm.weight.data = f.get_tensor("model.language_model.norm.weight")
        
        for i in range(24):
            layer = model.layers[i]
            # 前缀检测：Qwen3.5 某些实验版本（Multi-Token Prediction 分支）会把权重存在
            # "mtp.layers.{i}." 前缀下；若不存在则回落到标准的 "model.language_model.layers.{i}."。
            # 目前公开发布的可用权重均使用标准前缀，mtp 分支仅在实验性 checkpoint 中出现。
            prefix = f"model.language_model.layers.{i}." if f"model.language_model.layers.{i}.input_layernorm.weight" in f.keys() else f"mtp.layers.{i}."
            if prefix == f"mtp.layers.{i}." and f"{prefix}input_layernorm.weight" not in f.keys():
                prefix = f"model.language_model.layers.{i}."
                
            layer.input_layernorm.weight.data = f.get_tensor(f"{prefix}input_layernorm.weight")
            layer.post_attention_layernorm.weight.data = f.get_tensor(f"{prefix}post_attention_layernorm.weight")
            layer.mlp.gate_proj.weight.data = f.get_tensor(f"{prefix}mlp.gate_proj.weight")
            layer.mlp.up_proj.weight.data = f.get_tensor(f"{prefix}mlp.up_proj.weight")
            layer.mlp.down_proj.weight.data = f.get_tensor(f"{prefix}mlp.down_proj.weight")
            
            if layer.is_full_attn:
                attn = layer.attention
                attn.q_proj.weight.data = f.get_tensor(f"{prefix}self_attn.q_proj.weight")
                attn.k_proj.weight.data = f.get_tensor(f"{prefix}self_attn.k_proj.weight")
                attn.v_proj.weight.data = f.get_tensor(f"{prefix}self_attn.v_proj.weight")
                attn.o_proj.weight.data = f.get_tensor(f"{prefix}self_attn.o_proj.weight")
                attn.q_norm.weight.data = f.get_tensor(f"{prefix}self_attn.q_norm.weight")
                attn.k_norm.weight.data = f.get_tensor(f"{prefix}self_attn.k_norm.weight")
            else:
                attn = layer.attention
                attn.in_proj_qkv.weight.data = f.get_tensor(f"{prefix}linear_attn.in_proj_qkv.weight")
                attn.in_proj_z.weight.data = f.get_tensor(f"{prefix}linear_attn.in_proj_z.weight")
                attn.in_proj_a.weight.data = f.get_tensor(f"{prefix}linear_attn.in_proj_a.weight")
                attn.in_proj_b.weight.data = f.get_tensor(f"{prefix}linear_attn.in_proj_b.weight")
                attn.conv1d.weight.data = f.get_tensor(f"{prefix}linear_attn.conv1d.weight")
                attn.A_log.data = f.get_tensor(f"{prefix}linear_attn.A_log")
                attn.dt_bias.data = f.get_tensor(f"{prefix}linear_attn.dt_bias")
                attn.norm.weight.data = f.get_tensor(f"{prefix}linear_attn.norm.weight")
                attn.out_proj.weight.data = f.get_tensor(f"{prefix}linear_attn.out_proj.weight")
    
    model = model.to(torch.bfloat16)
    print("✅ 权重组装完毕！")
    return model

def main():
    print("\n=== 第八步：大一统！端到端文本生成引擎 ===\n")
    
    model_dir = "/Users/wangrui/localinfer/Qwen3.5-0.8B"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    model = Qwen3_5_Model()
    model = load_weights(model, os.path.join(model_dir, "model.safetensors-00001-of-00001.safetensors"))
    model.eval()
    
    prompt = "人工智能的未来是"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    print(f"\n[用户 Prompt]: {prompt}")
    print(f"[Token IDs]: {input_ids.tolist()[0]}")
    
    caches = [None] * 24
    
    # === Prefill 阶段 (逐词暖机) ===
    # ⚠️ 设计限制说明：此处采用「token-by-token」逐词串行消化 prompt。
    # 原因：Linear Attention 的状态机必须依赖上一时刻的状态， Python 层无法实现并行前缀扫描算法。
    # 副作用：全注意力层在 prefill 期间也是逐步积累 KV Cache，而非标准的“一次看全 prompt”。
    # 这是一种近似：对模型最终生成结果影响有限，但导致冷启动速度深受制限。
    print("\n[ 系统正在逐词流式消化 Prompt (Prefill)... ]")
    for i in range(input_ids.shape[1] - 1):
        token_id = input_ids[:, i:i+1]
        pos_id = torch.tensor([i], dtype=torch.long)
        with torch.no_grad():
            _, caches = model.forward_step(token_id, caches, pos_id)
        
    current_token = input_ids[:, -1:]
    current_pos = input_ids.shape[1] - 1
    
    # === Decode 阶段 (自回归生成) ===
    print("\n[ 状态机预热完毕，开始自回归吐词！ ]\n")
    print(f"🤖 >> {prompt}", end="")
    
    for step in range(15):
        pos_id = torch.tensor([current_pos], dtype=torch.long)
        with torch.no_grad():
            logits, caches = model.forward_step(current_token, caches, pos_id)
        
        # 使用贪心解码拿概率最大的词
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        word = tokenizer.decode(next_token[0])
        print(word, end="", flush=True)
        
        current_token = next_token
        current_pos += 1
        
    print("\n\n🎉 奇迹降临！我们的组装机甲成功根据 Prompt 进行了续写推演！")

if __name__ == "__main__":
    main()
