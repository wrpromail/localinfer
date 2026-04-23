import os
import math
import time
import resource
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import safe_open
from transformers import AutoTokenizer

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
    def __init__(self, hidden_size=1024, intermediate_size=3072):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class QwenRoPE:
    def __init__(self, head_dim=128, base=1000000.0, max_seq_len=40960):
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def _rotate_half(self, x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    def apply_rope(self, q, k, position_ids):
        cos = self.cos_cached[:, :, position_ids, :].squeeze(2).to(q.dtype) 
        sin = self.sin_cached[:, :, position_ids, :].squeeze(2).to(q.dtype)
        
        q_out = (q * cos) + (self._rotate_half(q) * sin)
        k_out = (k * cos) + (self._rotate_half(k) * sin)
        return q_out, k_out

class FullAttentionBlock(nn.Module):
    def __init__(self, hidden_size=1024, num_heads=16, num_kv_heads=8, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
        self.q_norm = QwenRMSNorm(head_dim)
        self.k_norm = QwenRMSNorm(head_dim)
        self.rope = QwenRoPE(head_dim=head_dim, base=1000000.0)

    def forward(self, hidden_states, kv_cache=None, position_ids=None):
        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
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
        
        if seq_len > 1:
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=scores.device), diagonal=1)
            scores.masked_fill_(causal_mask, float('-inf'))
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v_rep)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(attn_output), kv_cache

class Qwen3Block(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = QwenRMSNorm(1024)
        self.post_attention_layernorm = QwenRMSNorm(1024)
        self.mlp = QwenSwiGLU_MLP(1024, 3072)
        self.attention = FullAttentionBlock(1024, 16, 8, 128)

    def forward(self, x, cache, position_ids=None):
        residual = x
        x_normed = self.input_layernorm(x)
        attn_out, new_cache = self.attention(x_normed, cache, position_ids)
        x = residual + attn_out
        
        residual = x
        x_normed = self.post_attention_layernorm(x)
        mlp_out = self.mlp(x_normed)
        x = residual + mlp_out
        
        return x, new_cache

class Qwen3Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(151936, 1024)
        self.layers = nn.ModuleList([Qwen3Block(i) for i in range(28)])
        self.norm = QwenRMSNorm(1024)
        self.lm_head = nn.Linear(1024, 151936, bias=False)
        
    def forward_step(self, token_ids, caches, position_ids):
        x = self.embed_tokens(token_ids)
        new_caches = []
        for i, layer in enumerate(self.layers):
            x, new_cache = layer(x, caches[i], position_ids)
            new_caches.append(new_cache)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_caches

def load_weights(model, safetensor_path):
    print("正在加载 Qwen3-0.6B 模型权重...")
    try:
        with safe_open(safetensor_path, framework="pt", device="cpu") as f:
            model.embed_tokens.weight.data = f.get_tensor("model.embed_tokens.weight")
            model.norm.weight.data = f.get_tensor("model.norm.weight")
            
            if "lm_head.weight" in f.keys():
                model.lm_head.weight.data = f.get_tensor("lm_head.weight")
            else:
                model.lm_head.weight.data = model.embed_tokens.weight.data.clone()
            
            for i in range(28):
                layer = model.layers[i]
                prefix = f"model.layers.{i}."
                    
                layer.input_layernorm.weight.data = f.get_tensor(f"{prefix}input_layernorm.weight")
                layer.post_attention_layernorm.weight.data = f.get_tensor(f"{prefix}post_attention_layernorm.weight")
                
                layer.mlp.gate_proj.weight.data = f.get_tensor(f"{prefix}mlp.gate_proj.weight")
                layer.mlp.up_proj.weight.data = f.get_tensor(f"{prefix}mlp.up_proj.weight")
                layer.mlp.down_proj.weight.data = f.get_tensor(f"{prefix}mlp.down_proj.weight")
                
                attn = layer.attention
                attn.q_proj.weight.data = f.get_tensor(f"{prefix}self_attn.q_proj.weight")
                attn.k_proj.weight.data = f.get_tensor(f"{prefix}self_attn.k_proj.weight")
                attn.v_proj.weight.data = f.get_tensor(f"{prefix}self_attn.v_proj.weight")
                attn.o_proj.weight.data = f.get_tensor(f"{prefix}self_attn.o_proj.weight")
                
                attn.q_norm.weight.data = f.get_tensor(f"{prefix}self_attn.q_norm.weight")
                attn.k_norm.weight.data = f.get_tensor(f"{prefix}self_attn.k_norm.weight")

    except Exception as e:
        print(f"权重加载失败！错误详情: {e}")
        # Optionally print keys for debugging
        with safe_open(safetensor_path, framework="pt", device="cpu") as f:
            keys = f.keys()
            print("可用键的前20个:", list(keys)[:20])
        raise e
        
    model = model.to(torch.bfloat16)
    print("✅ 权重组装完毕！")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-kv-cache", action="store_true", help="Disable KV Cache and recompute full sequence")
    args = parser.parse_args()
    use_kv_cache = not args.no_kv_cache

    print("\n=== Qwen3-0.6B 纯全注意力推理流水线 ===\n")
    print(f"模式: {'KV Cache ON' if use_kv_cache else 'KV Cache OFF (Full Recompute)'}")
    
    model_dir = "/Users/wangrui/localinfer/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    model = Qwen3Model()
    
    safetensor_path = os.path.join(model_dir, "model.safetensors")
    
    model = load_weights(model, safetensor_path)
    model.eval()
    # 注：model.to(bfloat16) 已在 load_weights() 内部完成，此处无需重复调用
    
    prompt = "人工智能的未来是"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    print(f"\n[用户 Prompt]: {prompt}")
    print(f"[Token IDs]: {input_ids.tolist()[0]}")
    
    caches = [None] * 28
    
    # === Prefill 阶段 ===
    if use_kv_cache:
        print("\n[ 系统正在逐词消化 Prompt (Prefill)... ]")
        prefill_start_time = time.time()
        for i in range(input_ids.shape[1] - 1):
            token_id = input_ids[:, i:i+1]
            pos_id = torch.tensor([i], dtype=torch.long)
            with torch.no_grad():
                _, caches = model.forward_step(token_id, caches, pos_id)
        prefill_end_time = time.time()
        prefill_time = prefill_end_time - prefill_start_time
    else:
        prefill_time = 0.0
        
    # === Decode 阶段 ===
    # 两种模式的复杂度对比：
    # - KV Cache ON (默认)：每步仅处理 1 个 token，计算量 O(1)，延迟稳定
    # - KV Cache OFF：每步将整个序列重新输入，第 k 步计算量 O(k)，总计 O(N²)，延迟线性增长
    print(f"\n[ 状态预热完毕，开始自回归吐词！(KV Cache: {'ON' if use_kv_cache else 'OFF'}) ]\n")
    print(f"🤖 >> {prompt}", end="")
    
    decode_start_time = time.time()
    decode_tokens = 20
    token_times = []
    
    if use_kv_cache:
        current_token = input_ids[:, -1:]
        current_pos = input_ids.shape[1] - 1
        
        for step in range(decode_tokens):
            step_start = time.time()
            pos_id = torch.tensor([current_pos], dtype=torch.long)
            with torch.no_grad():
                logits, caches = model.forward_step(current_token, caches, pos_id)
            
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            word = tokenizer.decode(next_token[0])
            print(word, end="", flush=True)
            
            current_token = next_token
            current_pos += 1
            token_times.append((time.time() - step_start) * 1000)
    else:
        current_seq = input_ids
        for step in range(decode_tokens):
            step_start = time.time()
            caches_none = [None] * 28
            pos_ids = torch.arange(current_seq.shape[1], dtype=torch.long)
            with torch.no_grad():
                logits, _ = model.forward_step(current_seq, caches_none, pos_ids)
            
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            word = tokenizer.decode(next_token[0])
            print(word, end="", flush=True)
            
            current_seq = torch.cat([current_seq, next_token], dim=1)
            token_times.append((time.time() - step_start) * 1000)
        
    decode_end_time = time.time()
    decode_time = decode_end_time - decode_start_time
    
    print("\n\n🎉 推理完成！")
    
    # === 性能统计 ===
    print("\n=== 📊 性能统计 (Profiling) ===")
    print(f"Prefill 耗时: {prefill_time:.4f} s" + (f" ({input_ids.shape[1] - 1} tokens)" if use_kv_cache else " (Skipped)"))
    print(f"Decode 总耗时: {decode_time:.4f} s ({decode_tokens} tokens)")
    print(f"生成速度 (Decode): {decode_tokens / decode_time:.2f} tokens/s")
    print(f"单 Token 平均耗时: {(decode_time / decode_tokens) * 1000:.2f} ms/token")
    
    print("\n[逐 Token 耗时 (ms)]:")
    for i, t_ms in enumerate(token_times):
        print(f"  Token {i+1:02d}: {t_ms:.2f} ms")
    
    # Mac 上的 ru_maxrss 单位是 bytes，Linux 上是 kilobytes
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == 'Darwin':
        memory_mb = max_rss / (1024 * 1024)
    else:
        memory_mb = max_rss / 1024
    print(f"峰值内存消耗 (Peak RAM): {memory_mb:.2f} MB")

if __name__ == "__main__":
    main()
