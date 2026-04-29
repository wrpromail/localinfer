"""
10_backend_benchmark.py
========================
多后端阶梯测试框架 — Qwen3-0.6B
支持后端: CPU / Mac MPS / NVIDIA CUDA / AMD ROCm (HIP)

用法:
  # 仅 CPU
  python 10_backend_benchmark.py --backend cpu

  # Mac Metal
  python 10_backend_benchmark.py --backend mps

  # NVIDIA CUDA
  python 10_backend_benchmark.py --backend cuda

  # 全部可用后端自动扫描
  python 10_backend_benchmark.py --backend auto

阶梯参数:
  --decode-steps  生成 token 数（默认 50）
  --prompt        自定义 prompt
  --dtype         bfloat16 / float16 / float32

输出: 统一格式 JSON + 人类可读表格，方便跨后端横向对比
"""

import os
import math
import time
import json
import argparse
import resource
import platform

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import safe_open
from transformers import AutoTokenizer


# ─────────────────────────────────────────────
# 1. 模型组件 (与 09 相同，与 device/dtype 解耦)
# ─────────────────────────────────────────────

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
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class QwenRoPE:
    def __init__(self, head_dim=128, base=1_000_000.0, max_seq_len=40960):
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, None, :, :]
        self.sin_cached = emb.sin()[None, None, :, :]

    def to(self, device):
        self.cos_cached = self.cos_cached.to(device)
        self.sin_cached = self.sin_cached.to(device)
        return self

    def _rotate_half(self, x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    def apply_rope(self, q, k, position_ids):
        cos = self.cos_cached[:, :, position_ids, :].squeeze(2).to(q.dtype)
        sin = self.sin_cached[:, :, position_ids, :].squeeze(2).to(q.dtype)
        return (q * cos) + (self._rotate_half(q) * sin), \
               (k * cos) + (self._rotate_half(k) * sin)


class FullAttentionBlock(nn.Module):
    def __init__(self, hidden_size=1024, num_heads=16, num_kv_heads=8, head_dim=128):
        super().__init__()
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim     = head_dim

        self.q_proj = nn.Linear(hidden_size, num_heads    * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads   * head_dim, hidden_size,  bias=False)

        self.q_norm = QwenRMSNorm(head_dim)
        self.k_norm = QwenRMSNorm(head_dim)
        self.rope   = QwenRoPE(head_dim=head_dim)

    def forward(self, hidden_states, kv_cache=None, position_ids=None):
        B, S, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, S, self.num_heads,    self.head_dim)
        k = self.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = self.rope.apply_rope(q, k, position_ids)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2) if past_k is not None else k
            v = torch.cat([past_v, v], dim=2) if past_v is not None else v

        groups = self.num_heads // self.num_kv_heads
        k_rep  = k.repeat_interleave(groups, dim=1)
        v_rep  = v.repeat_interleave(groups, dim=1)

        # ★ 使用 SDPA — CUDA 上自动启用 FlashAttention kernel
        attn_out = F.scaled_dot_product_attention(
            q, k_rep, v_rep,
            is_causal=(S > 1)
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(attn_out), (k, v)


class Qwen3Block(nn.Module):
    def __init__(self, layer_idx):
        super().__init__()
        self.input_layernorm        = QwenRMSNorm(1024)
        self.post_attention_layernorm = QwenRMSNorm(1024)
        self.mlp       = QwenSwiGLU_MLP(1024, 3072)
        self.attention = FullAttentionBlock(1024, 16, 8, 128)

    def forward(self, x, cache, position_ids=None):
        res = x
        x, new_cache = self.attention(self.input_layernorm(x), cache, position_ids)
        x = res + x
        res = x
        x = res + self.mlp(self.post_attention_layernorm(x))
        return x, new_cache


class Qwen3Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(151936, 1024)
        self.layers       = nn.ModuleList([Qwen3Block(i) for i in range(28)])
        self.norm         = QwenRMSNorm(1024)
        self.lm_head      = nn.Linear(1024, 151936, bias=False)

    def forward_step(self, token_ids, caches, position_ids):
        x = self.embed_tokens(token_ids)
        new_caches = []
        for i, layer in enumerate(self.layers):
            x, c = layer(x, caches[i], position_ids)
            new_caches.append(c)
        return self.lm_head(self.norm(x)), new_caches


# ─────────────────────────────────────────────
# 2. 权重加载
# ─────────────────────────────────────────────

def load_weights(model, safetensor_path, device, dtype):
    print(f"  加载权重 → device={device}, dtype={dtype}")
    with safe_open(safetensor_path, framework="pt", device="cpu") as f:
        model.embed_tokens.weight.data = f.get_tensor("model.embed_tokens.weight")
        model.norm.weight.data         = f.get_tensor("model.norm.weight")
        if "lm_head.weight" in f.keys():
            model.lm_head.weight.data  = f.get_tensor("lm_head.weight")
        else:
            model.lm_head.weight.data  = model.embed_tokens.weight.data.clone()

        for i in range(28):
            layer  = model.layers[i]
            prefix = f"model.layers.{i}."
            layer.input_layernorm.weight.data          = f.get_tensor(f"{prefix}input_layernorm.weight")
            layer.post_attention_layernorm.weight.data = f.get_tensor(f"{prefix}post_attention_layernorm.weight")
            layer.mlp.gate_proj.weight.data = f.get_tensor(f"{prefix}mlp.gate_proj.weight")
            layer.mlp.up_proj.weight.data   = f.get_tensor(f"{prefix}mlp.up_proj.weight")
            layer.mlp.down_proj.weight.data = f.get_tensor(f"{prefix}mlp.down_proj.weight")
            attn = layer.attention
            attn.q_proj.weight.data = f.get_tensor(f"{prefix}self_attn.q_proj.weight")
            attn.k_proj.weight.data = f.get_tensor(f"{prefix}self_attn.k_proj.weight")
            attn.v_proj.weight.data = f.get_tensor(f"{prefix}self_attn.v_proj.weight")
            attn.o_proj.weight.data = f.get_tensor(f"{prefix}self_attn.o_proj.weight")
            attn.q_norm.weight.data = f.get_tensor(f"{prefix}self_attn.q_norm.weight")
            attn.k_norm.weight.data = f.get_tensor(f"{prefix}self_attn.k_norm.weight")

    model = model.to(dtype=dtype, device=device)

    # 将 RoPE 缓存也移动到目标设备
    for layer in model.layers:
        layer.attention.rope.to(device)

    print("  ✅ 权重加载完毕")
    return model


# ─────────────────────────────────────────────
# 3. 阶梯 Benchmark 核心
# ─────────────────────────────────────────────

def get_memory_mb(device_str: str) -> float:
    """返回当前显存/内存占用 (MB)"""
    if "cuda" in device_str:
        return torch.cuda.memory_allocated() / 1024**2
    if "mps" in device_str:
        return torch.mps.current_allocated_memory() / 1024**2
    # CPU: 使用 RSS
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return rss / 1024**2
    return rss / 1024


def run_benchmark(model, tokenizer, device, prompt, decode_steps) -> dict:
    """执行 prefill + decode，返回性能数据字典"""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    num_prompt_tokens = input_ids.shape[1]
    caches = [None] * 28

    # ── Prefill ──────────────────────────────
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(num_prompt_tokens - 1):
            tid = input_ids[:, i:i+1]
            pos = torch.tensor([i], dtype=torch.long, device=device)
            _, caches = model.forward_step(tid, caches, pos)
    if "cuda" in device:
        torch.cuda.synchronize()
    elif "mps" in device:
        torch.mps.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000

    # ── Decode (阶梯计时) ────────────────────
    current_token = input_ids[:, -1:]
    current_pos   = num_prompt_tokens - 1
    token_times_ms = []
    generated_tokens = []

    for step in range(decode_steps):
        pos = torch.tensor([current_pos], dtype=torch.long, device=device)
        t_s = time.perf_counter()
        with torch.no_grad():
            logits, caches = model.forward_step(current_token, caches, pos)
        if "cuda" in device:
            torch.cuda.synchronize()
        elif "mps" in device:
            torch.mps.synchronize()
        t_ms = (time.perf_counter() - t_s) * 1000
        token_times_ms.append(t_ms)

        next_token    = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        current_token = next_token
        current_pos  += 1
        generated_tokens.append(next_token.item())

    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # ── 汇总 ─────────────────────────────────
    avg_tbt   = sum(token_times_ms) / len(token_times_ms)
    ttft      = prefill_ms + token_times_ms[0]  # Time To First Token
    speed     = 1000 / avg_tbt                  # tokens/s
    mem_mb    = get_memory_mb(device)

    return {
        "prefill_ms"     : round(prefill_ms, 2),
        "ttft_ms"        : round(ttft, 2),
        "avg_tbt_ms"     : round(avg_tbt, 2),
        "speed_tps"      : round(speed, 2),
        "mem_mb"         : round(mem_mb, 1),
        "token_times_ms" : [round(t, 2) for t in token_times_ms],
        "generated_text" : generated_text,
        "decode_steps"   : decode_steps,
        "prompt_tokens"  : num_prompt_tokens,
    }


# ─────────────────────────────────────────────
# 4. 后端探测
# ─────────────────────────────────────────────

def detect_backends():
    backends = ["cpu"]
    if torch.backends.mps.is_available():
        backends.append("mps")
    if torch.cuda.is_available():
        backends.append("cuda")
    return backends


def backend_info(device_str: str) -> str:
    if device_str == "cpu":
        return f"CPU ({platform.processor() or platform.machine()})"
    if device_str == "mps":
        return "Apple Metal (MPS)"
    if device_str.startswith("cuda"):
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        mem  = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        return f"CUDA — {name} ({mem:.1f} GB)"
    return device_str


# ─────────────────────────────────────────────
# 5. 打印报告
# ─────────────────────────────────────────────

def print_report(results: list[dict]):
    sep = "─" * 72
    print(f"\n{sep}")
    print("  📊  多后端阶梯测试报告  (Qwen3-0.6B)")
    print(sep)
    print(f"{'后端':<30} {'TTFT(ms)':>10} {'TBT(ms)':>10} {'速度(t/s)':>10} {'内存(MB)':>10}")
    print(sep)
    for r in results:
        print(f"{r['backend_label']:<30} {r['ttft_ms']:>10.1f} {r['avg_tbt_ms']:>10.1f} {r['speed_tps']:>10.1f} {r['mem_mb']:>10.1f}")
    print(sep)

    # 逐 Token 阶梯耗时
    print("\n  ⏱  逐 Token 耗时 (ms) — 阶梯视图")
    print(sep)
    headers = ["step"] + [r["backend_label"][:20] for r in results]
    col_w = 22
    print("  " + "".join(h.ljust(col_w) for h in headers))
    max_steps = max(len(r["token_times_ms"]) for r in results)
    for step in range(max_steps):
        row = [f"Token {step+1:02d}"]
        for r in results:
            tms = r["token_times_ms"]
            row.append(f"{tms[step]:.2f}" if step < len(tms) else "—")
        print("  " + "".join(c.ljust(col_w) for c in row))
    print(sep)

    print("\n  💬  生成内容预览")
    for r in results:
        print(f"  [{r['backend_label']}] {r['generated_text'][:80]}")
    print()


# ─────────────────────────────────────────────
# 6. 主程序
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Qwen3-0.6B 多后端阶梯测试")
    parser.add_argument("--backend",      default="auto",
                        help="cpu / mps / cuda / auto（自动扫描所有可用后端）")
    parser.add_argument("--model-dir",    default="/Users/wangrui/localinfer/Qwen3-0.6B")
    parser.add_argument("--decode-steps", type=int, default=30,  help="生成的 token 数")
    parser.add_argument("--prompt",       default="人工智能的未来是")
    parser.add_argument("--dtype",        default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--output-json",  default="", help="保存结果到 JSON 文件（可选）")
    args = parser.parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # 确定要测试的后端列表
    if args.backend == "auto":
        backends = detect_backends()
        print(f"  🔍 自动探测到可用后端: {backends}")
    else:
        backends = [args.backend]

    safetensor_path = os.path.join(args.model_dir, "model.safetensors")
    tokenizer       = AutoTokenizer.from_pretrained(args.model_dir)

    all_results = []

    for backend in backends:
        print(f"\n{'='*60}")
        label = backend_info(backend)
        print(f"  🚀 测试后端: {label}")
        print(f"     dtype={args.dtype}  steps={args.decode_steps}  prompt='{args.prompt}'")
        print(f"{'='*60}")

        # MPS 不完全支持 bfloat16，自动降级
        effective_dtype = dtype
        if backend == "mps" and dtype == torch.bfloat16:
            print("  ⚠️  MPS 对 bfloat16 支持有限，自动降级为 float16")
            effective_dtype = torch.float16

        try:
            model = Qwen3Model()
            model = load_weights(model, safetensor_path, backend, effective_dtype)
            model.eval()

            result = run_benchmark(
                model, tokenizer, backend,
                args.prompt, args.decode_steps
            )
            result["backend"]       = backend
            result["backend_label"] = label
            result["dtype"]         = str(effective_dtype).replace("torch.", "")
            all_results.append(result)

            print(f"  ✅ TTFT={result['ttft_ms']}ms  "
                  f"TBT={result['avg_tbt_ms']}ms  "
                  f"Speed={result['speed_tps']}t/s  "
                  f"Mem={result['mem_mb']}MB")

            # 释放显存
            del model
            if backend == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ❌ 后端 {backend} 测试失败: {e}")

    if all_results:
        print_report(all_results)

        if args.output_json:
            # token_times_ms 列表在 JSON 保留完整
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"  📁 结果已保存 → {args.output_json}")


if __name__ == "__main__":
    main()
