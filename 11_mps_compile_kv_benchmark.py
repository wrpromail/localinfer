#!/usr/bin/env python3
"""
11_mps_compile_kv_benchmark.py
================================
在 Mac MPS 上对比开启 KV Cache 时 torch.compile 与 eager 的 decode 性能。

默认路径使用预分配 KV Cache：
  - 不在 decode 中 torch.cat 扩容。
  - 每次 run 都重新初始化 cache，避免不同测试互相污染。
  - attention 使用固定 max_cache_len 的 K/V 和固定形状 mask，减少图重编译。
"""

import argparse
import importlib
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer


backend_mod = importlib.import_module("10_backend_benchmark")
Qwen3Model = backend_mod.Qwen3Model
load_weights = backend_mod.load_weights
get_memory_mb = backend_mod.get_memory_mb

NUM_LAYERS = 28
BATCH_SIZE = 1
NUM_KV_HEADS = 8
HEAD_DIM = 128


def sync_device(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }[name]


class StaticKVForward(nn.Module):
    def __init__(self, model: Qwen3Model, max_cache_len: int, device: str):
        super().__init__()
        self.model = model
        self.max_cache_len = max_cache_len
        self.register_buffer(
            "cache_positions",
            torch.arange(max_cache_len, dtype=torch.long, device=device),
            persistent=False,
        )

    def forward(self, token_ids, key_cache, value_cache, cache_position):
        x = self.model.embed_tokens(token_ids)

        for layer_idx, layer in enumerate(self.model.layers):
            residual = x
            hidden_states = layer.input_layernorm(x)
            attn = layer.attention
            batch_size, seq_len, _ = hidden_states.shape

            q = attn.q_proj(hidden_states).view(batch_size, seq_len, attn.num_heads, attn.head_dim)
            k = attn.k_proj(hidden_states).view(batch_size, seq_len, attn.num_kv_heads, attn.head_dim)
            v = attn.v_proj(hidden_states).view(batch_size, seq_len, attn.num_kv_heads, attn.head_dim)

            q = attn.q_norm(q).transpose(1, 2)
            k = attn.k_norm(k).transpose(1, 2)
            v = v.transpose(1, 2)
            q, k = attn.rope.apply_rope(q, k, cache_position)

            key_cache[layer_idx].index_copy_(2, cache_position, k)
            value_cache[layer_idx].index_copy_(2, cache_position, v)

            groups = attn.num_heads // attn.num_kv_heads
            k_rep = key_cache[layer_idx].repeat_interleave(groups, dim=1)
            v_rep = value_cache[layer_idx].repeat_interleave(groups, dim=1)
            allowed_positions = (self.cache_positions <= cache_position[0]).view(1, 1, 1, self.max_cache_len)

            attn_out = F.scaled_dot_product_attention(
                q,
                k_rep,
                v_rep,
                attn_mask=allowed_positions,
                is_causal=False,
            )
            attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            x = residual + attn.o_proj(attn_out)

            residual = x
            x = residual + layer.mlp(layer.post_attention_layernorm(x))

        return self.model.lm_head(self.model.norm(x))


def allocate_static_cache(max_cache_len: int, dtype: torch.dtype, device: str):
    shape = (NUM_LAYERS, BATCH_SIZE, NUM_KV_HEADS, max_cache_len, HEAD_DIM)
    key_cache = torch.empty(shape, dtype=dtype, device=device)
    value_cache = torch.empty(shape, dtype=dtype, device=device)
    return key_cache, value_cache


def make_static_forward(model: Qwen3Model, max_cache_len: int, device: str,
                        use_compile: bool, dynamic_compile: bool):
    runner = StaticKVForward(model, max_cache_len, device).eval()
    if not use_compile:
        return runner, 0.0

    print("  编译 StaticKVForward: torch.compile(mode='reduce-overhead')")
    t0 = time.perf_counter()
    compiled_runner = torch.compile(
        runner,
        mode="reduce-overhead",
        dynamic=dynamic_compile,
    )
    return compiled_runner, time.perf_counter() - t0


def run_static_kv_once(runner, tokenizer, prompt: str, decode_steps: int,
                       max_cache_len: int, dtype: torch.dtype, device: str) -> dict:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_tokens = input_ids.shape[1]
    total_required = prompt_tokens + decode_steps
    if total_required > max_cache_len:
        raise ValueError(f"prompt_tokens + decode_steps = {total_required}, 超过 max_cache_len={max_cache_len}")

    key_cache, value_cache = allocate_static_cache(max_cache_len, dtype, device)
    generated_tokens = []

    sync_device(device)
    prefill_start = time.perf_counter()
    with torch.inference_mode():
        for index in range(prompt_tokens - 1):
            token_id = input_ids[:, index:index + 1]
            position_id = torch.tensor([index], dtype=torch.long, device=device)
            runner(token_id, key_cache, value_cache, position_id)
    sync_device(device)
    prefill_ms = (time.perf_counter() - prefill_start) * 1000

    current_token = input_ids[:, -1:]
    current_pos = prompt_tokens - 1
    token_times_ms = []
    decode_start = time.perf_counter()

    with torch.inference_mode():
        for _ in range(decode_steps):
            position_id = torch.tensor([current_pos], dtype=torch.long, device=device)
            step_start = time.perf_counter()
            logits = runner(current_token, key_cache, value_cache, position_id)
            sync_device(device)
            token_times_ms.append((time.perf_counter() - step_start) * 1000)

            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated_tokens.append(next_token.item())
            current_token = next_token
            current_pos += 1

    sync_device(device)
    decode_total_ms = (time.perf_counter() - decode_start) * 1000
    avg_tbt_ms = sum(token_times_ms) / len(token_times_ms)

    return {
        "kv_cache_impl": "static_preallocated",
        "decode_steps": decode_steps,
        "prompt_tokens": prompt_tokens,
        "max_cache_len": max_cache_len,
        "prefill_ms": round(prefill_ms, 2),
        "decode_total_ms": round(decode_total_ms, 2),
        "avg_tbt_ms": round(avg_tbt_ms, 2),
        "speed_tps": round(1000 / avg_tbt_ms, 2),
        "mem_mb": round(get_memory_mb(device), 1),
        "token_times_ms": [round(item, 2) for item in token_times_ms],
        "generated_text": tokenizer.decode(generated_tokens, skip_special_tokens=True),
    }


def run_dynamic_kv_once(model, tokenizer, prompt: str, decode_steps: int, device: str) -> dict:
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    prompt_tokens = input_ids.shape[1]
    caches = [None] * NUM_LAYERS
    generated_tokens = []

    sync_device(device)
    prefill_start = time.perf_counter()
    with torch.inference_mode():
        for index in range(prompt_tokens - 1):
            token_id = input_ids[:, index:index + 1]
            position_id = torch.tensor([index], dtype=torch.long, device=device)
            _, caches = model.forward_step(token_id, caches, position_id)
    sync_device(device)
    prefill_ms = (time.perf_counter() - prefill_start) * 1000

    current_token = input_ids[:, -1:]
    current_pos = prompt_tokens - 1
    token_times_ms = []
    decode_start = time.perf_counter()

    with torch.inference_mode():
        for _ in range(decode_steps):
            position_id = torch.tensor([current_pos], dtype=torch.long, device=device)
            step_start = time.perf_counter()
            logits, caches = model.forward_step(current_token, caches, position_id)
            sync_device(device)
            token_times_ms.append((time.perf_counter() - step_start) * 1000)

            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated_tokens.append(next_token.item())
            current_token = next_token
            current_pos += 1

    sync_device(device)
    decode_total_ms = (time.perf_counter() - decode_start) * 1000
    avg_tbt_ms = sum(token_times_ms) / len(token_times_ms)
    return {
        "kv_cache_impl": "dynamic_cat",
        "decode_steps": decode_steps,
        "prompt_tokens": prompt_tokens,
        "max_cache_len": None,
        "prefill_ms": round(prefill_ms, 2),
        "decode_total_ms": round(decode_total_ms, 2),
        "avg_tbt_ms": round(avg_tbt_ms, 2),
        "speed_tps": round(1000 / avg_tbt_ms, 2),
        "mem_mb": round(get_memory_mb(device), 1),
        "token_times_ms": [round(item, 2) for item in token_times_ms],
        "generated_text": tokenizer.decode(generated_tokens, skip_special_tokens=True),
    }


def print_summary(rows: list[dict]) -> None:
    print("\n=== MPS KV Cache / torch.compile 对比 ===")
    print(f"{'impl':<20} {'compile':<9} {'decode':>6} {'prefill(ms)':>12} "
          f"{'decode(ms)':>11} {'tbt(ms)':>9} {'tok/s':>8} {'mem(MB)':>9}")
    print("-" * 96)
    for row in rows:
        if "error" in row:
            print(f"{row['kv_cache_impl']:<20} {str(row['use_compile']):<9} "
                  f"{row['decode_steps']:>6}  ERROR: {row['error']}")
            continue
        print(f"{row['kv_cache_impl']:<20} {str(row['use_compile']):<9} "
              f"{row['decode_steps']:>6} {row['prefill_ms']:>12.2f} "
              f"{row['decode_total_ms']:>11.2f} {row['avg_tbt_ms']:>9.2f} "
              f"{row['speed_tps']:>8.2f} {row['mem_mb']:>9.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MPS 开启 KV Cache 时 torch.compile/eager decode benchmark")
    parser.add_argument("--model-dir", default="/Users/wangrui/localinfer/Qwen3-0.6B")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--prompt", default="人工智能的未来是")
    parser.add_argument("--decode-lengths", type=int, nargs="+", default=[10, 20, 50])
    parser.add_argument("--max-cache-len", type=int, default=256)
    parser.add_argument("--output-json", default="mps_static_kv_compile_benchmark_float16.json")
    parser.add_argument("--dynamic-compile", action="store_true", help="为 torch.compile 启用 dynamic=True")
    parser.add_argument("--skip-compile", action="store_true", help="只测试 eager 模式")
    parser.add_argument("--include-dynamic-kv", action="store_true", help="附加测试旧的 torch.cat 动态 KV cache eager 路径")
    args = parser.parse_args()

    device = "mps"
    if not torch.backends.mps.is_available():
        raise RuntimeError("当前 PyTorch 不支持 MPS: 请检查 torch.backends.mps.is_built()/is_available()")

    dtype = dtype_from_name(args.dtype)
    if dtype == torch.bfloat16:
        print("MPS 对 bfloat16 支持有限，本脚本自动使用 float16")
        dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    prompt_tokens = len(tokenizer.encode(args.prompt))
    required_len = prompt_tokens + max(args.decode_lengths)
    if required_len > args.max_cache_len:
        raise ValueError(f"prompt_tokens + max(decode_lengths) = {required_len}, 超过 max_cache_len={args.max_cache_len}")

    print(f"PyTorch={torch.__version__}  device={device}  dtype={dtype}")
    print(f"prompt_tokens={prompt_tokens}  max_cache_len={args.max_cache_len}")
    print("加载 tokenizer 与模型权重（仅一次）...")
    model = Qwen3Model()
    model = load_weights(model, os.path.join(args.model_dir, "model.safetensors"), device, dtype)
    model.eval()

    rows = []

    if args.include_dynamic_kv:
        print("\n配置: dynamic torch.cat KV cache, compile=False")
        for decode_steps in args.decode_lengths:
            print(f"  decode_steps={decode_steps}")
            row = run_dynamic_kv_once(model, tokenizer, args.prompt, decode_steps, device)
            row["use_compile"] = False
            rows.append(row)

    compile_options = [False] if args.skip_compile else [False, True]
    for use_compile in compile_options:
        runner, compile_create_s = make_static_forward(
            model,
            args.max_cache_len,
            device,
            use_compile,
            args.dynamic_compile,
        )
        if compile_create_s:
            print(f"  compile wrapper 创建耗时: {compile_create_s:.3f}s")

        if use_compile:
            print("\n配置: static preallocated KV cache, compile=True")
            warmup_start = time.perf_counter()
            try:
                run_static_kv_once(runner, tokenizer, args.prompt, 2, args.max_cache_len, dtype, device)
                print(f"  编译 warmup 耗时: {time.perf_counter() - warmup_start:.2f}s")
            except Exception as exc:
                print(f"  编译 warmup 失败: {exc}")
                for decode_steps in args.decode_lengths:
                    rows.append({
                        "kv_cache_impl": "static_preallocated",
                        "use_compile": True,
                        "decode_steps": decode_steps,
                        "error": repr(exc),
                    })
                continue
        else:
            print("\n配置: static preallocated KV cache, compile=False")

        for decode_steps in args.decode_lengths:
            print(f"  decode_steps={decode_steps}")
            try:
                row = run_static_kv_once(runner, tokenizer, args.prompt, decode_steps, args.max_cache_len, dtype, device)
                row["use_compile"] = use_compile
                rows.append(row)
            except Exception as exc:
                print(f"  失败: {exc}")
                rows.append({
                    "kv_cache_impl": "static_preallocated",
                    "use_compile": use_compile,
                    "decode_steps": decode_steps,
                    "error": repr(exc),
                })

    print_summary(rows)

    output_path = Path(args.output_json)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    main()
