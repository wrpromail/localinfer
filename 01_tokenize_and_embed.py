import os
import torch
from transformers import AutoTokenizer
from safetensors.torch import safe_open

# 1. 加载分词器与输入文本
model_dir = "/Users/wangrui/localinfer/Qwen3.5-0.8B"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

text = "你好，世界"
input_ids = tokenizer(text, return_tensors="pt")["input_ids"]

print(f"Token IDs: {input_ids}")
print(f"Token 形状: {list(input_ids.shape)}")

# 2. 从 safetensors 加载 Embedding 权重
safetensor_path = os.path.join(model_dir, "model.safetensors-00001-of-00001.safetensors")
with safe_open(safetensor_path, framework="pt", device="cpu") as f:
    embed_weight = f.get_tensor("model.language_model.embed_tokens.weight")

# 3. 手工执行词嵌入 (Embedding) 查表
input_embeddings = embed_weight[input_ids]

print(f"Embedding 输出形状: {list(input_embeddings.shape)}")
print(f"第一个 Token 的前5个特征: {input_embeddings[0, 0, :5]}")
