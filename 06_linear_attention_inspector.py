import os
import torch
from safetensors.torch import safe_open

def main():
    print("=== 第六步：硬核拆解 Linear Attention (Mamba变体) 的内部算子 ===")
    print("由于各家大模型对状态空间模型 (SSM) 的实现细节各不相同，")
    print("在盲目写数学公式前，我们必须像法医一样，先解剖它的物理权重形状！\n")
    
    model_dir = "/Users/wangrui/localinfer/Qwen3.5-0.8B"
    safetensor_path = os.path.join(model_dir, "model.safetensors-00001-of-00001.safetensors")
    
    # 我们解剖第 0 层 (这是一个线性层)
    layer_idx = 0
    
    # 通过刚才的正则搜索，我们锁定了这些奇特的权重名
    weights_to_load = [
        "in_proj_qkv.weight",
        "in_proj_z.weight",
        "in_proj_a.weight",
        "in_proj_b.weight",
        "conv1d.weight",
        "norm.weight",
        "A_log",
        "dt_bias",
        "out_proj.weight"
    ]
    
    with safe_open(safetensor_path, framework="pt", device="cpu") as f:
        print(f"--- 核心参数 Shape 探查 (以 Layer {layer_idx} 为例) ---")
        for w_name in weights_to_load:
            full_key = f"model.language_model.layers.{layer_idx}.linear_attn.{w_name}"
            if full_key in f.keys():
                t = f.get_tensor(full_key)
                print(f"👉 {w_name.ljust(20)} | 形状: {str(list(t.shape)).ljust(15)} | Dtype: {t.dtype}")
            else:
                print(f"❌ 未找到: {w_name}")
                
if __name__ == "__main__":
    main()
