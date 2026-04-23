import os
try:
    from safetensors.torch import safe_open  # 与项目其他脚本保持一致，返回 PyTorch tensor
except ImportError:
    print("Please install safetensors and torch first: pip install safetensors torch")
    exit(1)

def main():
    model_dir = "/Users/wangrui/localinfer/Qwen3.5-0.8B"
    safetensor_path = os.path.join(model_dir, "model.safetensors-00001-of-00001.safetensors")
    
    print(f"Loading weights from: {safetensor_path}")
    print("-" * 50)
    
    # 使用 safe_open 可以实现按需加载（基于 mmap），不会把 1.7GB 整个吃进内存
    with safe_open(safetensor_path, framework="pt", device="cpu") as f:
        # 定义我们要找的三个矩阵的 Key
        up_proj_key = "model.layers.0.mlp.up_proj.weight"
        gate_proj_key = "model.layers.0.mlp.gate_proj.weight"
        down_proj_key = "model.layers.0.mlp.down_proj.weight"
        
        keys = f.keys()
        
        if up_proj_key in keys:
            # 实际读取 Tensor 数据
            up_proj = f.get_tensor(up_proj_key)
            gate_proj = f.get_tensor(gate_proj_key)
            down_proj = f.get_tensor(down_proj_key)
            
            print(f"✅ 成功提取 Layer 0 的 MLP 权重！\n")
            
            print(f"1. up_proj 矩阵:   {up_proj_key}")
            print(f"   Shape: {list(up_proj.shape)}  | 数据类型: {up_proj.dtype}")
            print(f"   理论预期: [3584, 1024] (即 intermediate_size, hidden_size)\n")
            
            print(f"2. gate_proj 矩阵: {gate_proj_key}")
            print(f"   Shape: {list(gate_proj.shape)}  | 数据类型: {gate_proj.dtype}")
            print(f"   理论预期: [3584, 1024] (即 intermediate_size, hidden_size)\n")
            
            print(f"3. down_proj 矩阵: {down_proj_key}")
            print(f"   Shape: {list(down_proj.shape)}  | 数据类型: {down_proj.dtype}")
            print(f"   理论预期: [1024, 3584] (即 hidden_size, intermediate_size)\n")
            
            # 打印部分真实数据看看
            print("-" * 50)
            print("下面是 up_proj 矩阵前 5 个元素的值 (感受一下 bfloat16 数据):")
            print(up_proj.flatten()[:5])
            
        else:
            print(f"未找到 {up_proj_key}，可能是因为由于混合架构，第一层没有使用这个命名方式。")
            print("我们来看看 Layer 0 都有哪些 MLP 相关的 Key：")
            layer0_mlp_keys = [k for k in keys if "layers.0" in k]
            for k in layer0_mlp_keys:
                print("  ->", k)

if __name__ == "__main__":
    main()
