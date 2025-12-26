import torch
import os
import sys
import numpy as np
import safetensors.torch
from transformers import AutoConfig
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Attention, DeepseekV2RotaryEmbedding

# 将当前目录加入路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from test_mla_torch_golden import ManualDeepseekV2Attention
except ImportError:
    print("❌ Error: Could not import ManualDeepseekV2Attention from test_mla_torch_golden.py")
    sys.exit(1)

def calculate_metrics(out_hf, out_manual):
    # 转为 float32 numpy 进行统计
    h = out_hf.to(torch.float32).detach().cpu().numpy().flatten()
    m = out_manual.to(torch.float32).detach().cpu().numpy().flatten()

    # 1. 余弦相似度 (核心指标)
    dot = np.dot(h, m)
    norm_h = np.linalg.norm(h)
    norm_m = np.linalg.norm(m)
    # 防止分母为0
    cos_sim = dot / (norm_h * norm_m + 1e-10)

    # 2. 最大绝对误差 (在原始权重下，这个值会很大，仅供参考)
    diff = np.abs(h - m)
    max_diff = np.max(diff)

    # 3. 相对误差 (过滤掉极小值)
    mask = np.abs(h) > 1e-4
    if np.sum(mask) > 0:
        rel_err = diff[mask] / (np.abs(h[mask]) + 1e-10)
        p99_rel_err = np.percentile(rel_err, 99) # 取99分位，排除异常点
    else:
        p99_rel_err = 0.0

    return cos_sim, max_diff, p99_rel_err

def check_correctness(model_path, device="cuda"):
    print(f"\n{'='*80}")
    print(f"MLA Verification (RAW WEIGHTS - NO NORMALIZATION)")
    print(f"{'='*80}\n")
    
    # 建议先用 float32 跑原始权重，因为 DeepSeek 的数值范围在 BF16 下容易溢出
    # 如果你想测 BF16，就把这里改成 torch.bfloat16
    dtype = torch.bfloat16 
    print(f"Running dtype: {dtype}")
    
    config = AutoConfig.from_pretrained(model_path)
    config._attn_implementation = "eager"
    config.rope_scaling = None 
    
    print(">>> 1. Initializing Models...")
    hf_model = DeepseekV2Attention(config=config, layer_idx=0).to(device=device, dtype=dtype)
    hf_rope = DeepseekV2RotaryEmbedding(config=config).to(device=device, dtype=dtype)
    manual_model = ManualDeepseekV2Attention(config).to(device=device, dtype=dtype)

    # =======================================================
    # [修改点] 加载原始权重，不进行归一化
    # =======================================================
    print(">>> 2. Loading RAW Weights (Warning: Values might be large)...")
    loaded_tensors = {}
    for fname in sorted(os.listdir(model_path)):
        if fname.endswith(".safetensors"):
            fpath = os.path.join(model_path, fname)
            with safetensors.safe_open(fpath, framework="pt") as f:
                for key in f.keys():
                    if "self_attn." in key:
                        sub_key = key.split("self_attn.")[1]
                        raw = f.get_tensor(key)
                        
                        # [重要] 直接加载，不减均值除方差
                        # 仅仅转为目标 dtype
                        loaded_tensors[sub_key] = raw.to(dtype)
    
    hf_model.load_state_dict(loaded_tensors, strict=False)
    manual_model.load_state_dict(loaded_tensors, strict=False)
    hf_model.eval()
    manual_model.eval()

    # 3. 构造输入
    seq_len = 128
    hidden_states = torch.randn(1, seq_len, config.hidden_size, device=device, dtype=dtype)
    # 稍微把输入数值搞小一点，防止乘法后溢出 (DeepSeek 内部会有 RMSNorm，但输入小点安全)
    hidden_states = hidden_states * 0.01 
    
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    print(">>> 3. Running Inference...")
    with torch.no_grad():
        # A. 运行 HF RoPE
        rope_out_hf = hf_rope(hidden_states, position_ids)

        # B. 数据桥接 (Data Bridging)
        if isinstance(rope_out_hf, torch.Tensor):
            r = rope_out_hf.real.float().squeeze()
            i = rope_out_hf.imag.float().squeeze()
            # Interleaved 布局对齐
            cos_man = r.repeat_interleave(2, dim=-1).to(dtype)
            sin_man = i.repeat_interleave(2, dim=-1).to(dtype)
            manual_rope_input = (cos_man, sin_man)
        elif isinstance(rope_out_hf, tuple):
            manual_rope_input = (rope_out_hf[0].squeeze(), rope_out_hf[1].squeeze())
        else:
            raise ValueError(f"Unknown RoPE type: {type(rope_out_hf)}")

        # C. 执行 Forward
        hf_output = hf_model(hidden_states, position_embeddings=rope_out_hf)[0]
        manual_output = manual_model(hidden_states, manual_rope_input)

    # 4. 统计
    print(">>> 4. Calculating Metrics...")
    
    # 打印一些统计值，让你看看现在的数值范围有多大
    print(f"   HF Output Mean: {hf_output.float().mean():.4f}, Max: {hf_output.abs().max():.4f}")
    
    cos_sim, max_diff, p99_err = calculate_metrics(hf_output, manual_output)

    print("\n" + "="*60)
    print("VERIFICATION REPORT (RAW WEIGHTS)")
    print("="*60)
    print(f"Metric                 | Value       | Expectation")
    print(f"-----------------------|-------------|-------------")
    print(f"Cosine Similarity      | {cos_sim:.6f}    | > 0.99")
    print(f"Max Absolute Diff      | {max_diff:.4f}      | (Variable)")
    print("-" * 60)
    
    if cos_sim > 0.99:
        print("\n✅ PASS: Models align well on raw weights.")
    else:
        print("\n❌ FAIL: Alignment lost.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()
    
    check_correctness(args.model_path)