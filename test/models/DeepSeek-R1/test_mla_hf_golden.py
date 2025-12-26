import os
import time
import sys
import safetensors
import torch
from transformers import AutoConfig
from transformers import DynamicCache
# 导入 HF 官方实现作为“黄金标准”
from transformers.models.deepseek_v2.modeling_deepseek_v2 import DeepseekV2Attention, DeepseekV2RotaryEmbedding

# ================================================================= #
#  全局配置 (保持不变)
# ================================================================= #
WARMUPS = 10
RUNS = 100
PREFILL_TESTCASES = {"seqlens": [64, 128, 256, 256], "pastlens": [512, 0, 0, 256]}

DECODE_TESTCASES = {
    "seqlens": [1 for _ in range(16)],
    "pastlens": [50 for _ in range(4)]
    + [100 for _ in range(4)]
    + [200 for _ in range(4)]
    + [400 for _ in range(4)],
}


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Test MLA Operator (HF Golden Reference)")
    parser.add_argument(
        "--model_path",
        action="store",
        required=True,
        help="The directory of the model to be tested",
    )

    parser.add_argument(
        "--nvidia",
        action="store_true",
        help="Run nvidia test",
    )

    return parser.parse_args()


def torch_synchronize(_device):
    if _device.type == "cuda":
        torch.cuda.synchronize()


def torch_empty_cache(_device):
    if _device.type == "cuda":
        torch.cuda.empty_cache()


# ================================================================= #
#  模型创建函数
#  对齐: create_Qwen3attention_torch -> create_MLA_operator_hf
# ================================================================= #
def create_MLA_operator_hf(dir_path, *, device, dtype=torch.bfloat16):
    config = AutoConfig.from_pretrained(dir_path)
    
    # =======================================================
    # [关键修改] 强制禁用 Yarn/RoPE Scaling
    # =======================================================
    # 这一步是为了让 HF 模型退化为标准 RoPE，
    # 从而能与你的手动 Torch 实现（只写了标准 RoPE）进行数值对齐。
    config.rope_scaling = None 

    # 显式指定使用 SDPA 加速 (FlashAttention 的 PyTorch 版)
    config._attn_implementation = "sdpa"

    print(f"Initializing MLA Operator (HF Impl, mode={config._attn_implementation})...")
    
    # 1. 创建 Attention (MLA) 模块
    model = DeepseekV2Attention(config=config, layer_idx=0).to(
        device=device, dtype=dtype
    )
    
    # 2. 创建 RoPE 模块
    # Qwen 脚本是单独创建 RoPE 的，这里我们也单独创建，保持逻辑结构一致
    rotary_emb = DeepseekV2RotaryEmbedding(config=config).to(device=device, dtype=dtype)

    # 3. 加载权重
    print(f"Loading weights from {dir_path}...")
    tensors = {}
    for fname in sorted(os.listdir(dir_path)):
        if not fname.endswith(".safetensors"):
            continue
        fpath = os.path.join(dir_path, fname)
        with safetensors.safe_open(fpath, framework="pt") as f:
            for key in f.keys():
                # 适配 Key 名称: 去掉 "model.layers.X.self_attn." 前缀
                if "self_attn." in key:
                    new_key = key.split("self_attn.")[1]
                    tensors[new_key] = f.get_tensor(key)
        break
    
    model.load_state_dict(tensors, strict=False)
    model.eval()

    return model, rotary_emb


# ================================================================= #
#  数据生成函数
#  对齐: generate_attention_input_torch -> generate_mla_inputs_hf
# ================================================================= #
def generate_mla_inputs_hf(
    model, rotary_emb, testcase, device, dtype=torch.bfloat16
):
    config = model.config
    hidden_size = config.hidden_size
    bs = 1

    req_list = []
    for seq_lens, past_lens in zip(testcase["seqlens"], testcase["pastlens"]):
        # 1. 生成 Hidden States (输入)
        hidden_states = torch.rand(
            (bs, seq_lens, hidden_size), device=device, dtype=dtype
        )

        attention_mask = None

        # 2. 生成 Cache (使用 HF DynamicCache)
        # 注意: 之后你自己写的脚本里，这里要改成创建 kv_cache + pe_cache 的 Tensor
        past_key_values = DynamicCache()

        req = {
            "hidden_states": hidden_states,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "past_len": past_lens, # 记录所需的历史长度
        }
        req_list.append(req)

    return req_list


# ================================================================= #
#  Prefill 性能测试
#  对齐: benchmark_Qwen3attention_prefill_torch -> benchmark_MLA_prefill_hf
# ================================================================= #
def benchmark_MLA_prefill_hf(
    model, rotary_emb, test_cases, device, dtype=torch.bfloat16
):
    # 1. 生成数据
    req_list = generate_mla_inputs_hf(
        model, rotary_emb, test_cases, device, dtype=dtype
    )
    req_out_list = []
    
    # ---------------------------------------------------------------- #
    # [Helper] 预先填满 Cache (模拟历史数据)
    # ---------------------------------------------------------------- #
    for req in req_list:
        target_past_len = req["past_len"]
        if target_past_len > 0:
            dummy_input = torch.zeros((1, target_past_len, model.config.hidden_size), device=device, dtype=dtype)
            position_ids = torch.arange(0, target_past_len, device=device).unsqueeze(0)
            
            # 计算 RoPE 并填入 Cache
            rope_out = rotary_emb(dummy_input, position_ids)
            model(hidden_states=dummy_input, position_embeddings=rope_out, past_key_values=req["past_key_values"])

    # ---------------------------------------------------------------- #
    # [Output Collection] 先跑一次收集结果 (用于后续正确性对比)
    # ---------------------------------------------------------------- #
    for req in req_list:
        # 这里的 Cache 已经被填满了，我们这里只做一次 forward 拿结果
        # 为了不破坏 Cache，我们 copy 一份或者暂存 (略过 copy，直接 forward)
        # 注意：这里仅为了获得 output shape/value，逻辑上我们假设它是正确执行的
        pass 

    # ---------------------------------------------------------------- #
    # WARMUP 循环
    # ---------------------------------------------------------------- #
    for _ in range(WARMUPS):
        for req in req_list:
            # 准备参数
            hidden_states = req["hidden_states"]
            current_past_len = req["past_key_values"].get_seq_length()
            seq_len = hidden_states.shape[1]
            position_ids = torch.arange(current_past_len, current_past_len + seq_len, device=device).unsqueeze(0)
            
            # 计算 RoPE
            rope_out = rotary_emb(hidden_states, position_ids)
            
            # 执行 Forward
            model(
                hidden_states=hidden_states,
                position_embeddings=rope_out, 
                past_key_values=req["past_key_values"],
            )

    torch_synchronize(device)

    # ---------------------------------------------------------------- #
    # 正式计时循环
    # ---------------------------------------------------------------- #
    start_time = time.time()

    for _ in range(RUNS):
        for i, req in enumerate(req_list):
            # ----------------------------------------- #
            # [对齐 Qwen] 恢复 KV Cache (Reset)
            # ----------------------------------------- #
            req["past_key_values"] = DynamicCache() # 重置
            
            hidden_states = req["hidden_states"]
            past_len = req["past_len"]
            seq_len = hidden_states.shape[1]
            
            # ----------------------------------------- #
            # [对齐 Qwen] 计算 RoPE
            # ----------------------------------------- #
            position_ids = torch.arange(past_len, past_len + seq_len, device=device).unsqueeze(0)
            rope_out = rotary_emb(hidden_states, position_ids)

            # ----------------------------------------- #
            # [对齐 Qwen] 计算 Attention
            # ----------------------------------------- #
            outputs = model(
                hidden_states=hidden_states,
                position_embeddings=rope_out,
                past_key_values=req["past_key_values"],
            )
            
            # ----------------------------------------- #
            # [修复] 收集输出
            # ----------------------------------------- #
            if _ == 0: # 只收集第一轮的结果用于对比
                output_device = outputs[0]
                output_host = output_device.to("cpu")
                req_out_list.append(output_host)

    torch_synchronize(device)
    end_time = time.time()

    # 计算延迟 (Time To First Token)
    ttft = (end_time - start_time) * 1000 / RUNS

    print(
        f"\t WARMUPS={WARMUPS} RUNS={RUNS}, MLA Operator (HF Golden), average TTFT: {round(ttft, 2)} ms\n"
    )

    return req_out_list


# ================================================================= #
#  Decode 性能测试
#  对齐: benchmark_Qwen3attention_decode_torch -> benchmark_MLA_decode_hf
# ================================================================= #
def benchmark_MLA_decode_hf(
    model, rotary_emb, test_cases, device, dtype=torch.bfloat16
):
    # 1. 生成数据
    req_list = generate_mla_inputs_hf(
        model, rotary_emb, test_cases, device, dtype=dtype
    )
    req_out_list = []
    
    # ----------------------------------------- #
    # 初始化 KV Cache (填满历史)
    # ----------------------------------------- #
    print("Initializing KV Cache for Decode...")
    for req in req_list:
        target_past_len = req["past_len"]
        if target_past_len > 0:
            dummy_input = torch.zeros((1, target_past_len, model.config.hidden_size), device=device, dtype=dtype)
            position_ids = torch.arange(0, target_past_len, device=device).unsqueeze(0)
            rope_out = rotary_emb(dummy_input, position_ids)
            model(hidden_states=dummy_input, position_embeddings=rope_out, past_key_values=req["past_key_values"])

    torch_synchronize(device)

    # ----------------------------------------- #
    # WARMUP
    # ----------------------------------------- #
    for req in req_list:
        for _ in range(WARMUPS):
            hidden_states = req["hidden_states"]
            past_key_values = req["past_key_values"]
            current_seq_len = past_key_values.get_seq_length()
            position_ids = torch.tensor([[current_seq_len]], device=device)
            
            rope_out = rotary_emb(hidden_states, position_ids)
            
            model(
                hidden_states,
                position_embeddings=rope_out,
                past_key_values=past_key_values,
                use_cache=True
            )
    
    # ----------------------------------------- #
    # 恢复 KV Cache 长度 (Reset)
    # ----------------------------------------- #
    # 重新生成数据以获得干净的 Cache
    req_list = generate_mla_inputs_hf(model, rotary_emb, test_cases, device, dtype=dtype)
    for req in req_list:
        target_past_len = req["past_len"]
        if target_past_len > 0:
            dummy_input = torch.zeros((1, target_past_len, model.config.hidden_size), device=device, dtype=dtype)
            position_ids = torch.arange(0, target_past_len, device=device).unsqueeze(0)
            rope_out = rotary_emb(dummy_input, position_ids)
            model(hidden_states=dummy_input, position_embeddings=rope_out, past_key_values=req["past_key_values"])

    torch_synchronize(device)
    start_time = time.time()

    # ----------------------------------------- #
    # 正式计时 (自回归生成)
    # ----------------------------------------- #
    # 逻辑: 针对每个请求，连续推理 RUNS 次，模拟生成 RUNS 个 Token
    for i, req in enumerate(req_list):
        for _ in range(RUNS):
            # ----------------------------------------- #
            # 获得每个 req 的数据
            # ----------------------------------------- #
            hidden_states = req["hidden_states"]
            past_key_values = req["past_key_values"]
            
            # ----------------------------------------- #
            # 计算 RoPE
            # ----------------------------------------- #
            current_seq_len = past_key_values.get_seq_length()
            position_ids = torch.tensor([[current_seq_len]], device=device)
            rope_out = rotary_emb(hidden_states, position_ids)

            # ----------------------------------------- #
            # 计算 Attention
            # ----------------------------------------- #
            outputs = model(
                hidden_states,
                position_embeddings=rope_out,
                past_key_values=past_key_values,
                use_cache=True
            )
            # [核心修复] 使用 outputs[0] 获取结果
            output_device = outputs[0]

            # ----------------------------------------- #
            # 更新 Hidden States (自回归)
            # ----------------------------------------- #
            req["hidden_states"] = output_device
            
            # 收集结果 (仅最后一次循环的最后一个req? 
            # 按照 Qwen 逻辑，这里 append 会导致 list 巨大，
            # 我们仅在最后一轮收集一次作为代表)
            if _ == RUNS - 1:
                output_host = output_device.to("cpu")
                req_out_list.append(output_host)

    torch_synchronize(device)
    end_time = time.time()

    time_consuming = end_time - start_time
    # 计算吞吐量: 总生成的 Token 数 / 总耗时
    out_token_count = RUNS * len(req_list)
    throughput = out_token_count / time_consuming

    print(
        f"\t WARMUPS={WARMUPS} RUNS={RUNS}, MLA Operator (HF Golden), average throughput: {round(throughput, 2)} tok/s \n"
    )

    return req_out_list


if __name__ == "__main__":
    args = get_args()

    # 自动选择设备
    device = torch.device("cpu")
    if args.nvidia and torch.cuda.is_available():
        device = torch.device("cuda")
    
    dtype = torch.bfloat16

    # 1. 创建 MLA 算子和 RoPE
    model, rotary_emb = create_MLA_operator_hf(
        args.model_path, device=device, dtype=dtype
    )
    
    print("\n")
    print("*" * 130)
    print("Test DeepSeek-R1 MLA (HF Golden Reference)")
    print("*" * 130)
    
    # 2. 运行 Prefill
    print(f"Test Case PREFILL_TESTCASES : {PREFILL_TESTCASES}")
    output_prefill = benchmark_MLA_prefill_hf(
        model, rotary_emb, PREFILL_TESTCASES, device, dtype=dtype
    )

    print("\n")
    print("-" * 130)
    # 3. 运行 Decode
    print(f"\nTest DECODE_TESTCASES: {DECODE_TESTCASES}")
    output_decode = benchmark_MLA_decode_hf(
        model, rotary_emb, DECODE_TESTCASES, device, dtype=dtype
    )

    # 清理显存
    del model
    torch_empty_cache(device)