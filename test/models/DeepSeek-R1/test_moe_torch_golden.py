import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import safetensors
import os
import sys
from transformers import AutoConfig

# ================================================================= #
#  手动实现 DeepSeek V3 MoE 逻辑 (复刻官方，修复 Bug)
# ================================================================= #

class DeepseekV3MLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size or config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        
        # SwiGLU: Gate + Up projection
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class DeepseekV3MoE(nn.Module):
    """
    手动实现的 MoE 模块，逻辑与 DeepSeek 官方论文一致：
    Output = Shared_Expert(x) + Sum(Routed_Experts(x) * Routing_Weights)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        
        # 1. Shared Experts (一直激活)
        if self.n_shared_experts > 0:
            self.shared_experts = DeepseekV3MLP(
                config=config, 
                intermediate_size=config.moe_intermediate_size * self.n_shared_experts
            )
        
        # 2. Routed Experts (按需激活)
        # 使用 ModuleList 方便加载权重
        self.experts = nn.ModuleList([
            DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(self.n_routed_experts)
        ])
        
        # 3. Router (Gate)
        self.gate = nn.Linear(self.hidden_size, self.n_routed_experts, bias=False)

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        
        # [Step 1] 展平输入: [B, S, H] -> [Total_Tokens, H]
        hidden_states = hidden_states.view(-1, self.hidden_size)
        
        # [Step 2] 计算 Shared Experts 输出
        final_hidden_states = torch.zeros_like(hidden_states)
        if self.n_shared_experts > 0:
            final_hidden_states = final_hidden_states + self.shared_experts(hidden_states)
            
        # [Step 3] 计算 Router Logits
        router_logits = self.gate(hidden_states) # [Total_Tokens, n_routed_experts]
        
        # [Step 4] 选出 TopK Experts
        # DeepSeek 使用 sigmoid 后选 TopK
        routing_weights = router_logits.sigmoid()
        topk_weights, topk_indices = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        
        # [Step 5] 归一化权重 (Standard DeepSeek Logic)
        # 注意: config.norm_topk_prob 默认为 True
        if getattr(self.config, "norm_topk_prob", True):
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)
            
        # 应用 routed_scaling_factor
        if self.routed_scaling_factor is not None:
            topk_weights = topk_weights * self.routed_scaling_factor

        # [Step 6] 计算 Routed Experts (Python 循环参考实现)
        # 为了保证精度对比的绝对正确性，我们使用最直观的 mask 或循环累加
        # 这里使用"按 Expert 聚合"的方式，这是 PyTorch 上比较高效且准确的写法
        
        # 展平 index 和 weight
        # topk_indices: [Total_Tokens, K]
        # topk_weights: [Total_Tokens, K]
        
        # 初始化 routed output
        routed_output = torch.zeros_like(hidden_states)
        
        # 为了避免极其缓慢的逐 Token 循环，我们遍历 Expert
        # 这种方式数学上等价，且能在 GPU 上并行
        for expert_idx in range(self.n_routed_experts):
            # 找出哪些 Token 选中了当前 Expert
            # mask shape: [Total_Tokens, K]
            mask = (topk_indices == expert_idx)
            
            # 这一步: [Total_Tokens] (是否有任意一个 TopK 选中了该 Expert)
            batch_mask = mask.any(dim=-1)
            
            if batch_mask.any():
                # 选出需要计算的 tokens
                # input_chunk: [Num_Selected, H]
                input_chunk = hidden_states[batch_mask]
                
                # Expert 计算
                expert_out = self.experts[expert_idx](input_chunk)
                
                # 找出对应的权重
                # mask: [Total_Tokens, K], topk_weights: [Total_Tokens, K]
                # 我们需要把 weight 广播并选取出来
                # 这种写法稍微有点 trick，但为了准确性：
                
                # 获取这批 token 在 topk 中的位置索引
                # 我们知道 batch_mask 对应的行里，必然有一个 True
                # 我们需要那个 True 对应的 weight
                
                # 简化逻辑：对于被选中的 Token，把结果加回去
                # 先把 expert_out 映射回 full batch 大小
                expanded_expert_out = torch.zeros_like(hidden_states)
                expanded_expert_out[batch_mask] = expert_out
                
                # 计算加权
                # 这里的 weight 比较难取，因为一个 Token 可能多次选中同一个 Expert (理论上 TopK 不会，但逻辑上要严谨)
                # 实际上 TopK 返回的 indices 不会重复。
                
                # 创建一个全零的 weight map
                weight_map = torch.zeros((hidden_states.size(0),), dtype=hidden_states.dtype, device=hidden_states.device)
                
                # 填充 weight: 只填充 batch_mask 为 True 的位置
                # mask[batch_mask] 选出了所有 True 的行
                # topk_weights[batch_mask] 选出了对应的权重行
                # 我们需要 mask 对应的那个位置的 weight
                
                # 利用 sum 提取 weight (因为每行只有一个 True)
                relevant_weights = (topk_weights * mask.float()).sum(dim=-1)
                
                # 累加: Output += Expert(x) * Weight
                routed_output += expanded_expert_out * relevant_weights.unsqueeze(-1)

        final_hidden_states = final_hidden_states + routed_output
        
        # 恢复形状
        return final_hidden_states.view(orig_shape)


# ================================================================= #
#  测试配置 (严格对齐 qwen3_moe)
# ================================================================= #

WARMUPS = 10
RUNS = 100

PREFILL_TESTCASES = {
    "seqlens": [64, 128, 256, 256], 
    "total_tokens": 704
}

DECODE_TESTCASES = {
    "seqlens": [1 for _ in range(16)],
    "total_tokens": 16
}

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Test DeepSeek MoE Operator")
    parser.add_argument("--model_path", action="store", help="model path")
    parser.add_argument("--cpu", action="store_true", help="Run cpu test")
    parser.add_argument("--nvidia", action="store_true", help="Run nvidia test")
    parser.add_argument("--metax", action="store_true", help="Run metax test")
    parser.add_argument("--moore", action="store_true", help="Run moore test")
    parser.add_argument("--iluvatar", action="store_true", help="Run iluvatar test")
    return parser.parse_args()


def torch_synchronize(_device):
    if _device == "cuda":
        torch.cuda.synchronize()
    elif _device == "musa":
        torch.musa.synchronize()


def torch_empty_cache(_device):
    if _device == "cuda":
        torch.cuda.empty_cache()
    elif _device == "musa":
        torch.musa.empty_cache()


def create_moe_torch(dir_path, device, dtype=torch.bfloat16):
    config = AutoConfig.from_pretrained(dir_path)
    
    # 强制补全属性
    if not hasattr(config, "mlp_bias"): config.mlp_bias = False
    
    print(f"Initializing Manual DeepseekV3MoE (Golden Reference)...")
    # 使用我们要手动实现的类
    moe = DeepseekV3MoE(config).to(device=device, dtype=dtype)
    
    tensors = {}
    print(f"Loading weights from {dir_path}...")
    for fname in sorted(os.listdir(dir_path)):
        if not fname.endswith(".safetensors"): continue
        fpath = os.path.join(dir_path, fname)
        with safetensors.safe_open(fpath, framework="pt") as f:
            for key in f.keys():
                # 映射规则:
                # 1. model.layers.X.mlp.gate.weight -> gate.weight
                # 2. model.layers.X.mlp.experts.0.gate_proj.weight -> experts.0.gate_proj.weight
                # 3. model.layers.X.mlp.shared_experts.gate_proj.weight -> shared_experts.gate_proj.weight
                
                if ".mlp." in key:
                    new_key = key.split(".mlp.")[1]
                    tensors[new_key] = f.get_tensor(key)
    
    if len(tensors) == 0:
        print("❌ Error: 'gate.weight' not found!")
        sys.exit(1)
    
    # 手动加载权重到我们的 Module 中
    # 因为我们的类结构和 HF 原版是一一对应的，所以 key 应该是匹配的
    missing, unexpected = moe.load_state_dict(tensors, strict=False)
    
    print(f"Weights loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    # 预期可能会有一些 irrelevant keys (比如 rms_norm 相关的)，只要 experts 和 gate 在就行
    
    moe.eval()
    return moe


def generate_moe_input_torch(testcase, model_config, dtype=torch.bfloat16):
    # 构造 (1, Total_Tokens, Hidden) 格式输入
    # 这是最通用的格式，既满足 Gate 的维度需求，也能模拟 Batch
    total_tokens = testcase["total_tokens"]
    hidden_size = model_config.hidden_size
    input_tensor = torch.rand((1, total_tokens, hidden_size), device="cpu", dtype=dtype)
    return input_tensor


def benchmark_moe_torch(moe, testcase, device, dtype):
    input_host = generate_moe_input_torch(testcase, moe.config, dtype=dtype)
    input_device = input_host.to(device=device)

    # 第一次运行，用于 Check
    output_device = moe(input_device)
    output_host = output_device.to("cpu")

    # Warmup
    for _ in range(WARMUPS):
        moe(input_device)
    torch_synchronize(device)

    # Timing
    start_time = time.time()
    for _ in range(RUNS):
        moe(input_device)
    torch_synchronize(device)
    end_time = time.time()

    total_time = end_time - start_time
    total_tokens = testcase["total_tokens"] * RUNS
    
    print(
        f"\t WARMUPS={WARMUPS} RUNS={RUNS}, MoE Torch average latency: {round(total_time * 1000 / RUNS, 2)} ms   throughput: {round(total_tokens / total_time, 2)} tok/s"
    )
    return output_host


if __name__ == "__main__":
    args = get_args()
    
    device = "cpu"
    if args.nvidia:
        device = "cuda"
    elif args.metax:
        device = "cuda"
    # ... 其他平台省略，保持代码简洁 ...

    dtype = torch.bfloat16

    moe = create_moe_torch(args.model_path, device=device, dtype=dtype)

    print("*" * 130)
    print("Test DeepSeek MoE (Manual Golden Reference)")
    print("*" * 130)
    
    print(f"Test Case PREFILL_TESTCASES : {PREFILL_TESTCASES}")
    output_prefill = benchmark_moe_torch(
        moe, PREFILL_TESTCASES, device=device, dtype=dtype
    )

    print("\n")
    print("-" * 130)
    print(f"\nTest DECODE_TESTCASES: {DECODE_TESTCASES}")
    output_decode = benchmark_moe_torch(
        moe, DECODE_TESTCASES, device=device, dtype=dtype
    )

    del moe
    torch_empty_cache(device)