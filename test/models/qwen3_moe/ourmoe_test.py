import argparse
import re
import numpy as np

import time
import torch
import os
from transformers import AutoConfig
from transformers.models import qwen3_moe       # 对拍相关
import sys



sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../QIYUAN_GROUP-InfiniCore/python"))

import infinicore
from infinilm.modeling_utils import load_state_dict
from infinilm.models.qwen3moe.qwen3moe import Qwen3MoeSparseMoeBlock
from infinilm.generation.utils import infini_to_numpy             # 对拍
# WARMUPS = 10
# RUNS = 100
WARMUPS = 0
RUNS = 1

PREFILL_TESTCASES = {"seqlens": [64, 128, 256, 256], "pastlens": [512, 0, 0, 256]}
DECODE_TESTCASES = {
    "seqlens": [1 for _ in range(16)],
    "pastlens": [50 for _ in range(4)]
    + [100 for _ in range(4)]
    + [200 for _ in range(4)]
    + [400 for _ in range(4)],
}


def get_args():
    parser = argparse.ArgumentParser(description="Test Qwen3 MoE block with InfiniLM")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run cpu test",
    )
    parser.add_argument(
        "--nvidia",
        action="store_true",
        help="Run nvidia test",
    )
    parser.add_argument(
        "--metax",
        action="store_true",
        help="Run metax test",
    )
    parser.add_argument(
        "--moore",
        action="store_true",
        help="Run moore test",
    )
    parser.add_argument(
        "--iluvatar",
        action="store_true",
        help="Run iluvatar test",
    )
    parser.add_argument(
#================ 对拍 ========================#
        "--check",
        action="store_true",
        help="Compare against a Torch reference implementation",
    )
    parser.add_argument(
        "--check_device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device used for the Torch reference when --check is enabled",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used to generate deterministic inputs",
    )
    parser.add_argument(
#================ 对拍 ========================#
        "--model_path",
        type=str,
        required=True,
        help="Path to the Qwen3-MoE checkpoint directory",
    )
    parser.add_argument(
        "--layer_idx",
        type=int,
        default=0,
        help="Which decoder layer's MoE block to load weights from",
    )
    return parser.parse_args()


def resolve_device(args) -> str:
    if args.cpu:
        return "cpu"
    if args.nvidia:
        return "cuda"
    if args.metax:
        return "cuda"
    if args.moore:
        return "musa"
    if args.iluvatar:
        return "cuda"
    raise ValueError(
        "Usage: python test/models/qwen3_moe/ourmoe_test.py "
        "[--cpu | --nvidia | --metax | --moore | --iluvatar] "
        "--model_path=<path/to/model_path>"
    )


def to_torch_dtype(infini_dtype: infinicore.dtype):
    utils = getattr(infinicore, "utils", None)
    if utils is not None:
        mapper = getattr(utils, "to_torch_dtype", None)
        if callable(mapper):
            return mapper(infini_dtype)
#================ 对拍 ========================#
    if infini_dtype == infinicore.float32:
        return torch.float32
#================ 对拍 ========================#
    return torch.bfloat16


#================ 对拍 ========================#
def load_moe_state_dict_torch(model_path: str, layer_idx: int):
    prefix = f"model.layers.{layer_idx}.mlp."
    tensors = {}
    for fname in sorted(os.listdir(model_path)):
        if not fname.endswith(".safetensors"):
            continue
        checkpoint = load_state_dict(os.path.join(model_path, fname))
        for full_name, tensor in checkpoint.items():
            if full_name.startswith(prefix):
                tensors[full_name[len(prefix) :]] = tensor
    if not tensors:
        raise FileNotFoundError(f"Cannot find MoE weights with prefix '{prefix}' under {model_path}")
    return tensors


def create_moe_torch(config, model_path: str, device_str: str, dtype, layer_idx: int):
    moe = qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock(config).to(device=device_str, dtype=dtype)
    moe.load_state_dict(load_moe_state_dict_torch(model_path, layer_idx), strict=True)
    moe.eval()
    return moe

#================ 对拍 ========================#
def load_moe_weights(model_path: str, device_str: str, dtype, layer_idx: int, config):
    prefix = f"model.layers.{layer_idx}.mlp."
    torch_dtype = to_torch_dtype(dtype)

    gate_weight = None
    expert_parts: dict[int, dict[str, torch.Tensor]] = {}

    for fname in sorted(os.listdir(model_path)):
        if not fname.endswith(".safetensors"):
            continue

        checkpoint = load_state_dict(os.path.join(model_path, fname))
        for full_name, tensor in checkpoint.items():
            if not full_name.startswith(prefix):
                continue

            local_name = full_name[len(prefix) :]
            if local_name == "gate.weight":
                gate_weight = tensor
                continue

            match = re.match(r"experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", local_name)
            if match:
                idx = int(match.group(1))
                expert_parts.setdefault(idx, {})[match.group(2)] = tensor

    if gate_weight is None or not expert_parts:
        raise FileNotFoundError(
            f"Cannot find MoE weights with prefix '{prefix}' under {model_path}"
        )

    num_experts = config.num_experts
    hidden_dim = config.hidden_size
    inter_dim = config.moe_intermediate_size

    gate_up = torch.empty((num_experts, 2 * inter_dim, hidden_dim))
    down = torch.empty((num_experts, hidden_dim, inter_dim))

    for expert_idx in range(num_experts):
        parts = expert_parts.get(expert_idx)
        if parts is None or any(k not in parts for k in ("gate_proj", "up_proj", "down_proj")):
            raise KeyError(f"Missing weights for expert {expert_idx} in {model_path}")

        gate_proj = parts["gate_proj"]
        up_proj = parts["up_proj"]
        down_proj = parts["down_proj"]
        gate_up[expert_idx] = torch.cat([gate_proj, up_proj], dim=0)
        down[expert_idx] = down_proj

    gate_weight = gate_weight.to(device=device_str, dtype=torch_dtype)
    gate_up = gate_up.to(device=device_str, dtype=torch_dtype)
    down = down.to(device=device_str, dtype=torch_dtype)

    return {
        "gate.weight": infinicore.from_torch(gate_weight),
        "experts.gate_up_proj": infinicore.from_torch(gate_up),
        "experts.down_proj": infinicore.from_torch(down),
    }


def create_moe(model_path: str, device: infinicore.device, dtype, layer_idx: int):
    config = AutoConfig.from_pretrained(model_path)
    moe = Qwen3MoeSparseMoeBlock(config, device=device, dtype=dtype)

    moe_state = load_moe_weights(model_path, device.type, dtype, layer_idx, config)
    moe.load_state_dict(moe_state)
    if hasattr(moe, "eval"):
        moe.eval()
    return moe, config


# def generate_moe_input(testcase, hidden_size: int, device, dtype):
def generate_moe_input(testcase, hidden_size: int, device, dtype, *, seed: int):   # 对拍
    total_seqlen = sum(testcase["seqlens"])
    # host = np.random.default_rng().standard_normal(
    #     (1, total_seqlen, hidden_size)
    # ).astype(np.float32)
    # return infinicore.from_numpy(host, dtype=dtype, device=device)
#================ 对拍 ========================#    
    host = np.random.default_rng(seed).standard_normal((1, total_seqlen, hidden_size)).astype(np.float32)
    return host, infinicore.from_numpy(host, dtype=dtype, device=device)
#================ 对拍 ========================#


def _sync_device(device: infinicore.device):
    for name in ("synchronize", "device_synchronize"):
        fn = getattr(infinicore, name, None)
        if callable(fn):
            try:
                fn(device)
            except TypeError:
                fn()
            break


# def benchmark_moe(moe, testcase, hidden_size: int, device, dtype):
#     input_tensor = generate_moe_input(testcase, hidden_size, device, dtype)
#================ 对拍 ========================#    
def benchmark_moe(
    moe,
    testcase,
    hidden_size: int,
    device,
    dtype,
    *,
    seed: int,
    check: bool,
    torch_moe=None,
    torch_device_str: str | None = None,
    torch_dtype=None,
):
    host, input_tensor = generate_moe_input(testcase, hidden_size, device, dtype, seed=seed)
#================ 对拍 ========================#    
    hidden_out, routing_out = moe(input_tensor)

    print(
        f"\tOutput hidden shape: {getattr(hidden_out, 'shape', '?')}, routing shape: {getattr(routing_out, 'shape', '?')}"
    )

#================ 对拍 ========================#     
    if check:
        _sync_device(device)

        hidden_cpu = hidden_out
        if hidden_cpu.device.type != "cpu":
            hidden_cpu = hidden_cpu.to(infinicore.device("cpu", 0))
        if hasattr(hidden_cpu, "is_contiguous") and callable(hidden_cpu.is_contiguous):
            if not hidden_cpu.is_contiguous():
                hidden_cpu = hidden_cpu.contiguous()

        out_inf = infini_to_numpy(hidden_cpu).astype(np.float32, copy=False)

        torch_inp = torch.from_numpy(host).to(device=torch_device_str, dtype=torch_dtype)
        with torch.no_grad():
            out_torch, _ = torch_moe(torch_inp)
        out_torch = out_torch.detach().to("cpu").to(dtype=torch.float32).numpy()

        inf_nan = int(np.isnan(out_inf).sum())
        inf_inf = int(np.isinf(out_inf).sum())
        torch_nan = int(np.isnan(out_torch).sum())
        torch_inf = int(np.isinf(out_torch).sum())

        finite = np.isfinite(out_inf) & np.isfinite(out_torch)
        if finite.any():
            diff = out_torch[finite] - out_inf[finite]
            diff_abs_max = float(np.max(np.abs(diff)))
            diff_abs_mean = float(np.mean(np.abs(diff)))
        else:
            diff_abs_max = float("nan")
            diff_abs_mean = float("nan")

        print(f"\t Output stats (torch) - Sum: {out_torch.sum():.4f}, Mean: {out_torch.mean():.4f}")
        print(f"\t Output stats (infini) - Sum: {out_inf.sum():.4f}, Mean: {out_inf.mean():.4f}")
        print(f"\t NaN/Inf count (torch): {torch_nan}/{torch_inf}   (infini): {inf_nan}/{inf_inf}")
        print(f"\t First 5 values (torch): {out_torch.reshape(-1)[:5].tolist()}")
        print(f"\t First 5 values (infini): {out_inf.reshape(-1)[:5].tolist()}")
        print(f"\t Diff abs max: {diff_abs_max:.6f}, mean: {diff_abs_mean:.6f}")
#================ 对拍 ========================# 

    for _ in range(WARMUPS):
        moe(input_tensor)
    _sync_device(device)

    t0 = time.time()
    for _ in range(RUNS):
        moe(input_tensor)
    _sync_device(device)
    t1 = time.time()

    total_time = t1 - t0
    total_tokens = sum(testcase["seqlens"]) * RUNS
    print(
        f"\tWARMUPS={WARMUPS} RUNS={RUNS}, latency: {round(total_time * 1000 / RUNS, 2)} ms   throughput: {round(total_tokens / total_time, 2)} tok/s"
    )

    return hidden_out


if __name__ == "__main__":
    args = get_args()
    print(args)

    device_str = resolve_device(args)
    if device_str == "musa":
        try:
            import torch_musa  # noqa: F401
        except ImportError:
            print("torch_musa is required for MUSA devices, falling back to CPU")
            device_str = "cpu"

    infini_device = infinicore.device(device_str, 0)
    # infini_dtype = infinicore.bfloat16
    # Switch to float32 to bypass infinicore BF16 conversion issues
    infini_dtype = infinicore.float32

    moe, config = create_moe(
        args.model_path, device=infini_device, dtype=infini_dtype, layer_idx=args.layer_idx
    )
    hidden_size = config.hidden_size

    print("*" * 130)
    print("Test Qwen3 MoE (InfiniLM)")
    print("*" * 130)

    print(f"Test Case PREFILL_TESTCASES : {PREFILL_TESTCASES}")
    # benchmark_moe(moe, PREFILL_TESTCASES, hidden_size, device=infini_device, dtype=infini_dtype)
#================ 对拍 ========================#     
    torch_dtype = to_torch_dtype(infini_dtype)
    torch_moe = None
    torch_check_device = args.check_device
    if args.check:
        torch_moe = create_moe_torch(config, args.model_path, torch_check_device, torch_dtype, args.layer_idx)

    benchmark_moe(
        moe,
        PREFILL_TESTCASES,
        hidden_size,
        device=infini_device,
        dtype=infini_dtype,
        seed=args.seed,
        check=args.check,
        torch_moe=torch_moe,
        torch_device_str=torch_check_device,
        torch_dtype=torch_dtype,
    )
#================ 对拍 ========================# 

    print("\n" + "-" * 130)
    print(f"\nTest DECODE_TESTCASES: {DECODE_TESTCASES}")
    # benchmark_moe(moe, DECODE_TESTCASES, hidden_size, device=infini_device, dtype=infini_dtype)
#================ 对拍 ========================#     
    benchmark_moe(
        moe,
        DECODE_TESTCASES,
        hidden_size,
        device=infini_device,
        dtype=infini_dtype,
        seed=args.seed + 1,
        check=args.check,
        torch_moe=torch_moe,
        torch_device_str=torch_check_device,
        torch_dtype=torch_dtype,
    )
#================ 对拍 ========================#     
