import time
import torch
import transformers
import safetensors
import os
from transformers import AutoConfig
from transformers.models import qwen3_moe
import sys

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

    parser = argparse.ArgumentParser(description="Test Operator")
    parser.add_argument(
        "--model_path",
        action="store",
        help="The directory of the model to be tested",
    )

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
    moe = qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock(config).to(
        device=device, dtype=dtype
    )
    tensors = {}
    for fname in sorted(os.listdir(dir_path)):
        if not fname.endswith(".safetensors"):
            continue
        fpath = os.path.join(dir_path, fname)
        with safetensors.safe_open(fpath, framework="pt") as f:
            for key in f.keys():
                if "model.layers.0.mlp." in key:
                    tensors[key[len("model.layers.0.mlp.") :]] = f.get_tensor(key)
        break
    moe.load_state_dict(tensors)
    return moe


def generate_moe_input_torch(testcase, dtype=torch.bfloat16):
    total_seqlen = sum(testcase["seqlens"])
    input_tensor = torch.rand((1, total_seqlen, 2048), device="cpu", dtype=dtype)
    return input_tensor


def benchmark_moe_torch(moe, testcase, device, dtype):
    """"""
    input_host = generate_moe_input_torch(testcase, dtype=dtype)
    input_device = input_host.to(device=device)

    output_device, _ = moe(input_device)
    output_host = output_device.to("cpu")

    for _ in range(WARMUPS):
        moe(input_device)
    torch_synchronize(device)

    start_time = time.time()
    for _ in range(RUNS):
        moe(input_device)
    torch_synchronize(device)
    end_time = time.time()

    total_time = end_time - start_time
    total_tokens = sum(testcase["seqlens"]) * RUNS
    print(
        f"\t WARMUPS={WARMUPS} RUNS={RUNS}, MoE Torch average latency: {round(total_time * 1000 / RUNS, 2)} ms   throughput: {round(total_tokens / total_time, 2)} tok/s"
    )
    return output_host


if __name__ == "__main__":
    args = get_args()
    print(args)

    model_path = args.model_path
    dtype = torch.bfloat16
    # Parse command line arguments
    device = "cpu"
    if args.cpu:
        device = "cpu"
    elif args.nvidia:
        device = "cuda"
    elif args.metax:
        device = "cuda"
    elif args.moore:
        device = "musa"
        import torch_musa
    elif args.iluvatar:
        device = "cuda"
    else:
        print(
            "Usage:  python test/models/qwen3_moe/moe_test.py [--cpu | --nvidia | --metax | --moore | --iluvatar] --model_path=<path/to/model_path>"
        )
        sys.exit(1)

    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------
    # -----------------------------------------------------------------------------

    moe = create_moe_torch(model_path, device=device, dtype=dtype)

    print("*" * 130)
    print("Test Qwen3 MoE")
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

    # clean up device memory
    del moe
    torch_empty_cache(device)
