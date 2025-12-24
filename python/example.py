from transformers import AutoTokenizer
import argparse

from icinfer import SamplingParams
from icinfer.engine.libinfinicore_infer import DeviceType
from icinfer.engine.llm_engine import InfiniEngine


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--device-type", type=str, default="nvidia")
    parser.add_argument("--ndev", type=int, default=1)
    parser.add_argument("--max-kvcache-tokens", type=int, default=10240)
    parser.add_argument("--enable-paged-attn", action="store_true")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    model_path = args.model_path
    if model_path == None:
        raise ValueError("Error: --model-path is required.")
    max_kvcache_tokens = args.max_kvcache_tokens
    device_type = DeviceType.DEVICE_TYPE_CPU
    if args.device_type == "cpu":
        device_type = DeviceType.DEVICE_TYPE_CPU
    elif args.device_type == "nvidia":
        device_type = DeviceType.DEVICE_TYPE_NVIDIA
    elif args.device_type == "cambricon":
        device_type = DeviceType.DEVICE_TYPE_CAMBRICON
    elif args.device_type == "ascend":
        device_type = DeviceType.DEVICE_TYPE_ASCEND
    elif args.device_type == "metax":
        device_type = DeviceType.DEVICE_TYPE_METAX
    elif args.device_type == "moore":
        device_type = DeviceType.DEVICE_TYPE_MOORE
    elif args.device_type == "iluvatar":
        device_type = DeviceType.DEVICE_TYPE_ILUVATAR
    elif args.device_type == "qy":
        device_type = DeviceType.DEVICE_TYPE_QY
    else:
        raise ValueError("Error: --device_type is required.")

    if not args.enable_paged_attn:
        raise ValueError(
            "This script currently only supports paged attention. "
            "Please enable it with --enable-paged-attn."
        )

    path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    llm = InfiniEngine(
        path,
        device=device_type,
        ndev=args.ndev,
        enforce_eager=True, 
        tensor_parallel_size=args.ndev,
        trust_remote_code=True, 
        attention_bias=True,
        enable_paged_attn=args.enable_paged_attn,
        max_kvcache_tokens=max_kvcache_tokens
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=128
    )

    prompts = [
        "山东最高的山是？",
    ] * 16

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        for prompt in prompts
    ]
    outputs, avg_prefill_throughput, avg_decode_throughput,  avg_ttft, avg_tbt, cache_efficiency = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")
    print(f"batch_size: {len(prompts)}, n_dev: {args.ndev}, is_paged_attn: {args.enable_paged_attn}")
    print(f"Avg Prefill Throughput: {avg_prefill_throughput:.2f} tok/s")
    print(f"Avg Decode Throughput: {avg_decode_throughput:.2f} tok/s")
    print(f"Avg TTFT: {avg_ttft*1000:.2f} ms")
    print(f"Avg TBT: {avg_tbt*1000:.2f} ms")
    print(f"Cache Efficiency: {cache_efficiency*100:.2f}%")

if __name__ == "__main__":
    main()

"""
# 运行 example.py 测试脚本
python path/to/example.py \
    --model-path <path_to_your_model> \
    --device-type <device_type> \
    --ndev <num_devices> \
    --max-kvcache-tokens <kv_cache_limit> \
    --enable-paged-attn
"""
