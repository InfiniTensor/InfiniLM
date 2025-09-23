import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import time
import sys
from random import randint, seed
# from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams

from icinfer import LLM, SamplingParams
from icinfer.engine.libinfinicore_infer import DeviceType

import logging
logger = logging.getLogger(__name__)
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="/home/wanghaojie/vllm/huggingface/Llama-2-7b-chat-hf")
    # parser.add_argument("--model-path", type=str, default="/home/wanghaojie/vllm/huggingface/FM9G_70B_SFT_MHA/")
    parser.add_argument("--model-path", type=str, default="/home/wanghaojie/vllm/huggingface/9G7B_MHA/")
    parser.add_argument("--device-type", type=str, default="nvidia")
    parser.add_argument("--ndev", type=int, default=4)
    parser.add_argument("--max-kvcache-tokens", type=int, default=131072)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    model_path = args.model_path
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
    else:
        logger.info(
            # "Usage: python jiuge.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore] <path/to/model_dir> [n_device]"
            "Usage: python jiuge.py [cpu | nvidia| cambricon | ascend | metax | moore] <path/to/model_dir> [n_device]"
        )
        sys.exit(1)

    seed(0)
    # num_seqs = 128
    num_seqs = 8
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser("/home/wanghaojie/vllm/huggingface/9G7B_MHA/")
    llm = LLM(path, device=device_type, enforce_eager=True, 
              tensor_parallel_size=args.ndev, trust_remote_code=True, 
              attention_bias=True, enable_paged_attn=True, max_kvcache_tokens=max_kvcache_tokens)


    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    # llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    outputs = llm.generate(prompt_token_ids, sampling_params)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
