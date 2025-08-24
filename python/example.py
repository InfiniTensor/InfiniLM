import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import sys
from transformers import AutoTokenizer
import argparse

from icinfer import LLM, SamplingParams
from icinfer.engine.libinfinicore_infer import DeviceType

import logging
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model-path", type=str, default="/home/wanghaojie/vllm/huggingface/Llama-2-7b-chat-hf")
    # parser.add_argument("--model-path", type=str, default="/home/wanghaojie/vllm/huggingface/FM9G_70B_SFT_MHA/")
    parser.add_argument("--model-path", type=str, default="/home/wanghaojie/vllm/huggingface/9G7B_MHA/")
    parser.add_argument("--device-type", type=str, default="nvidia")
    parser.add_argument("--ndev", type=int, default=4)
    parser.add_argument("--max-kvcache-tokens", type=int, default=65536)
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

    # path = os.path.expanduser("~/vllm/huggingface/Qwen3-0.6B/")
    # tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    # llm = LLM(path, enforce_eager=True, tensor_parallel_size=1, trust_remote_code=True)
    path = os.path.expanduser("/home/wanghaojie/vllm/huggingface/9G7B_MHA/")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    llm = LLM(path, device=device_type, enforce_eager=True, 
              tensor_parallel_size=args.ndev, trust_remote_code=True, 
              attention_bias=True, enable_paged_attn=True, max_kvcache_tokens=max_kvcache_tokens)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=128)
    prompts = [
        "introduce yourself",
        # "list all prime numbers within 100",
        "山东最高的山是？",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()


"""





"""