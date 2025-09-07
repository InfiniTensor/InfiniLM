import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
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
    # parser.add_argument("--model-path", type=str, default="/home/wanghaojie/vllm/huggingface/FM9G_70B_SFT_MHA/")
    parser.add_argument("--model-path", type=str, default="/home/wanghaojie/vllm/huggingface/9G7B_MHA/")
    parser.add_argument("--device-type", type=str, default="nvidia")
    parser.add_argument("--ndev", type=int, default=1)
    parser.add_argument("--max-kvcache-tokens", type=int, default=10240)
    # parser.add_argument("--max-kvcache-tokens", type=int, default=65536)
    parser.add_argument("--enable-paged-attn", action="store_true")
    # parser.add_argument("--enable-paged-attn", type=bool, default=True)
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
    # path = os.path.expanduser("/home/wanghaojie/vllm/huggingface/9G7B_MHA/")
    path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    llm = LLM(path, device=device_type, enforce_eager=True, 
              tensor_parallel_size=args.ndev, trust_remote_code=True, 
              attention_bias=True, enable_paged_attn=args.enable_paged_attn, max_kvcache_tokens=max_kvcache_tokens)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=128)
    # prompts = [
    #     "introduce yourself",
    #     # "list all prime numbers within 100",
    #     "山东最高的山是？",
    #     "如果猫能写诗，它们会写些什么？",
    #     "描述一个没有重力的世界。",
    #     "如果地球停止自转，会发生什么？",
    #     "假设你是一只会飞的鲸鱼，描述你的日常生活。",
    #     "如果人类可以与植物沟通，世界会变成什么样？",
    #     "描述一个由糖果构成的城市。",
    #     "如果时间旅行成为可能，你最想去哪个时代？",
    #     "想象一下，如果地球上只有蓝色，其他颜色都消失了。",
    #     "如果动物能上网，它们会浏览什么网站？",
    #     "描述一个没有声音的世界。",
    #     "如果人类可以在水下呼吸，城市会如何变化？",
    #     "想象一下，如果天空是绿色的，云是紫色的。",
    #     "如果你能与任何历史人物共进晚餐，你会选择谁？",
    #     "描述一个没有夜晚的星球。",
    #     "如果地球上只有一种语言，世界会如何运作？",
    #     "想象一下，如果所有的书都变成了音乐。",
    #     "如果你可以变成任何一种动物，你会选择什么？",
    #     "描述一个由机器人统治的未来世界。",
    #     "如果你能与任何虚构角色成为朋友，你会选择谁？",
    #     "想象一下，如果每个人都能读懂他人的思想。"
    # ] * 2
    prompts = [
        # "描述一个由糖果构成的城市。",
        # "如果时间旅行成为可能，你最想去哪个时代？",
        # "如果时间旅行成为可能，你最想去哪个时代？",
        # "想象一下，如果地球上只有蓝色，其他颜色都消失了。",
        # "如果动物能上网，它们会浏览什么网站？",
        # "描述一个由糖果构成的城市。",
        # "如果时间旅行成为可能，你最想去哪个时代？",
        # "想象一下，如果地球上只有蓝色，其他颜色都消失了。",
        # "如果动物能上网，它们会浏览什么网站？",
        
        "如果人类可以与植物沟通，世界会变成什么样？",
        "描述一个由糖果构成的城市。",
        "如果时间旅行成为可能，你最想去哪个时代？",
        "想象一下，如果地球上只有蓝色，其他颜色都消失了。",
        "如果动物能上网，它们会浏览什么网站？",
        "描述一个没有声音的世界。",
        "如果人类可以在水下呼吸，城市会如何变化？",
        "想象一下，如果天空是绿色的，云是紫色的。",
        # "如果你能与任何历史人物共进晚餐，你会选择谁？",
        # "描述一个没有夜晚的星球。",
        # "如果地球上只有一种语言，世界会如何运作？",
        # "想象一下，如果所有的书都变成了音乐。",
        # "如果你可以变成任何一种动物，你会选择什么？",
        # "描述一个由机器人统治的未来世界。",
        # "如果你能与任何虚构角色成为朋友，你会选择谁？",
        # "想象一下，如果每个人都能读懂他人的思想。"

        # "如果人类可以与植物沟通，世界会变成什么样？",
        # "描述一个由糖果构成的城市。",
        # "如果人类可以与植物沟通，世界会变成什么样？",

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
    outputs, avg_prefill_throughput, avg_decode_throughput,  avg_ttft, avg_tbt, cache_efficiency = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")
    # print("\n")
    # print(f"Prompt: {prompts[0]!r}")
    # print(f"Completion: {outputs[0]['text']!r}")
    print(f"batch_size: {len(prompts)}, n_dev: {args.ndev}, is_paged_attn: {args.enable_paged_attn}")
    print(f"Avg Prefill Throughput: {avg_prefill_throughput:.2f} tok/s")
    print(f"Avg Decode Throughput: {avg_decode_throughput:.2f} tok/s")
    print(f"Avg TTFT: {avg_ttft*1000:.2f} ms")
    print(f"Avg TBT: {avg_tbt*1000:.2f} ms")
    print(f"Cache Efficiency: {cache_efficiency*100:.2f}%")

if __name__ == "__main__":
    main()


"""
CLI:
python example.py --model-path /home/wanghaojie/vllm/huggingface/9G7B_MHA/ --device-type nvidia --ndev 4 --max-kvcache-tokens 10240 --enable-paged-attn
python example.py --model-path /home/wanghaojie/vllm/huggingface/9G7B_MHA/ --device-type nvidia --ndev 4

"""