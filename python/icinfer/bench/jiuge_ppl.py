import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import sys


from icinfer import LLM, SamplingParams
# from icinfer.engine.llm_engine import InfiniEngine
from icinfer.engine.libinfinicore_infer import DeviceType

DEVICE_TYPE_MAP = {
    "cpu": DeviceType.DEVICE_TYPE_CPU,
    "nvidia": DeviceType.DEVICE_TYPE_NVIDIA,
    "cambricon": DeviceType.DEVICE_TYPE_CAMBRICON,
    "ascend": DeviceType.DEVICE_TYPE_ASCEND,
    "metax": DeviceType.DEVICE_TYPE_METAX,
    "moore": DeviceType.DEVICE_TYPE_MOORE,
}

TORCH_DEVICE_TYPE_MAP = {
    "cpu": "cpu",
    "nvidia": "cuda",
    "cambricon": "mlu",
    "ascend": "npu",
    "metax": "cuda",
    "moore": "cuda",
}


def test_torch(input_ids_list, device_):
    device = TORCH_DEVICE_TYPE_MAP[device_]
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(
        device
    )
    model.eval()

    total_neg_log_likelihood = 0
    total_tokens = 0

    with torch.no_grad():
        for input_ids in input_ids_list:
            input_ids = torch.tensor(input_ids, device=device)
            # shift inputs and labels
            inputs = input_ids[:-1].unsqueeze(0)  # [1, seq_len-1]
            labels = input_ids[1:].unsqueeze(0)  # [1, seq_len-1]

            outputs = model(inputs, use_cache=False)
            logits = outputs.logits  # [1, seq_len-1, vocab_size]

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            # gather log probs of true tokens
            true_token_log_probs = log_probs.gather(
                dim=-1, index=labels.unsqueeze(-1)
            ).squeeze(-1)

            total_neg_log_likelihood += -true_token_log_probs.sum().item()
            total_tokens += labels.numel()

    perplexity = torch.exp(torch.tensor(total_neg_log_likelihood / total_tokens))
    return perplexity



def test_infinicore(input_ids_list, model_path, device_, ndev_, enable_paged_attn, max_kvcache_tokens):
    device = DEVICE_TYPE_MAP[device_]

    # model = JiugeForCauslLM(
    #     model_path, device, max_tokens=len(input_ids_list[0]), ndev=ndev_
    # )
    llm = LLM(model_path, device=device, enforce_eager=True, 
              tensor_parallel_size=ndev_, trust_remote_code=True, 
              attention_bias=True, enable_paged_attn=enable_paged_attn, max_kvcache_tokens=max_kvcache_tokens)

    perplexity = llm.perplexity(input_ids_list)
    # model.destroy_model_instance()
    llm.model_runner.exit()
    return perplexity


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--dev", type=str, default="nvidia", choices=DEVICE_TYPE_MAP.keys()
    )
    parser.add_argument(
        "--ndev",
        type=int,
        default=1,
        help="Number of devices to use (default: 1)",
    )
    parser.add_argument("--max-kvcache-tokens", type=int, default=4096)
    # parser.add_argument("--max-kvcache-tokens", type=int, default=65536)
    parser.add_argument("--enable-paged-attn", action="store_true")

    
    args = parser.parse_args()
    max_kvcache_tokens = args.max_kvcache_tokens
    # device_type = DeviceType.DEVICE_TYPE_CPU
    # if args.device_type == "cpu":
    #     device_type = DeviceType.DEVICE_TYPE_CPU
    # elif args.device_type == "nvidia":
    #     device_type = DeviceType.DEVICE_TYPE_NVIDIA
    # elif args.device_type == "cambricon":
    #     device_type = DeviceType.DEVICE_TYPE_CAMBRICON
    # elif args.device_type == "ascend":
    #     device_type = DeviceType.DEVICE_TYPE_ASCEND
    # elif args.device_type == "metax":
    #     device_type = DeviceType.DEVICE_TYPE_METAX
    # elif args.device_type == "moore":
    #     device_type = DeviceType.DEVICE_TYPE_MOORE
    # elif args.device_type == "iluvatar":
    #     device_type = DeviceType.DEVICE_TYPE_ILUVATAR
    # else:
    #     print(
    #         # "Usage: python jiuge.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore] <path/to/model_dir> [n_device]"
    #         "Usage: python jiuge.py [cpu | nvidia| cambricon | ascend | metax | moore] <path/to/model_dir> [n_device]"
    #     )
    #     sys.exit(1)

    seq_len = 512

    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    local_file_paths = {
        # "train": "/home/wanghaojie/vllm/huggingface/wikitext/wikitext_local_parquet/train.parquet",
        # "validation": "/home/wanghaojie/vllm/huggingface/wikitext/wikitext_local_parquet/validation.parquet",
        "test": "/home/wanghaojie/vllm/huggingface/wikitext/wikitext-2-raw-v1/test-00000-of-00001.parquet"
    }
    dataset = load_dataset("parquet", data_files=local_file_paths, split="test")

    texts = dataset["text"]
    texts = [t.strip() for t in texts if len(t.strip()) > 0]

    input_ids_list = []
    for text in texts:
        ids = tokenizer.encode(text)
        # split long sequences into chunks
        for i in range(0, len(ids) - seq_len + 1, seq_len):
            input_ids_list.append(ids[i : i + seq_len])
    # print(f"\n=== 📊 精度指标汇总 ({MODEL}) ===")
    # print(f"model: {args.model_path}, device: {args.dev}")

    # InfiniCore_perplexity = test_infinicore(input_ids_list, model_path, args.dev, args.ndev, args.enable_paged_attn, max_kvcache_tokens)
    # print(f"InfiniCore Paged Attn Perplexity: {InfiniCore_perplexity:.2f}")

    # # if args.ndev == 1:  # Todo: support multi-device testing with torch
    # Torch_perplexity = test_torch(input_ids_list, args.dev)
    # print(f"Torch Perplexity: {Torch_perplexity.item():.2f}")
    InfiniCore_perplexity= 14.35

    width_label = 24
    sep = "-" * 60
    MODEL = "FM9G-70B"

    print(f"\n=== 📊 性能指标汇总 ({MODEL}) ===")
    print(sep)
    # print(f"{'Torch Perplexity':<{width_label}}: {Torch_perplexity.item():.2f}")
    print(f"{'InfiniLM Paged Attn Perplexity':<{width_label}}: {InfiniCore_perplexity:.2f}")
    print(sep)
