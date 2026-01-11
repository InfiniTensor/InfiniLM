import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from jiuge import JiugeForCauslLM
from libinfinicore_infer import DeviceType

DEVICE_TYPE_MAP = {
    "cpu": DeviceType.DEVICE_TYPE_CPU,
    "nvidia": DeviceType.DEVICE_TYPE_NVIDIA,
    "cambricon": DeviceType.DEVICE_TYPE_CAMBRICON,
    "ascend": DeviceType.DEVICE_TYPE_ASCEND,
    "metax": DeviceType.DEVICE_TYPE_METAX,
    "moore": DeviceType.DEVICE_TYPE_MOORE,
    "iluvatar": DeviceType.DEVICE_TYPE_ILUVATAR,
    "kunlun": DeviceType.DEVICE_TYPE_KUNLUN,
    "hygon": DeviceType.DEVICE_TYPE_HYGON,
}

TORCH_DEVICE_TYPE_MAP = {
    "cpu": "cpu",
    "nvidia": "cuda",
    "cambricon": "mlu",
    "ascend": "npu",
    "metax": "cuda",
    "moore": "cuda",
    "iluvatar": "cuda",
    "kunlun": "cuda",
    "hygon": "cuda",
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


def test_infinicore(input_ids_list, device_, ndev_):
    device = DEVICE_TYPE_MAP[device_]

    model = JiugeForCauslLM(
        model_path, device, max_tokens=len(input_ids_list[0]), ndev=ndev_
    )
    perplexity = model.perplexity(input_ids_list)
    model.destroy_model_instance()
    return perplexity


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument(
        "--dev", type=str, default="cpu", choices=DEVICE_TYPE_MAP.keys()
    )
    parser.add_argument(
        "--ndev",
        type=int,
        default=1,
        help="Number of devices to use (default: 1)",
    )
    args = parser.parse_args()

    seq_len = 512

    model_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    texts = dataset["text"]
    texts = [t.strip() for t in texts if len(t.strip()) > 0]

    input_ids_list = []
    for text in texts:
        ids = tokenizer.encode(text)
        # split long sequences into chunks
        for i in range(0, len(ids) - seq_len + 1, seq_len):
            input_ids_list.append(ids[i : i + seq_len])

    perplexity = test_infinicore(input_ids_list, args.dev, args.ndev)
    print(f"InfiniCore Perplexity: {perplexity:.2f}")

    if args.ndev == 1:  # Todo: support multi-device testing with torch
        perplexity = test_torch(input_ids_list, args.dev)
        print(f"Torch Perplexity: {perplexity.item():.2f}")
