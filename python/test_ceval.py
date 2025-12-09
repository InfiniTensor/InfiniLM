import argparse
from datasets import load_dataset
from transformers import AutoTokenizer

from icinfer import SamplingParams
from icinfer.engine.libinfinicore_infer import DeviceType
from icinfer.engine.llm_engine import InfiniEngine


# ---------------------------
# argparse
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--device-type", type=str, default="nvidia")
    parser.add_argument("--ndev", type=int, default=1)
    parser.add_argument("--max-kvcache-tokens", type=int, default=10240)
    parser.add_argument("--enable-paged-attn", action="store_true")

    parser.add_argument("--subject", type=str, default="computer_network",
                        help="CEval subject, e.g. 'computer_network', 'college_physics', etc.")
    parser.add_argument("--split", type=str, default="val")

    return parser.parse_args()


# ---------------------------
# device type parser
# ---------------------------
def parse_device_type(name):
    if name == "cpu": return DeviceType.DEVICE_TYPE_CPU
    if name == "nvidia": return DeviceType.DEVICE_TYPE_NVIDIA
    if name == "cambricon": return DeviceType.DEVICE_TYPE_CAMBRICON
    if name == "ascend": return DeviceType.DEVICE_TYPE_ASCEND
    if name == "metax": return DeviceType.DEVICE_TYPE_METAX
    if name == "moore": return DeviceType.DEVICE_TYPE_MOORE
    if name == "iluvatar": return DeviceType.DEVICE_TYPE_ILUVATAR
    raise ValueError("Unknown device type: ", name)


# ---------------------------
# extract choice
# ---------------------------
def extract_choice(output_text):
    output_text = output_text.strip().upper()
    for opt in ["A", "B", "C", "D"]:
        if opt in output_text:
            return opt
    return None


# ---------------------------
# main
# ---------------------------
def main():
    args = parse_args()

    model_path = args.model_path
    device_type = parse_device_type(args.device_type)

    # ---------------------------
    # Init tokenizer & engine
    # ---------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    llm = InfiniEngine(
        model_path,
        device=device_type,
        ndev=args.ndev,
        enforce_eager=True,
        tensor_parallel_size=args.ndev,
        trust_remote_code=True,
        attention_bias=True,
        enable_paged_attn=args.enable_paged_attn,
        max_kvcache_tokens=args.max_kvcache_tokens
    )

    sampling_params = SamplingParams(
        temperature=0.0,  # greedy
        max_tokens=64
    )

    # ---------------------------
    # Load dataset
    # ---------------------------
    print(f"Loading CEval subject: {args.subject}")
    dataset = load_dataset("<path to ceval>", name=args.subject)
    samples = dataset[args.split]

    total = len(samples)
    print(f"Total samples = {total}")

    # ---------------------------
    # Build all prompts (batch)
    # ---------------------------

    all_prompts = []
    for s in samples:
        
        q_text = (
            f"题目：{s['question']}\n"
            f"A. {s['A']}\n"
            f"B. {s['B']}\n"
            f"C. {s['C']}\n"
            f"D. {s['D']}\n"
            "请直接回答正确选项，例如：A"
        )

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": q_text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        all_prompts.append(prompt)

    # ---------------------------
    # Run ONE batch generation
    # ---------------------------
    print("Running batch generation...")
    outputs, avg_prefill, avg_decode, avg_ttft, avg_tbt, cache_eff = llm.generate(
        all_prompts,
        sampling_params
    )

    # ---------------------------
    # Evaluate accuracy
    # ---------------------------
    correct = 0

    for i, s in enumerate(samples):
        output_text = outputs[i]["text"]
        pred = extract_choice(output_text)
        gold = s["answer"].upper()

        if pred == gold:
            correct += 1

    # ---------------------------
    # Final result
    # ---------------------------
    print("-----------------------------")
    print(f"CEval Subject: {args.subject}")
    print(f"Accuracy: {correct / total:.4f}")
    print(f"batch_size: {len(all_prompts)}, n_dev: {args.ndev}, paged_attn: {args.enable_paged_attn}")
    print(f"Cache Efficiency: {cache_eff*100:.2f}%")
    print("-----------------------------")


if __name__ == "__main__":
    main()
