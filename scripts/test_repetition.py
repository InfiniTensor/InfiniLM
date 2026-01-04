#!/usr/bin/env python3
"""
Quick test to reproduce repetition issue for debugging.
Usage: python test_repetition.py [--cpu | --metax | --nvidia] <model_path> [repetition_penalty]
"""
import sys
import time
from jiuge import JiugeForCauslLM, InferTask, KVCache, DeviceType

def test_repetition(model_path, device_type, repetition_penalty=1.2, max_steps=1024, ndev=1):
    """Test repetition penalty with a simple prompt to quickly reproduce issues."""
    print(f"Loading model from {model_path}...")
    print(f"Device: {device_type}, Repetition Penalty: {repetition_penalty}")
    print("=" * 80)

    model = JiugeForCauslLM(model_path, device_type, ndev)

    # Use the exact hard-coded case from THINKING_TAG_AND_SYMBOL_LOOP_ISSUE.md
    system_message = "你是一个由启元实验室开发的九格助手，你擅长中英文对话，能够理解并处理各种问题，提供安全、有帮助、准确的回答。当前时间：2025-12-26#注意：回复之前注意结合上下文和工具返回内容进行回复"
    # user_message = "请详细分析人工智能技术在现代社会中的广泛应 用及其对各个行业产生的深远影响。人工智能作为21世纪最重要的技术革命之一，正在深刻改 变着我们的生活方式、工作模式和社会结构。从医疗健康领域的智能诊断系统，到金融行业的 风险控制和智能投顾，从教育领域的个性化学习平台，到制造业的智能制造和工业4.0，人工智能技术无处不在。在交通领域，自动驾驶技术正在逐步成熟，有望彻底改变我们的出行方式。 在零售行业，智能推荐系统和无人商店正在重塑购物体验。在农业领域，精准农业和智能灌溉 系统提高了生产效率。在能源领域，智能电网和能源管理系统优化了能源分配。在环境保护方 面，人工智能帮助监测污染、预测气候变化、优化资源利用。在科学研究中，AI加速了药物发 现、材料设计、天文观测等领域的突破。然而，人工智能的快速发展也带来了诸多挑战和思考 。就业市场的结构性变化、数据隐私和安全问题、算法偏见和公平性、技术伦理和责任等议题 日益凸显。我们需要在推动技术创新的同时，建立健全的法律法规和伦理框架，确保人工智能 技术的发展能够造福全人类，促进社会的公平、包容和可持续发展。请从技术发展、应用场景 、社会影响、伦理考量等多个维度进行全面深入的分析和讨论。"
    user_message="你是谁"

    input_content = model.tokenizer.apply_chat_template(
        conversation=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        add_generation_prompt=True,
        tokenize=False,
    )

    print(f"System message: {system_message[:50]}...")
    print(f"User message length: {len(user_message)} chars")
    print(f"Input tokens: {len(model.tokenizer.encode(input_content))}")
    print("=" * 80)
    print("Generated output:")
    print("-" * 80)

    tokens = model.tokenizer.encode(input_content)
    # Set random seed for deterministic testing
    import random
    import numpy as np
    import torch
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    infer_task = InferTask(
        0,
        tokens,
        model.max_context_len(),
        temperature=0.65,  # Match the curl request
        topk=0,  # Disabled topk (use full vocabulary)
        topp=0.7,  # Match the curl request (top_p: 0.7)
        end_tokens=model.eos_token_id,
        repetition_penalty=repetition_penalty,
    )
    infer_task.bind_kvcache(KVCache(model))

    generated_tokens = []
    output_text = ""

    for step_i in range(max_steps):
        start_time = time.time()
        output_tokens = model.batch_infer_one_round([infer_task])
        end_time = time.time()

        output_token = output_tokens[0]
        generated_tokens.append(output_token)
        output_str = model.tokenizer.decode([output_token])
        output_text += output_str

        print(output_str, end="", flush=True)

        if output_token in model.eos_token_id:
            print("\n[EOS token reached]")
            break

        infer_task.next(output_token)

        # if step_i > 0:
        #     step_time = (end_time - start_time) * 1000
        #     print(f" [{step_time:.2f}ms]", end="", flush=True)

    print("\n" + "=" * 80)
    print(f"Total generated tokens: {len(generated_tokens)}")
    print(f"Generated text length: {len(output_text)}")

    # Check for repetition
    print("\nRepetition Analysis:")
    print("-" * 80)

    # Check for repeated token sequences
    if len(generated_tokens) >= 4:
        # Check for 2-token repetition
        for i in range(len(generated_tokens) - 3):
            seq = tuple(generated_tokens[i:i+2])
            for j in range(i+2, len(generated_tokens) - 1):
                if tuple(generated_tokens[j:j+2]) == seq:
                    seq_text = model.tokenizer.decode(list(seq))
                    print(f"⚠️  Found 2-token repetition at positions {i}-{i+1} and {j}-{j+1}: '{seq_text}'")

        # Check for 3-token repetition
        for i in range(len(generated_tokens) - 4):
            seq = tuple(generated_tokens[i:i+3])
            for j in range(i+3, len(generated_tokens) - 2):
                if tuple(generated_tokens[j:j+3]) == seq:
                    seq_text = model.tokenizer.decode(list(seq))
                    print(f"⚠️  Found 3-token repetition at positions {i}-{i+2} and {j}-{j+2}: '{seq_text}'")

    # Check for repeated single tokens (more than 3 times in a row)
    consecutive_count = 1
    last_token = generated_tokens[0] if generated_tokens else None
    for i in range(1, len(generated_tokens)):
        if generated_tokens[i] == last_token:
            consecutive_count += 1
            if consecutive_count >= 4:
                token_text = model.tokenizer.decode([last_token])
                print(f"⚠️  Found {consecutive_count} consecutive repetitions of token '{token_text}' (id={last_token})")
        else:
            consecutive_count = 1
            last_token = generated_tokens[i]

    # Print token IDs for debugging
    print(f"\nGenerated token IDs: {generated_tokens[:50]}")  # First 50 tokens
    if len(generated_tokens) > 50:
        print(f"... (total {len(generated_tokens)} tokens)")

    # Print full token history from KV cache for debugging
    if infer_task.kvcache() is not None and hasattr(infer_task.kvcache(), 'tokens'):
        kv_tokens = infer_task.kvcache().tokens[:infer_task.pos + len(infer_task.tokens)]
        print(f"\nKV cache token history (pos={infer_task.pos}, len={len(infer_task.tokens)}):")
        print(f"  Total tokens in cache: {len([t for t in kv_tokens if t != 0])}")
        print(f"  First 20 tokens: {kv_tokens[:20]}")
        if len(kv_tokens) > 20:
            print(f"  Last 20 tokens: {kv_tokens[-20:]}")

    infer_task._kv_cache.drop(model)
    model.destroy_model_instance()

    return output_text, generated_tokens


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nUsage:")
        print("  python test_repetition.py [--cpu | --metax | --nvidia] <model_path> [repetition_penalty] [max_steps]")
        print("\nExample:")
        print("  python test_repetition.py --metax /path/to/model 1.2 30")
        sys.exit(1)

    device_arg = sys.argv[1]
    model_path = sys.argv[2]
    repetition_penalty = float(sys.argv[3]) if len(sys.argv) > 3 else 1.2
    max_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 1024  # Match curl request max_tokens: 1024
    ndev = int(sys.argv[5]) if len(sys.argv) > 5 else 1

    device_type = DeviceType.DEVICE_TYPE_CPU
    if device_arg == "--cpu":
        device_type = DeviceType.DEVICE_TYPE_CPU
    elif device_arg == "--metax":
        device_type = DeviceType.DEVICE_TYPE_METAX
    elif device_arg == "--nvidia":
        device_type = DeviceType.DEVICE_TYPE_NVIDIA
    elif device_arg == "--ascend":
        device_type = DeviceType.DEVICE_TYPE_ASCEND
    else:
        print(f"Unknown device: {device_arg}")
        print("Supported: --cpu, --metax, --nvidia, --ascend")
        sys.exit(1)

    test_repetition(model_path, device_type, repetition_penalty, max_steps, ndev)


if __name__ == "__main__":
    main()
