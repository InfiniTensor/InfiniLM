#!/usr/bin/env python3
"""Record a reproducible Transformers reference for LFM2-1.2B.

Run this on the NVIDIA instance after the official weights are available.  The
script performs greedy generation and writes token IDs plus per-step top-k
scores, which are more useful for backend comparison than decoded text alone.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Local LFM2-1.2B directory or Hugging Face model ID.",
    )
    parser.add_argument("--prompt", default="Who are you?")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=False,
    ).to(args.device)
    model.eval()

    messages = [{"role": "user", "content": args.prompt}]
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    encoded = {name: value.to(args.device) for name, value in encoded.items()}

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

    prompt_length = encoded["input_ids"].shape[1]
    sequence = generated.sequences[0].detach().cpu()
    generated_ids = sequence[prompt_length:]

    step_scores = []
    for step, logits in enumerate(generated.scores):
        values, indices = torch.topk(logits[0].float(), k=args.top_k)
        step_scores.append(
            {
                "step": step,
                "selected_token_id": int(generated_ids[step]),
                "top_token_ids": [int(x) for x in indices.detach().cpu()],
                "top_scores": [float(x) for x in values.detach().cpu()],
            }
        )

    result = {
        "model": args.model,
        "prompt": args.prompt,
        "chat_messages": messages,
        "input_ids": encoded["input_ids"][0].detach().cpu().tolist(),
        "generated_token_ids": generated_ids.tolist(),
        "full_token_ids": sequence.tolist(),
        "decoded_text": tokenizer.decode(
            sequence,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ),
        "generated_text": tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ),
        "step_scores": step_scores,
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "transformers": transformers.__version__,
            "device": args.device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(torch.device(args.device))
                if args.device.startswith("cuda")
                else None
            ),
            "dtype": "bfloat16",
            "attention_implementation": "eager",
            "decoding": "greedy",
        },
    }

    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
