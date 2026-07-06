#!/usr/bin/env python3
"""Stage 1/2 of the Qwen3-MoE prefill correctness check: dump the HF reference.

Loads Qwen3-30B-A3B-Thinking-2507 with HuggingFace transformers (**reference
only** — the InfiniLM adaptation itself must not depend on transformers), greedily
generates a few tokens for a fixed set of prompts, and saves the prompt token ids
+ generated token ids to JSON. Stage 2 (`check_prefill_logits.py`) then loads the
InfiniLM engine and compares against this file.

Two-phase design because the 30B model does not fit in memory twice: run this
first (HF alone), then run the checker (InfiniLM alone).

Usage:
    # single GPU / multi-GPU (device_map=auto splits across visible GPUs):
    python3 dump_hf_reference.py --model /path/Qwen3-30B-A3B-Thinking-2507 --device cuda
    # CPU (needs ~60GB RAM for bf16; slow but avoids GPU contention):
    python3 dump_hf_reference.py --model /path/... --device cpu
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Fixed prompts covering zh/en + reasoning. Kept in the reference file so the
# checker replays the exact same messages (single source of truth).
DEFAULT_PROMPTS = [
    [{"role": "user", "content": "你好，请介绍一下自己"}],
    [{"role": "user", "content": "用一句话解释什么是量子纠缠。"}],
    [{"role": "user", "content": "What is the capital of France?"}],
    [{"role": "user", "content": "计算 17 乘以 23 等于多少？请给出计算步骤。"}],
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="qwen3_moe_prefill_reference.json")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if args.device == "cuda" else None,
    )
    if args.device == "cpu":
        model = model.to("cpu")
    model.eval()

    entries = []
    for messages in DEFAULT_PROMPTS:
        # Newer transformers return a BatchEncoding (dict) here instead of a bare
        # tensor; handle both. Also pass attention_mask when available.
        tpl = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        if torch.is_tensor(tpl):
            input_ids = tpl.to(model.device)
            attn = None
        else:
            input_ids = tpl["input_ids"].to(model.device)
            attn = tpl.get("attention_mask")
            attn = attn.to(model.device) if attn is not None else None
        with torch.no_grad():
            out = model.generate(
                input_ids, attention_mask=attn,
                max_new_tokens=args.max_new_tokens, do_sample=False,
            )
        gen_ids = out[0, input_ids.shape[1]:].tolist()
        entries.append({
            "messages": messages,
            "prompt_token_ids": input_ids[0].tolist(),
            "gen_token_ids": gen_ids,
            "gen_text": tok.decode(gen_ids),
        })
        first = gen_ids[0] if gen_ids else None
        print(f"[hf] {messages[0]['content'][:20]!r} -> first_tok={first} "
              f"{tok.decode(gen_ids[:1])!r}")

    payload = {
        "model": args.model,
        "max_new_tokens": args.max_new_tokens,
        "note": "greedy (do_sample=False) HF reference; compare with check_prefill_logits.py",
        "entries": entries,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"saved {len(entries)} entries -> {args.out}")


if __name__ == "__main__":
    main()
