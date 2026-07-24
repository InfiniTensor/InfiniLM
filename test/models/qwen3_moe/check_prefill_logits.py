#!/usr/bin/env python3
"""Stage 2/2 of the Qwen3-MoE prefill correctness check: compare InfiniLM vs HF.

Loads the model through the InfiniLM engine (no transformers dependency in the
inference path), greedily generates for the same prompts saved by
`dump_hf_reference.py`, and checks alignment with the HF reference:

  1. prompt tokenization matches HF (validates chat template + thinking prefix),
  2. **first generated token matches** — the prefill next-token = argmax of the
     prefill logits; this is the meaningful "prefill 首步 logits 对齐" signal and
     is robust to the model's run-to-run non-determinism (TP all-reduce order +
     BF16 atomic scatter), which makes a full token-sequence / bit-exact logit
     comparison unreliable,
  3. matching-prefix length is reported as a stronger (best-effort) signal.

PASS when, for every prompt, the prompt tokens match and the first generated
token matches (and prefix >= --prefix-threshold).

Usage:
    python3 check_prefill_logits.py --model /path/Qwen3-30B-A3B-Thinking-2507 \
        --device metax --tp 2 --ref qwen3_moe_prefill_reference.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../python"))

from infinilm.llm.llm import LLM  # noqa: E402

# Same mapping as BaseConfig.get_device_str: the LLM engine only accepts backend
# strings (cpu/cuda/mlu/musa/...), so e.g. "metax"/"iluvatar" must map to "cuda".
_DEVICE_STR_MAP = {
    "cpu": "cpu",
    "nvidia": "cuda",
    "qy": "cuda",
    "cambricon": "mlu",
    "ascend": "ascend",
    "metax": "cuda",
    "moore": "musa",
    "iluvatar": "cuda",
    "kunlun": "kunlun",
    "hygon": "cuda",
    "ali": "cuda",
    # already-backend strings pass through:
    "cuda": "cuda",
    "mlu": "mlu",
    "musa": "musa",
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--ref", required=True, help="reference json from dump_hf_reference.py"
    )
    ap.add_argument("--device", default="metax")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument(
        "--prefix-threshold",
        type=int,
        default=1,
        help="min matching leading tokens required per prompt to PASS (default 1)",
    )
    ap.add_argument(
        "--cache-type",
        default="static",
        choices=["static", "paged"],
        help="KV cache type; static matches examples/test_infer.py's proven path",
    )
    args = ap.parse_args()

    with open(args.ref, encoding="utf-8") as f:
        ref = json.load(f)
    entries = ref["entries"]
    max_new = ref.get("max_new_tokens", 16)

    device_str = _DEVICE_STR_MAP.get(args.device.lower(), "cpu")
    # Mirror examples/test_infer.py's proven invocation (static cache, default
    # attn, no graph) so the engine initializes the same way it does there.
    model = LLM(
        model_path=os.path.expanduser(args.model),
        device=device_str,
        tensor_parallel_size=args.tp,
        cache_type=args.cache_type,
        max_batch_size=len(entries),
        max_tokens=max_new,
        temperature=1.0,
        top_k=1,  # greedy
        top_p=1.0,
        enable_graph=False,
        attn_backend="default",
    )

    conversations = [e["messages"] for e in entries]
    outputs = model.chat(messages=conversations)

    all_pass = True
    print(f"\n{'result':6} | prompt_tok | first_tok (ref vs this) | prefix | prompt")
    print("-" * 90)
    for e, out in zip(entries, outputs):
        ref_gen = e["gen_token_ids"]
        this_gen = list(out.outputs[0].token_ids)
        this_prompt = list(out.prompt_token_ids or [])

        prompt_match = (this_prompt == e["prompt_token_ids"]) if this_prompt else None

        prefix = 0
        for a, b in zip(ref_gen, this_gen):
            if a == b:
                prefix += 1
            else:
                break
        first_match = bool(ref_gen and this_gen and ref_gen[0] == this_gen[0])

        ok = (
            first_match
            and prefix >= args.prefix_threshold
            and (prompt_match is not False)
        )
        all_pass &= ok

        rt = ref_gen[0] if ref_gen else None
        tt = this_gen[0] if this_gen else None
        pm = {True: "ok", False: "MISMATCH", None: "n/a"}[prompt_match]
        print(
            f"{'PASS' if ok else 'FAIL':6} | {pm:10} | {str(rt):>7} {'==' if first_match else '!='} "
            f"{str(tt):<7} | {prefix:>2}/{min(len(ref_gen), len(this_gen))} | "
            f"{e['messages'][0]['content'][:24]!r}"
        )

    print("-" * 90)
    print(
        f"=== OVERALL: {'PASS' if all_pass else 'FAIL'} "
        f"({sum(1 for _ in entries)} prompts) ==="
    )
    if not all_pass:
        print(
            "提示：若仅个别 prefix 早停而 first_tok 全 ==，通常是非确定性/近似平票，非适配错误；"
            "若 first_tok 或 prompt_tok 不一致，才是真 bug（权重加载 / chat template / 路由）。"
        )
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
