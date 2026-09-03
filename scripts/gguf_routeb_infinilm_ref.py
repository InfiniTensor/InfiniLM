#!/usr/bin/env python3
"""Run deterministic raw-token completions through InfiniLM paged generate()."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--new-tokens", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--num-blocks", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument(
        "--case-ids",
        help="Optional comma-separated case IDs for focused regression runs.",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import numpy as np
    import infinicore
    from infinilm.cache import PagedKVCacheConfig
    from infinilm.distributed import DistConfig
    from infinilm.infer_engine import GenerationConfig, InferEngine
    from infinilm.modeling_utils import load_model_state_dict_by_file

    with open(args.inputs, encoding="utf-8") as f:
        source = json.load(f)
    if args.case_ids:
        requested = {item.strip() for item in args.case_ids.split(",") if item.strip()}
        source["cases"] = [item for item in source["cases"] if item["id"] in requested]
        found = {item["id"] for item in source["cases"]}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"unknown --case-ids: {missing}")

    started = time.time()
    engine = InferEngine(
        model_path=args.model_path,
        device=infinicore.device("cuda:0"),
        distributed_config=DistConfig(1),
        cache_config=PagedKVCacheConfig(
            args.num_blocks, args.block_size, max_batch_size=1),
        attention_backend="paged-attn",
    )
    load_model_state_dict_by_file(engine, args.model_path, dtype=engine.dtype)
    load_s = time.time() - started
    print("MODEL_LOADED %.3fs cases=%d" % (load_s, len(source["cases"])), flush=True)

    outputs = []
    all_ok = True
    for case in source["cases"]:
        runs = []
        for repeat in range(args.repeats):
            prompt = infinicore.from_list(
                [[int(x) for x in case["input_ids"]]], dtype=infinicore.int64)
            config = GenerationConfig(
                max_new_tokens=args.new_tokens,
                temperature=0.0,
                top_k=1,
                top_p=1.0,
                eos_token_id=None,
                stop_on_eos=False,
                ignore_eos=True,
            )
            run_started = time.time()
            generated = engine.generate(prompt, config)
            tokens = [int(np.asarray(x.to_numpy()).reshape(-1)[0]) for x in generated]
            runs.append({
                "repeat": repeat,
                "tokens": tokens,
                "elapsed_s": round(time.time() - run_started, 4),
            })
        deterministic = all(x["tokens"] == runs[0]["tokens"] for x in runs[1:])
        exact_length = all(len(x["tokens"]) == args.new_tokens for x in runs)
        ok = deterministic and exact_length
        all_ok &= ok
        outputs.append({
            "id": case["id"],
            "prompt": case["prompt"],
            "input_ids": case["input_ids"],
            "deterministic": deterministic,
            "exact_length": exact_length,
            "runs": runs,
        })
        print("%-10s deterministic=%s length=%s tokens=%s" % (
            case["id"], deterministic, exact_length, runs[0]["tokens"]), flush=True)

    result = {
        "engine": "InfiniLM",
        "model_path": os.path.abspath(args.model_path),
        "new_tokens": args.new_tokens,
        "repeats": args.repeats,
        "load_s": round(load_s, 4),
        "cases": outputs,
        "all_pass": all_ok,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("RESULT cases=%d all_pass=%s" % (len(outputs), all_ok), flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
