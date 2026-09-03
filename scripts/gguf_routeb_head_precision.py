#!/usr/bin/env python3
"""Recompute divergent lm_head rows in FP32 from GGUF weights and traced hidden states."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(
    os.environ.get("LLAMA_CPP_DIR", "/home/liuxd/llama.cpp"), "gguf-py"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--compare", required=True)
    ap.add_argument("--infinilm-trace", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import numpy as np
    from gguf import GGUFReader
    from gguf.constants import GGMLQuantizationType
    from gguf.quants import dequantize

    with open(args.compare, encoding="utf-8") as f:
        compared = {x["id"]: x for x in json.load(f)["cases"]}
    with open(args.infinilm_trace, encoding="utf-8") as f:
        traced = json.load(f)["cases"]
    with open(os.path.join(args.model_path, "model.safetensors.index.json"),
              encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    from safetensors import safe_open
    native_shard = safe_open(
        os.path.join(args.model_path, weight_map["lm_head.weight"]),
        framework="pt", device="cpu")
    native_head = native_shard.get_slice("lm_head.weight")
    reader = GGUFReader(args.gguf, "r")
    output = next(t for t in reader.tensors if t.name == "output.weight")
    type_name = GGMLQuantizationType(int(output.tensor_type)).name
    results = []
    for case in traced:
        item = compared[case["case_id"]]
        diff = int(item["first_difference"])
        llama_token = int(item["llama_tokens"][diff])
        infini_token = int(item["infinilm_tokens"][diff])
        step = case["steps"][diff]
        bits = np.asarray(step["hidden_bf16_bits"], dtype=np.uint16)
        hidden = (bits.astype(np.uint32) << 16).view(np.float32)
        rows = []
        for token_id in (llama_token, infini_token):
            raw_row = output.data[token_id:token_id + 1]
            row = np.asarray(
                dequantize(raw_row, GGMLQuantizationType(int(output.tensor_type))),
                dtype=np.float32).reshape(-1)
            rows.append(row)
        logits = [float(np.dot(hidden, row)) for row in rows]
        native_rows = [
            native_head[token_id:token_id + 1].float().numpy().reshape(-1)
            for token_id in (llama_token, infini_token)
        ]
        native_logits = [float(np.dot(hidden, row)) for row in native_rows]
        result = {
            "case_id": case["case_id"], "first_difference": diff,
            "llama_token": llama_token, "infinilm_token": infini_token,
            "llama_token_fp32_logit": logits[0],
            "infinilm_token_fp32_logit": logits[1],
            "fp32_margin_llama_minus_infinilm": logits[0] - logits[1],
            "fp32_winner": llama_token if logits[0] > logits[1] else infini_token,
            "bf16_weight_fp32_margin_llama_minus_infinilm":
                native_logits[0] - native_logits[1],
            "bf16_weight_fp32_winner":
                llama_token if native_logits[0] > native_logits[1] else infini_token,
            "hidden_shape": step["hidden_shape"],
        }
        results.append(result)
        print("%-10s llama=%6d infini=%6d gguf_f32=%+.8f bf16w_f32=%+.8f winner=%d" % (
            result["case_id"], llama_token, infini_token,
            result["fp32_margin_llama_minus_infinilm"],
            result["bf16_weight_fp32_margin_llama_minus_infinilm"],
            result["bf16_weight_fp32_winner"]),
            flush=True)
    report = {"gguf_lm_head_type": type_name, "cases": results}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("RESULT gguf_f32_llama_wins=%d/%d bf16_weight_f32_llama_wins=%d/%d" % (
        sum(x["fp32_winner"] == x["llama_token"] for x in results), len(results),
        sum(x["bf16_weight_fp32_winner"] == x["llama_token"] for x in results), len(results)),
        flush=True)


if __name__ == "__main__":
    main()
