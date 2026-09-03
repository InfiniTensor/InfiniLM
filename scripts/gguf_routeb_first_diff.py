#!/usr/bin/env python3
"""Inspect llama.cpp and InfiniLM logits at the first token divergence."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request


def post_json(url: str, body: dict, timeout: int = 180) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError("HTTP %d: %s" % (exc.code, detail[:2000])) from exc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--compare", required=True)
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:18080")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--num-blocks", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import numpy as np
    import infinicore
    from infinilm.cache import PagedKVCacheConfig
    from infinilm.distributed import DistConfig
    from infinilm.infer_engine import InferEngine
    from infinilm.lib import _infinilm
    from infinilm.modeling_utils import load_model_state_dict_by_file

    with open(args.inputs, encoding="utf-8") as f:
        inputs = {x["id"]: x for x in json.load(f)["cases"]}
    with open(args.compare, encoding="utf-8") as f:
        compared = {x["id"]: x for x in json.load(f)["cases"]}
    item = compared[args.case_id]
    first_diff = item["first_difference"]
    if first_diff is None:
        raise ValueError("case %s has no divergence" % args.case_id)
    common_generated = item["llama_tokens"][:first_diff]
    assert common_generated == item["infinilm_tokens"][:first_diff]
    prefix = [int(x) for x in inputs[args.case_id]["input_ids"] + common_generated]

    llama_body = {
        "prompt": prefix,
        "n_predict": 1,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "typical_p": 1.0,
        "repeat_penalty": 1.0,
        "repeat_last_n": 0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "seed": 1,
        "ignore_eos": True,
        "cache_prompt": False,
        "return_tokens": True,
        "n_probs": args.top_k,
        "stream": False,
        "samplers": ["top_k", "temperature"],
    }
    llama_response = post_json(
        args.server.rstrip("/") + "/completion", llama_body)
    llama_probs = llama_response["completion_probabilities"][0]["top_logprobs"]

    load_started = time.time()
    engine = InferEngine(
        model_path=args.model_path,
        device=infinicore.device("cuda:0"),
        distributed_config=DistConfig(1),
        cache_config=PagedKVCacheConfig(
            args.num_blocks, args.block_size, max_batch_size=1),
        attention_backend="paged-attn",
    )
    load_model_state_dict_by_file(engine, args.model_path, dtype=engine.dtype)
    load_s = time.time() - load_started

    length = len(prefix)
    positions = list(range(length))
    if engine.position_id_axes > 1:
        positions = [positions for _ in range(engine.position_id_axes)]
    tensors = {
        "input_ids": infinicore.from_list([prefix], dtype=infinicore.int64).view([1, length]),
        "position_ids": infinicore.from_list(positions, dtype=infinicore.int64),
        "past_kv_lengths": infinicore.from_list([0], dtype=infinicore.int32),
        "total_kv_lengths": infinicore.from_list([length], dtype=infinicore.int32),
        "input_offsets": infinicore.from_list([0, length], dtype=infinicore.int32),
        "cu_seqlens": infinicore.from_list([0, length], dtype=infinicore.int32),
        "block_tables": infinicore.from_list([[0]], dtype=infinicore.int32),
        "slot_mapping": infinicore.from_list(list(range(length)), dtype=infinicore.int64),
        "mamba_init_state_indices": infinicore.from_list([0], dtype=infinicore.int32),
        "mamba_final_state_indices": infinicore.from_list([1], dtype=infinicore.int32),
    }
    cpp_input = engine._build_input(
        tensors["input_ids"],
        position_ids=tensors["position_ids"],
        past_kv_lengths=tensors["past_kv_lengths"],
        total_kv_lengths=tensors["total_kv_lengths"],
        input_offsets=tensors["input_offsets"],
        cu_seqlens=tensors["cu_seqlens"],
        block_tables=tensors["block_tables"],
        slot_mapping=tensors["slot_mapping"],
        mamba_init_state_indices=tensors["mamba_init_state_indices"],
        mamba_final_state_indices=tensors["mamba_final_state_indices"],
        sample_all_positions=False,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
    )
    output = _infinilm.InferEngine.forward(engine, cpp_input)
    raw_logits = infinicore.Tensor(output.logits)
    logits_shape = list(raw_logits.shape)
    cpu_logits = raw_logits.to(infinicore.device("cpu", 0))
    if cpu_logits.dtype != infinicore.bfloat16:
        raise TypeError("expected BF16 logits, got %s" % cpu_logits.dtype)
    bits_type = ctypes.c_uint16 * cpu_logits.numel()
    bits = np.ctypeslib.as_array(bits_type.from_address(cpu_logits.data_ptr())).copy()
    all_logits = (bits.astype(np.uint32) << 16).view(np.float32).reshape(logits_shape)
    logits = all_logits.reshape(-1, logits_shape[-1])[-1]
    order = np.argpartition(logits, -args.top_k)[-args.top_k:]
    order = order[np.argsort(logits[order])[::-1]]
    max_logit = float(logits[order[0]])
    infini_top = [{"id": int(i), "logit": float(logits[i]),
                   "delta_from_top": float(logits[i] - max_logit)} for i in order]

    llama_map = {int(x["id"]): float(x["logprob"]) for x in llama_probs}
    infini_map = {int(x["id"]): float(x["delta_from_top"]) for x in infini_top}
    candidate_ids = sorted(set(llama_map) | set(infini_map))
    candidate_table = [{
        "id": token_id,
        "llama_logprob": llama_map.get(token_id),
        "infini_delta_from_top": infini_map.get(token_id),
    } for token_id in candidate_ids]

    result = {
        "case_id": args.case_id,
        "first_difference": first_diff,
        "base_input_ids": inputs[args.case_id]["input_ids"],
        "common_generated_prefix": common_generated,
        "diagnostic_prefix": prefix,
        "llama_selected": int(llama_response["tokens"][0]),
        "infinilm_selected": int(order[0]),
        "llama_top_logprobs": llama_probs,
        "infinilm_top_logits": infini_top,
        "candidate_table": candidate_table,
        "infinilm_logits_shape": logits_shape,
        "infinilm_logits_finite": bool(np.isfinite(logits).all()),
        "infinilm_load_s": round(load_s, 4),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("CASE=%s diff=%d prefix_len=%d llama=%d infini=%d" % (
        args.case_id, first_diff, len(prefix), result["llama_selected"],
        result["infinilm_selected"]))
    print("LLAMA_TOP5 %s" % [(x["id"], round(x["logprob"], 6))
                              for x in llama_probs[:5]])
    print("INFINI_TOP5 %s" % [(x["id"], round(x["delta_from_top"], 6))
                               for x in infini_top[:5]])
    print("FINITE=%s SHAPE=%s LOAD=%.3fs" % (
        result["infinilm_logits_finite"], result["infinilm_logits_shape"], load_s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
