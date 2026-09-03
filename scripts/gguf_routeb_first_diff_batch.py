#!/usr/bin/env python3
"""Inspect first-divergence logits for every non-exact Route-B case."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
import urllib.request


def post_json(url: str, body: dict, timeout: int = 180) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--compare", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--server", default="http://127.0.0.1:18080")
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--num-blocks", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import infinicore
    import numpy as np
    from infinilm.cache import PagedKVCacheConfig
    from infinilm.distributed import DistConfig
    from infinilm.infer_engine import InferEngine
    from infinilm.lib import _infinilm
    from infinilm.modeling_utils import load_model_state_dict_by_file

    with open(args.inputs, encoding="utf-8") as f:
        inputs = {x["id"]: x for x in json.load(f)["cases"]}
    with open(args.compare, encoding="utf-8") as f:
        compared = json.load(f)["cases"]
    divergent = [x for x in compared if x["first_difference"] is not None]

    started = time.time()
    engine = InferEngine(
        model_path=args.model_path,
        device=infinicore.device("cuda:0"),
        distributed_config=DistConfig(1),
        cache_config=PagedKVCacheConfig(
            args.num_blocks, args.block_size, max_batch_size=1
        ),
        attention_backend="paged-attn",
    )
    load_model_state_dict_by_file(engine, args.model_path, dtype=engine.dtype)
    load_s = time.time() - started

    results = []
    for item in divergent:
        case_id = item["id"]
        first_diff = item["first_difference"]
        common = item["llama_tokens"][:first_diff]
        assert common == item["infinilm_tokens"][:first_diff]
        prefix = [int(x) for x in inputs[case_id]["input_ids"] + common]
        body = {
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
        llama = post_json(args.server.rstrip("/") + "/completion", body)
        llama_probs = llama["completion_probabilities"][0]["top_logprobs"]

        length = len(prefix)
        positions = list(range(length))
        if engine.position_id_axes > 1:
            positions = [positions for _ in range(engine.position_id_axes)]
        tensors = {
            "input_ids": infinicore.from_list([prefix], dtype=infinicore.int64).view(
                [1, length]
            ),
            "position_ids": infinicore.from_list(positions, dtype=infinicore.int64),
            "past_kv_lengths": infinicore.from_list([0], dtype=infinicore.int32),
            "total_kv_lengths": infinicore.from_list([length], dtype=infinicore.int32),
            "input_offsets": infinicore.from_list([0, length], dtype=infinicore.int32),
            "cu_seqlens": infinicore.from_list([0, length], dtype=infinicore.int32),
            "block_tables": infinicore.from_list([[0]], dtype=infinicore.int32),
            "slot_mapping": infinicore.from_list(
                list(range(length)), dtype=infinicore.int64
            ),
            "mamba_init_state_indices": infinicore.from_list(
                [0], dtype=infinicore.int32
            ),
            "mamba_final_state_indices": infinicore.from_list(
                [1], dtype=infinicore.int32
            ),
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
        raw = infinicore.Tensor(output.logits)
        shape = list(raw.shape)
        cpu = raw.to(infinicore.device("cpu", 0))
        if cpu.dtype != infinicore.bfloat16:
            raise TypeError("expected BF16 logits, got %s" % cpu.dtype)
        bits_type = ctypes.c_uint16 * cpu.numel()
        bits = np.ctypeslib.as_array(bits_type.from_address(cpu.data_ptr())).copy()
        logits = (bits.astype(np.uint32) << 16).view(np.float32).reshape(shape)
        logits = logits.reshape(-1, shape[-1])[-1]
        order = np.argpartition(logits, -args.top_k)[-args.top_k :]
        order = order[np.argsort(logits[order], kind="stable")[::-1]]
        top_logit = float(logits[order[0]])
        infini_top = [
            {
                "id": int(i),
                "logit": float(logits[i]),
                "delta_from_top": float(logits[i] - top_logit),
            }
            for i in order
        ]
        llama_map = {int(x["id"]): float(x["logprob"]) for x in llama_probs}
        infini_map = {x["id"]: x["delta_from_top"] for x in infini_top}
        llama_selected = int(llama["tokens"][0])
        infini_selected = int(order[0])
        candidate_ids = sorted(set(llama_map) | set(infini_map))
        candidate_table = [
            {
                "id": token_id,
                "llama_logprob": llama_map.get(token_id),
                "infini_delta_from_top": infini_map.get(token_id),
                "infini_logit": float(logits[token_id]),
            }
            for token_id in candidate_ids
        ]
        selected_logits = {
            "llama_token_infini_logit": float(logits[llama_selected]),
            "infini_token_infini_logit": float(logits[infini_selected]),
            "infini_margin_selected_minus_llama": float(
                logits[infini_selected] - logits[llama_selected]
            ),
            "llama_margin_selected_minus_infini": float(
                llama_map[llama_selected] - llama_map.get(infini_selected, float("nan"))
            ),
        }
        result = {
            "case_id": case_id,
            "first_difference": first_diff,
            "prefix_length": len(prefix),
            "llama_selected": llama_selected,
            "infinilm_selected": infini_selected,
            "llama_top_logprobs": llama_probs,
            "infinilm_top_logits": infini_top,
            "selected_pair": selected_logits,
            "candidate_table": candidate_table,
            "infinilm_logits_shape": shape,
            "infinilm_logits_finite": bool(np.isfinite(logits).all()),
        }
        results.append(result)
        print(
            "%-10s diff=%2d llama=%6d infini=%6d llama_margin=%+.6f infini_margin=%+.6f"
            % (
                case_id,
                first_diff,
                llama_selected,
                infini_selected,
                selected_logits["llama_margin_selected_minus_infini"],
                selected_logits["infini_margin_selected_minus_llama"],
            )
        )

    report = {"load_s": round(load_s, 4), "case_count": len(results), "cases": results}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(
        "RESULT cases=%d finite=%s load=%.3fs"
        % (len(results), all(x["infinilm_logits_finite"] for x in results), load_s)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
