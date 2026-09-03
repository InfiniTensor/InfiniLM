#!/usr/bin/env python3
"""Trace InfiniLM's exact paged decode path and capture BF16 logits."""

import argparse
import ctypes
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--compare", required=True)
    ap.add_argument(
        "--expected-results",
        help="Optional InfiniLM result JSON supplying the sequence that the current runtime must reproduce; first-difference positions still come from --compare.",
    )
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--top-k", type=int, default=100)
    ap.add_argument("--num-blocks", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument(
        "--stop-at-first-diff",
        action="store_true",
        help="Stop each case immediately after its known first-difference step.",
    )
    ap.add_argument(
        "--prenorm-dump-root",
        help="Optional root for per-case pre-final-RMSNorm binary dumps.",
    )
    ap.add_argument(
        "--case-id",
        action="append",
        help="Optionally trace only the named case; repeat for multiple cases.",
    )
    ap.add_argument(
        "--operator-dump-layer",
        type=int,
        help="Override the per-case layer selected for generic operator dumps.",
    )
    ap.add_argument(
        "--gdn-dump-layer",
        type=int,
        help="Enable GDN intermediate dumps for this layer.",
    )
    ap.add_argument(
        "--gdn-dump-seq-len",
        type=int,
        default=1,
        help="Sequence length for GDN intermediate dumps (default: 1).",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--allow-token-mismatch",
        action="store_true",
        help="Diagnostic only: keep output even if selected tokens differ from expected.",
    )
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
        divergent = [
            x for x in json.load(f)["cases"] if x["first_difference"] is not None
        ]
    if args.case_id:
        selected = set(args.case_id)
        divergent = [x for x in divergent if x["id"] in selected]
        missing = selected - {x["id"] for x in divergent}
        if missing:
            raise ValueError("unknown or non-divergent case ids: %s" % sorted(missing))
    expected_by_id = None
    if args.expected_results:
        with open(args.expected_results, encoding="utf-8") as f:
            current = json.load(f)
        expected_by_id = {
            x["id"]: [int(t) for t in x["runs"][0]["tokens"]] for x in current["cases"]
        }
    if len(divergent) >= max(2, args.num_blocks // 4):
        raise ValueError("not enough independent Mamba cache rows")

    started = time.time()
    cache_config = PagedKVCacheConfig(
        args.num_blocks, args.block_size, max_batch_size=1
    )
    engine = InferEngine(
        model_path=args.model_path,
        device=infinicore.device("cuda:0"),
        distributed_config=DistConfig(1),
        cache_config=cache_config,
        attention_backend="paged-attn",
    )
    load_model_state_dict_by_file(engine, args.model_path, dtype=engine.dtype)
    load_s = time.time() - started
    results = []
    operator_dump_layers = {
        "zh_04": 63,
        "zh_06": 0,
        "code_04": 20,
        "math_04": 55,
    }

    for case_index, item in enumerate(divergent):
        case_id = item["id"]
        if args.prenorm_dump_root:
            case_dump_dir = os.path.join(args.prenorm_dump_root, case_id)
            os.makedirs(case_dump_dir, exist_ok=True)
            os.environ["INFINILM_FINAL_PRENORM_DUMP_DIR"] = case_dump_dir
            os.environ["INFINILM_FINAL_PRENORM_DUMP_NUMEL"] = "5120"
            # The final fused add-RMSNorm computes its scale from the unrounded
            # FP32 sum of layer-63 residual and FFN output, then normalizes the
            # BF16 materialized residual. Preserve both inputs for exact replay.
            os.environ["INFINILM_LAYER_DUMP_DIR"] = case_dump_dir
            os.environ["INFINILM_LAYER_DUMP_NUMEL"] = "5120"
            os.environ["INFINILM_OPERATOR_DUMP_LAYER"] = str(
                args.operator_dump_layer
                if args.operator_dump_layer is not None
                else operator_dump_layers.get(case_id, 63)
            )
            if case_id in operator_dump_layers:
                os.environ["INFINILM_ATTENTION_DUMP_DIR"] = case_dump_dir
                os.environ["INFINILM_ATTENTION_DUMP_LAYER"] = str(
                    operator_dump_layers[case_id]
                )
            if args.gdn_dump_layer is not None:
                os.environ["INFINILM_GDN_DUMP_LAYER"] = str(args.gdn_dump_layer)
                os.environ["INFINILM_GDN_DUMP_SEQ_LEN"] = str(args.gdn_dump_seq_len)
        prompt = [int(x) for x in inputs[case_id]["input_ids"]]
        expected = (
            expected_by_id[case_id]
            if expected_by_id is not None
            else [int(x) for x in item["infinilm_tokens"]]
        )
        first_diff = int(item["first_difference"])
        kv_block = case_index
        mamba_row = case_index + 1
        past = 0
        current = prompt
        steps = []
        generated = []
        case_new_tokens = first_diff + 1 if args.stop_at_first_diff else args.new_tokens
        for step in range(case_new_tokens):
            if args.prenorm_dump_root and step == first_diff:
                os.environ["INFINILM_LAYER_DUMP_FIRST_N"] = "64"
            else:
                os.environ.pop("INFINILM_LAYER_DUMP_FIRST_N", None)
            seq_len = len(current)
            total = past + seq_len
            positions = list(range(past, total))
            if engine.position_id_axes > 1:
                positions = [positions for _ in range(engine.position_id_axes)]
            slot_base = kv_block * args.block_size
            slot_mapping = [slot_base + i for i in range(past, total)]
            tensors = {
                "input_ids": infinicore.from_list(
                    [current], dtype=infinicore.int64
                ).view([1, seq_len]),
                "position_ids": infinicore.from_list(positions, dtype=infinicore.int64),
                "past_kv_lengths": infinicore.from_list([past], dtype=infinicore.int32),
                "total_kv_lengths": infinicore.from_list(
                    [total], dtype=infinicore.int32
                ),
                "input_offsets": infinicore.from_list(
                    [0, seq_len], dtype=infinicore.int32
                ),
                "cu_seqlens": infinicore.from_list([0, total], dtype=infinicore.int32),
                "block_tables": infinicore.from_list(
                    [[kv_block]], dtype=infinicore.int32
                ),
                "slot_mapping": infinicore.from_list(
                    slot_mapping, dtype=infinicore.int64
                ),
                "mamba_init_state_indices": infinicore.from_list(
                    [0 if step == 0 else mamba_row], dtype=infinicore.int32
                ),
                "mamba_final_state_indices": infinicore.from_list(
                    [mamba_row], dtype=infinicore.int32
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
            token = int(
                np.asarray(infinicore.Tensor(output.output_ids).to_numpy()).reshape(-1)[
                    0
                ]
            )
            raw = infinicore.Tensor(output.logits)
            shape = list(raw.shape)
            cpu = raw.to(infinicore.device("cpu", 0))
            if cpu.dtype == infinicore.bfloat16:
                bits_type = ctypes.c_uint16 * cpu.numel()
                bits = np.ctypeslib.as_array(
                    bits_type.from_address(cpu.data_ptr())
                ).copy()
                logits = (bits.astype(np.uint32) << 16).view(np.float32).reshape(shape)
            elif cpu.dtype == infinicore.float32:
                logits = (
                    np.ctypeslib.as_array(
                        (ctypes.c_float * cpu.numel()).from_address(cpu.data_ptr())
                    )
                    .copy()
                    .reshape(shape)
                )
            else:
                raise TypeError("expected BF16 or F32 logits, got %s" % cpu.dtype)
            logits = logits.reshape(-1, shape[-1])[-1]
            order = np.argpartition(logits, -args.top_k)[-args.top_k :]
            order = order[np.argsort(logits[order], kind="stable")[::-1]]
            top_logit = float(logits[order[0]])
            candidates = [
                {
                    "id": int(i),
                    "logit": float(logits[i]),
                    "delta_from_top": float(logits[i] - top_logit),
                }
                for i in order
            ]
            step_result = {
                "step": step,
                "selected": token,
                "logits_shape": shape,
                "top_logits": candidates,
            }
            if step == first_diff:
                hidden = infinicore.Tensor(output.hidden_states)
                hidden_shape = list(hidden.shape)
                hidden_cpu = hidden.to(infinicore.device("cpu", 0))
                step_result["hidden_shape"] = hidden_shape
                if hidden_cpu.dtype == infinicore.bfloat16:
                    hidden_bits_type = ctypes.c_uint16 * hidden_cpu.numel()
                    hidden_bits = np.ctypeslib.as_array(
                        hidden_bits_type.from_address(hidden_cpu.data_ptr())
                    ).copy()
                    step_result["hidden_dtype"] = "bfloat16"
                    step_result["hidden_bf16_bits"] = [int(x) for x in hidden_bits]
                elif hidden_cpu.dtype == infinicore.float32:
                    hidden_values = np.ctypeslib.as_array(
                        (ctypes.c_float * hidden_cpu.numel()).from_address(
                            hidden_cpu.data_ptr()
                        )
                    ).copy()
                    step_result["hidden_dtype"] = "float32"
                    step_result["hidden_f32"] = [float(x) for x in hidden_values]
                else:
                    raise TypeError(
                        "expected BF16 or F32 hidden state, got %s" % hidden_cpu.dtype
                    )
            steps.append(step_result)
            generated.append(token)
            current = [token]
            past = total
        expected = expected[:case_new_tokens]
        stable = generated == expected
        if not stable and not args.allow_token_mismatch:
            raise RuntimeError(
                "%s trace changed: %s != %s" % (case_id, generated, expected)
            )
        results.append({"case_id": case_id, "tokens": generated, "steps": steps})
        print(
            "%-10s tokens=%d stable=%s" % (case_id, len(generated), stable), flush=True
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(
            {"load_s": round(load_s, 4), "cases": results},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("RESULT cases=%d load=%.3fs" % (len(results), load_s), flush=True)


if __name__ == "__main__":
    main()
