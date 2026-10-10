#!/usr/bin/env python3
"""Dump first-token argmax / top-5 logits for MiniCPM5-2B on QY.

Runs one prefill forward (InferEngine) with static attn (graph off) and
optionally flash-attn + paged KV. Writes JSON under --out-dir.

Tensor→numpy patterns follow test/models/llama/test_llama_inference.py
(via InfiniCore Tensor.to_numpy). Does not wire HookRegistry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default=os.environ.get("MODEL_DIR", "/models/minicpm5-2b"),
        help="HF model directory",
    )
    p.add_argument("--device", default="qy", help="InfiniCore device type")
    p.add_argument("--device-index", type=int, default=0)
    p.add_argument(
        "--prompt",
        default="你好",
        help="User prompt (chat-templated like the OpenAI server)",
    )
    p.add_argument(
        "--modes",
        default="static,fa",
        help="Comma-separated modes: static and/or fa",
    )
    p.add_argument(
        "--out-dir",
        default=os.environ.get(
            "OUT_DIR",
            str(
                Path(__file__).resolve().parents[2]
                / "logs"
            ),
        ),
        help="Directory for m1-2b-first-token-{mode}.json",
    )
    p.add_argument("--max-cache-len", type=int, default=4096)
    p.add_argument("--num-blocks", type=int, default=64)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--top-k", type=int, default=5)
    return p.parse_args()


def _logits_hash(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()[:16]


def _top_k(logits_1d: np.ndarray, k: int) -> List[Dict[str, Any]]:
    k = min(k, int(logits_1d.shape[0]))
    idx = np.argpartition(logits_1d, -k)[-k:]
    idx = idx[np.argsort(-logits_1d[idx])]
    return [
        {"token_id": int(i), "logit": float(logits_1d[i])} for i in idx
    ]


def _tokenize_chat(model_dir: str, user_prompt: str) -> Tuple[Any, List[int], str]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    messages = [{"role": "user", "content": user_prompt}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer.encode(rendered, add_special_tokens=False)
    return tokenizer, input_ids, rendered


def _build_engine(
    model_path: str,
    device,
    *,
    mode: str,
    max_cache_len: int,
    num_blocks: int,
    block_size: int,
):
    from infinilm.cache.cache import PagedKVCacheConfig, StaticKVCacheConfig
    from infinilm.distributed import DistConfig
    from infinilm.infer_engine import InferEngine
    from infinilm.modeling_utils import load_model_state_dict_by_file

    if mode == "static":
        cache_config = StaticKVCacheConfig(
            max_batch_size=1, max_cache_len=max_cache_len
        )
        attention_backend = "default"
        enable_graph = False
    elif mode == "fa":
        cache_config = PagedKVCacheConfig(
            num_blocks=num_blocks,
            block_size=block_size,
            max_batch_size=1,
        )
        attention_backend = "flash-attn"
        enable_graph = False
    else:
        raise ValueError(f"unknown mode: {mode}")

    engine = InferEngine(
        model_path=model_path,
        device=device,
        distributed_config=DistConfig(1),
        cache_config=cache_config,
        enable_graph_compiling=enable_graph,
        attention_backend=attention_backend,
        weight_load_mode="sync",
    )
    load_model_state_dict_by_file(engine, model_path, dtype=engine.dtype)
    return engine


def _tensor_to_f32_numpy(tensor) -> np.ndarray:
    """InfiniCore→numpy; bf16 has no numpy dtype so go through torch."""
    import ctypes

    import infinicore
    import torch

    source = tensor
    if source.device.type != "cpu":
        source = source.to(infinicore.device("cpu", 0))
        try:
            infinicore.sync_device()
        except Exception:
            pass
    if not source.is_contiguous():
        source = source.contiguous()

    shape = list(source.shape)
    dtype_name = str(source.dtype)
    if "bfloat16" in dtype_name:
        torch_t = torch.empty(shape, dtype=torch.bfloat16)
        if torch_t.numel():
            ctypes.memmove(torch_t.data_ptr(), source.data_ptr(), torch_t.numel() * 2)
        return torch_t.float().numpy()
    if "float16" in dtype_name or dtype_name.endswith("f16"):
        torch_t = torch.empty(shape, dtype=torch.float16)
        if torch_t.numel():
            ctypes.memmove(torch_t.data_ptr(), source.data_ptr(), torch_t.numel() * 2)
        return torch_t.float().numpy()
    return np.asarray(source.to_numpy(), dtype=np.float32)


def _prefill_logits(engine, input_ids: List[int], mode: str, block_size: int):
    import infinicore

    seq_len = len(input_ids)
    batch_size = 1
    past_seq_len = 0

    ids = infinicore.from_list([input_ids], dtype=infinicore.int64)
    position_ids = infinicore.from_list(
        [list(range(past_seq_len, past_seq_len + seq_len))],
        dtype=infinicore.int64,
    )
    past_kv_lengths = infinicore.from_list(
        [past_seq_len] * batch_size, dtype=infinicore.int32
    )
    total_kv_lengths = infinicore.from_list(
        [past_seq_len + seq_len] * batch_size, dtype=infinicore.int32
    )
    cu_seqlens = infinicore.from_list(
        [(past_seq_len + seq_len) * i for i in range(batch_size + 1)],
        dtype=infinicore.int32,
    )
    input_offsets = infinicore.from_list(
        [seq_len * i for i in range(batch_size + 1)],
        dtype=infinicore.int32,
    )

    block_tables = None
    slot_mapping = None
    if mode == "fa":
        # Match InferEngine.generate paged prefill bookkeeping (batch=1).
        max_blocks = (seq_len + block_size - 1) // block_size
        max_blocks = max(max_blocks, 1)
        block_tables = infinicore.from_list(
            [list(range(max_blocks))], dtype=infinicore.int32
        )
        slot_mapping = infinicore.from_list(
            list(range(seq_len)), dtype=infinicore.int64
        )
        ids = ids.view([1, seq_len])

    out = engine.forward_raw(
        input_ids=ids,
        position_ids=position_ids,
        past_kv_lengths=past_kv_lengths,
        total_kv_lengths=total_kv_lengths,
        input_offsets=input_offsets,
        cu_seqlens=cu_seqlens,
        block_tables=block_tables,
        slot_mapping=slot_mapping,
        sample_all_positions=True,
    )
    arr = _tensor_to_f32_numpy(out["logits"])
    # Expected [B, T, V] or [B, 1, V] (last token only).
    if arr.ndim == 3:
        last = arr[0, -1, :]
    elif arr.ndim == 2:
        last = arr[-1, :]
    else:
        raise RuntimeError(f"unexpected logits shape: {arr.shape}")
    return arr, last.astype(np.float32, copy=False)


def _run_mode(args: argparse.Namespace, mode: str) -> Dict[str, Any]:
    import infinicore

    record: Dict[str, Any] = {
        "mode": mode,
        "model": args.model,
        "device": args.device,
        "prompt": args.prompt,
        "status": "ok",
    }
    try:
        tokenizer, input_ids, rendered = _tokenize_chat(args.model, args.prompt)
        record["rendered_prompt"] = rendered
        record["input_ids"] = input_ids
        record["num_prompt_tokens"] = len(input_ids)

        device = infinicore.device(args.device, args.device_index)
        engine = _build_engine(
            args.model,
            device,
            mode=mode,
            max_cache_len=args.max_cache_len,
            num_blocks=args.num_blocks,
            block_size=args.block_size,
        )
        full_logits, last = _prefill_logits(
            engine, input_ids, mode=mode, block_size=args.block_size
        )
        argmax_id = int(np.argmax(last))
        top = _top_k(last, args.top_k)
        # Attach decoded text for readability.
        for item in top:
            item["token"] = tokenizer.decode(
                [item["token_id"]], skip_special_tokens=False
            )
        record.update(
            {
                "argmax": argmax_id,
                "argmax_token": tokenizer.decode(
                    [argmax_id], skip_special_tokens=False
                ),
                "top5": top,
                "logits_shape": list(full_logits.shape),
                "logits_dtype": str(full_logits.dtype),
                "logits_hash": _logits_hash(last),
                "logits_l2": float(np.linalg.norm(last)),
                "logits_stats": {
                    "min": float(last.min()),
                    "max": float(last.max()),
                    "mean": float(last.mean()),
                },
            }
        )
    except Exception as e:
        record["status"] = "error"
        record["error"] = f"{type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc()
    return record


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    rc = 0
    for mode in modes:
        print(f"=== mode={mode} ===", flush=True)
        record = _run_mode(args, mode)
        out_path = out_dir / f"m1-2b-first-token-{mode}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"wrote {out_path} status={record.get('status')}", flush=True)
        if record.get("status") == "ok":
            print(
                f"  argmax={record['argmax']} ({record['argmax_token']!r}) "
                f"hash={record['logits_hash']} l2={record['logits_l2']:.4f}",
                flush=True,
            )
            for i, item in enumerate(record["top5"]):
                print(
                    f"  top{i+1}: id={item['token_id']} "
                    f"logit={item['logit']:.6f} token={item['token']!r}",
                    flush=True,
                )
        else:
            print(f"  ERROR: {record.get('error')}", flush=True)
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
