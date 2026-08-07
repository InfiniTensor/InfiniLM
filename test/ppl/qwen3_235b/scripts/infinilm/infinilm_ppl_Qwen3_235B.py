#!/usr/bin/env python3
"""Calculate true token-level PPL with the current InfiniLM C++ TP engine."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _gpu_guard import require_idle_gpu
from _ppl_common import (
    SCORING_METHOD,
    canonical_indices_sha256,
    iter_sliding_windows,
    load_manifest,
)


RESULT_SCHEMA = "qwen3_235b_true_ppl_result/v1"
DEFAULT_MODEL = "/data1/Qwen3_235B"
EXPECTED_MODEL_TYPE = "qwen3_moe"
EXPECTED_VOCAB_SIZE = 151936
PAGED_KV_BLOCK_SIZE = 256


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="True shifted-token PPL for Qwen3_235B with InfiniLM TP8"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--token-manifest", required=True)
    parser.add_argument("--window", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument(
        "--max-scored-tokens",
        type=int,
        default=10240,
        help="maximum target tokens to score; 0 scores the full manifest",
    )
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--attention", default="flash-attn")
    parser.add_argument("--json-output")
    args = parser.parse_args(argv)

    if not Path(args.model).is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    if not Path(args.token_manifest).is_file():
        parser.error(f"token manifest does not exist: {args.token_manifest}")
    if args.window < 2:
        parser.error("--window must be at least 2")
    if args.stride < 1 or args.stride >= args.window:
        parser.error("--stride must satisfy 1 <= stride < window")
    if args.max_scored_tokens < 0:
        parser.error("--max-scored-tokens must be non-negative")
    if args.tp_size < 1:
        parser.error("--tp-size must be positive")

    args.model = str(Path(args.model).resolve())
    args.token_manifest = str(Path(args.token_manifest).resolve())
    if args.json_output:
        args.json_output = str(Path(args.json_output).resolve())
    return args


def _atomic_json(path_value: str, payload: dict[str, Any]) -> None:
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_model_config(model_path: str) -> dict[str, Any]:
    config_path = Path(model_path) / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read model config {config_path}: {error}") from error
    if not isinstance(config, dict):
        raise RuntimeError(f"model config must be an object: {config_path}")
    if config.get("model_type") != EXPECTED_MODEL_TYPE:
        raise RuntimeError(
            f"expected model_type={EXPECTED_MODEL_TYPE!r}, "
            f"got {config.get('model_type')!r}"
        )
    if int(config.get("vocab_size", 0)) != EXPECTED_VOCAB_SIZE:
        raise RuntimeError(
            f"expected vocab_size={EXPECTED_VOCAB_SIZE}, "
            f"got {config.get('vocab_size')!r}"
        )
    return config


def _is_quantized(config: dict[str, Any]) -> bool:
    quantization = config.get("quantization_config")
    return isinstance(quantization, dict) and bool(quantization)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import infinicore
    from infinilm.cache import PagedKVCacheConfig
    from infinilm.distributed import DistConfig
    from infinilm.infer_engine import InferEngine
    from infinilm.modeling_utils import load_model_state_dict_by_file

    corpus = load_manifest(args.token_manifest)
    model_config = _read_model_config(args.model)
    if any(token >= EXPECTED_VOCAB_SIZE for token in corpus.token_ids):
        raise RuntimeError("token manifest contains an ID outside the model vocabulary")

    available_targets = corpus.token_count - 1
    scored_token_count = (
        available_targets
        if args.max_scored_tokens == 0
        else min(args.max_scored_tokens, available_targets)
    )
    max_targets = None if args.max_scored_tokens == 0 else scored_token_count
    windows = list(
        iter_sliding_windows(
            corpus.token_ids,
            args.window,
            args.stride,
            max_targets,
        )
    )
    if sum(window.scored_token_count for window in windows) != scored_token_count:
        raise RuntimeError("sliding-window plan does not match scored token count")

    first_scored_token_index = 1
    last_scored_token_index_exclusive = 1 + scored_token_count
    indices_sha256 = canonical_indices_sha256(
        range(first_scored_token_index, last_scored_token_index_exclusive)
    )
    precision = "W8A8" if _is_quantized(model_config) else "BF16"
    config_payload = {
        "backend": "infinilm",
        "model": args.model,
        "precision": precision,
        "tp_size": args.tp_size,
        "attention": args.attention,
        "graph_enabled": False,
        "window_size": args.window,
        "stride": args.stride,
        "scored_token_count": scored_token_count,
        "scoring_method": SCORING_METHOD,
        "corpus_manifest_sha256": corpus.manifest_sha256,
        "corpus_token_ids_sha256": corpus.token_ids_sha256,
    }
    print(
        "INFINILM_QWEN3_235B_PPL_CONFIG "
        + json.dumps(config_payload, ensure_ascii=False, sort_keys=True),
        flush=True,
    )

    device = infinicore.device("cuda", 0)
    load_start = time.perf_counter()
    model = InferEngine(
        args.model,
        device=device,
        distributed_config=DistConfig(args.tp_size),
        # This InfiniLM branch's flash-attn backend consumes the paged KV-cache
        # layout while retaining flash-attn as the attention implementation.
        cache_config=PagedKVCacheConfig(
            num_blocks=(args.window + PAGED_KV_BLOCK_SIZE - 1)
            // PAGED_KV_BLOCK_SIZE,
            block_size=PAGED_KV_BLOCK_SIZE,
        ),
        enable_graph_compiling=False,
        attention_backend=args.attention,
    )
    if not hasattr(model, "score_nll"):
        raise RuntimeError(
            "installed InfiniLM lacks InferEngine.score_nll; rebuild the PPL scoring patch"
        )
    load_model_state_dict_by_file(model, args.model, dtype=model.dtype)
    model_load_seconds = time.perf_counter() - load_start

    window_nll_values: list[float] = []
    window_results: list[dict[str, Any]] = []
    infinicore.sync_device()
    scoring_start = time.perf_counter()
    for window in windows:
        input_tokens = list(window.token_ids[:-1])
        label_tokens = list(window.token_ids[1:])
        if not input_tokens or len(input_tokens) != len(label_tokens):
            raise RuntimeError(f"invalid causal shift in window {window.index}")
        input_ids = infinicore.from_list(
            [input_tokens], dtype=infinicore.int64
        )
        labels = infinicore.from_list(
            [label_tokens], dtype=infinicore.int64
        )
        nll, returned_tokens = model.score_nll(
            input_ids,
            labels,
            score_start=window.prediction_start,
        )
        if returned_tokens != window.scored_token_count:
            raise RuntimeError(
                f"window {window.index} scored {returned_tokens} tokens, "
                f"expected {window.scored_token_count}"
            )
        if not math.isfinite(nll) or nll < 0:
            raise RuntimeError(f"window {window.index} returned invalid NLL {nll}")
        window_nll_values.append(nll)
        window_results.append(
            {
                "index": window.index,
                "context_start": window.token_start,
                "target_start": window.score_start,
                "target_end": window.score_end,
                "input_token_count": len(window.token_ids),
                "scored_token_count": returned_tokens,
                "nll": nll,
            }
        )
        print(
            f"PPL window {window.index + 1}/{len(windows)} "
            f"tokens={returned_tokens} nll={nll:.6f}",
            flush=True,
        )

    infinicore.sync_device()
    scoring_seconds = time.perf_counter() - scoring_start
    total_nll = math.fsum(window_nll_values)
    mean_nll = total_nll / scored_token_count
    try:
        ppl = math.exp(mean_nll)
    except OverflowError as error:
        raise RuntimeError(f"PPL overflow at mean NLL={mean_nll}") from error
    if not math.isfinite(ppl):
        raise RuntimeError(f"PPL is not finite: {ppl}")

    result = {
        "schema": RESULT_SCHEMA,
        "status": "PASS",
        "backend": "infinilm",
        "model": args.model,
        "precision": precision,
        "tp_size": args.tp_size,
        "attention": args.attention,
        "graph_enabled": False,
        "corpus_manifest": args.token_manifest,
        "corpus_manifest_sha256": corpus.manifest_sha256,
        "corpus_token_ids_sha256": corpus.token_ids_sha256,
        "corpus_token_count": corpus.token_count,
        "window_size": args.window,
        "stride": args.stride,
        "scoring_method": SCORING_METHOD,
        "first_scored_token_index": first_scored_token_index,
        "last_scored_token_index_exclusive": last_scored_token_index_exclusive,
        "scored_token_indices_sha256": indices_sha256,
        "scored_token_count": scored_token_count,
        "total_nll": total_nll,
        "mean_nll": mean_nll,
        "ppl": ppl,
        "windows": window_results,
        "window_count": len(window_results),
        "scoring_seconds": scoring_seconds,
        "scored_tokens_per_second": scored_token_count / scoring_seconds,
        "model_load_seconds": model_load_seconds,
        "vocab_size": EXPECTED_VOCAB_SIZE,
    }
    if args.json_output:
        _atomic_json(args.json_output, result)
    print(
        "INFINILM_QWEN3_235B_PPL_RESULT "
        + json.dumps(result, ensure_ascii=False, sort_keys=True),
        flush=True,
    )
    print(
        f"InfiniLM {precision} true PPL: {ppl:.6f} "
        f"(mean NLL={mean_nll:.6f}, tokens={scored_token_count})",
        flush=True,
    )

    del model
    gc.collect()
    infinicore.sync_device()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        # Validate the workload before checking or reserving GPUs.
        load_manifest(args.token_manifest)
        require_idle_gpu()
        _run(args)
    except BaseException as error:
        completion = {
            "schema": RESULT_SCHEMA,
            "status": "ERROR",
            "exit_code": 1,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
        print(
            "INFINILM_QWEN3_235B_PPL_COMPLETE "
            + json.dumps(completion, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
        raise
    print(
        "INFINILM_QWEN3_235B_PPL_COMPLETE "
        + json.dumps(
            {"schema": RESULT_SCHEMA, "status": "PASS", "exit_code": 0},
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
