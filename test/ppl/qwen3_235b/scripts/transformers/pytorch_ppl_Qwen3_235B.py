#!/usr/bin/env python3
"""Calculate true token-level PPL for Qwen3_235B with Transformers TP8.

The input is a framework-neutral token manifest.  Both the Transformers and
InfiniLM runners must consume the same manifest so their PPL values score the
same target tokens instead of independently tokenizing the source corpus.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
for import_path in (SCRIPT_DIR, SCRIPTS_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import _pytorch_runner as benchmark_runner
from _ppl_common import (
    SCORING_METHOD,
    canonical_indices_sha256,
    iter_sliding_windows,
    load_manifest,
)


RESULT_SCHEMA = "qwen3_235b_true_ppl_result/v1"
DEFAULT_MODEL = "/data1/Qwen3_235B"
DEFAULT_WINDOW_SIZE = 256
DEFAULT_STRIDE = 128
DEFAULT_MAX_SCORED_TOKENS = 10240
EXPECTED_VOCAB_SIZE = 151936

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="True shifted-token PPL for Qwen3_235B BF16 on Hygon TP8"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--token-manifest", required=True)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument(
        "--max-scored-tokens",
        type=int,
        default=DEFAULT_MAX_SCORED_TOKENS,
        help="maximum shifted target tokens to score; 0 scores the full manifest",
    )
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument(
        "--attention",
        choices=("eager",),
        default="eager",
        help="BW1100 correctness path; SDPA is unsupported for this model stack",
    )
    parser.add_argument(
        "--json-output",
        help="optional rank-0 result path; the result is always printed as JSON",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    manifest_path = Path(args.token_manifest)
    if not model_path.is_dir():
        parser.error(f"model directory does not exist: {model_path}")
    if not manifest_path.is_file():
        parser.error(f"token manifest does not exist: {manifest_path}")
    if args.window < 2:
        parser.error("--window must be at least 2")
    if args.stride < 1 or args.stride >= args.window:
        parser.error("--stride must satisfy 1 <= stride < window")
    if args.max_scored_tokens < 0:
        parser.error("--max-scored-tokens must be non-negative")
    if args.tp_size < 1:
        parser.error("--tp-size must be positive")

    args.model = str(model_path.resolve())
    args.token_manifest = str(manifest_path.resolve())
    if args.json_output:
        args.json_output = str(Path(args.json_output).resolve())
    return args

def _write_json_atomic(path_value: str, payload: dict[str, Any]) -> None:
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_worker_impl(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as functional
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", world_size))
    if world_size != args.tp_size or local_world_size != args.tp_size:
        raise RuntimeError(
            "PPL runner requires one-host TP with WORLD_SIZE="
            f"LOCAL_WORLD_SIZE={args.tp_size}; got {world_size}/{local_world_size}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() < local_world_size:
        raise RuntimeError(
            f"need {local_world_size} visible GPUs, found {torch.cuda.device_count()}"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    # RCCL initializes lazily. Reserve communicator memory before the 235B
    # checkpoint consumes nearly all device memory.
    communicator_probe = torch.ones(1, dtype=torch.int32, device=device)
    dist.all_reduce(communicator_probe)
    if int(communicator_probe.item()) != args.tp_size:
        raise RuntimeError("Transformers TP8 RCCL communicator probe failed")
    dist.barrier()
    torch.cuda.synchronize(device)
    del communicator_probe
    torch.random.default_generator.manual_seed(0)
    torch.cuda.manual_seed(0)
    if rank != 0:
        transformers.utils.logging.disable_progress_bar()

    corpus_manifest = load_manifest(args.token_manifest)
    token_ids = corpus_manifest.token_ids
    corpus = {
        "manifest_path": str(corpus_manifest.path.resolve()),
        "manifest_sha256": corpus_manifest.manifest_sha256,
        "token_ids_sha256": corpus_manifest.token_ids_sha256,
        "token_count": corpus_manifest.token_count,
        "source_sha256": corpus_manifest.payload["source_sha256"],
        "tokenizer_sha256": corpus_manifest.payload["tokenizer_sha256"],
        "source": corpus_manifest.payload.get("source"),
        "tokenizer": corpus_manifest.payload.get("tokenizer"),
    }
    model_config = AutoConfig.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    architecture = benchmark_runner._validate_qwen3_235b_architecture(model_config)
    if getattr(model_config, "quantization_config", None):
        raise RuntimeError("Transformers PPL baseline requires the BF16 checkpoint")
    vocab_size = int(model_config.vocab_size)
    if vocab_size != EXPECTED_VOCAB_SIZE:
        raise RuntimeError(
            f"expected complete Qwen3_235B vocabulary {EXPECTED_VOCAB_SIZE}, "
            f"got {vocab_size}"
        )
    if any(token >= vocab_size for token in token_ids):
        raise RuntimeError("token manifest contains an ID outside the model vocabulary")
    maximum_positions = int(
        getattr(model_config, "max_position_embeddings", 0) or 0
    )
    if maximum_positions and args.window > maximum_positions:
        raise RuntimeError(
            f"window={args.window} exceeds max_position_embeddings={maximum_positions}"
        )

    scoring_scenario = benchmark_runner.Scenario("true_ppl", 1, args.window, 1)
    tp_plan, tp_metadata = benchmark_runner._build_qwen3_moe_tp_plan(
        model_config, args.tp_size, scoring_scenario, 1
    )
    grouped_mm_fallback = benchmark_runner._install_hygon_grouped_mm_guard(torch)
    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=model_config,
        dtype=torch.bfloat16,
        attn_implementation=args.attention,
        tp_plan=tp_plan,
        local_files_only=True,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    tp_validation = benchmark_runner._validate_and_set_local_gqa(model, tp_metadata)
    model.eval()
    torch.cuda.synchronize(device)
    load_seconds = benchmark_runner._max_across_ranks(
        time.perf_counter() - load_start, torch, dist, device
    )

    loaded_tp_plan = getattr(model, "_tp_plan", None)
    if not loaded_tp_plan:
        raise RuntimeError("model loaded without a non-empty Transformers TP plan")
    resolved_attention = getattr(model.config, "_attn_implementation", None)
    if resolved_attention != args.attention:
        raise RuntimeError(
            f"requested attention={args.attention!r}, resolved={resolved_attention!r}"
        )
    forward_parameters = inspect.signature(model.forward).parameters
    if "logits_to_keep" in forward_parameters:
        logits_limit_argument = "logits_to_keep"
    elif "num_logits_to_keep" in forward_parameters:
        logits_limit_argument = "num_logits_to_keep"
    else:
        raise RuntimeError(
            "model.forward has no logits_to_keep argument; refusing full-context "
            "vocabulary materialization"
        )

    available_targets = len(token_ids) - 1
    scored_token_count = (
        available_targets
        if args.max_scored_tokens == 0
        else min(args.max_scored_tokens, available_targets)
    )
    if scored_token_count < 1:
        raise RuntimeError("the selected corpus range contains no shifted target token")
    first_scored_token_index = 1
    last_scored_token_index_exclusive = 1 + scored_token_count
    scored_token_indices_sha256 = canonical_indices_sha256(
        range(first_scored_token_index, last_scored_token_index_exclusive)
    )
    scoring_method = SCORING_METHOD

    config = {
        "backend": "transformers",
        "model": args.model,
        "dtype": "bfloat16",
        "tp_size": args.tp_size,
        "attention": resolved_attention,
        "window_size": args.window,
        "stride": args.stride,
        "requested_max_scored_tokens": args.max_scored_tokens,
        "scored_token_count": scored_token_count,
        "scoring_method": scoring_method,
        "first_scored_token_index": first_scored_token_index,
        "last_scored_token_index_exclusive": last_scored_token_index_exclusive,
        "scored_token_indices_sha256": scored_token_indices_sha256,
        "corpus_manifest": corpus,
        "vocab_size": vocab_size,
        "architecture": architecture,
        "model_load_seconds": load_seconds,
        "tp_plan": tp_metadata,
        "tp_validation": tp_validation,
        "hygon_transformers_grouped_mm_fallback": grouped_mm_fallback,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
    }
    if rank == 0:
        print(
            "PYTORCH_QWEN3_235B_PPL_CONFIG "
            + json.dumps(config, ensure_ascii=False, sort_keys=True),
            flush=True,
        )

    total_nll = 0.0
    windows: list[dict[str, Any]] = []
    scored_by_windows = 0
    torch.cuda.synchronize(device)
    scoring_start = time.perf_counter()
    with torch.inference_mode():
        for window in iter_sliding_windows(
            token_ids,
            args.window,
            args.stride,
            None if args.max_scored_tokens == 0 else args.max_scored_tokens,
        ):
            display_index = window.index + 1
            target_count = window.scored_token_count
            input_slice = window.token_ids
            expected_prediction_start = len(input_slice) - target_count - 1
            if (
                window.prediction_start != expected_prediction_start
                or window.prediction_end != len(input_slice) - 1
            ):
                raise RuntimeError(
                    f"window {display_index} retained-logits alignment is invalid"
                )
            input_ids = torch.tensor(
                input_slice, dtype=torch.long, device=device
            ).unsqueeze(0)
            outputs = model(
                input_ids=input_ids,
                use_cache=False,
                return_dict=True,
                **{logits_limit_argument: target_count + 1},
            )
            logits = benchmark_runner._materialize_logits(outputs.logits)
            expected_shape = (1, target_count + 1, vocab_size)
            if tuple(logits.shape) != expected_shape:
                raise RuntimeError(
                    "incomplete or unexpected logits: "
                    f"got {tuple(logits.shape)}, expected {expected_shape}"
                )
            score_logits = logits[:, :-1, :]
            labels = torch.tensor(
                input_slice[window.target_start : window.target_end],
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            finite = torch.isfinite(score_logits).all().to(dtype=torch.int32)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not bool(finite.item()):
                raise RuntimeError(
                    f"window {display_index} contains non-finite logits"
                )
            window_nll_tensor = functional.cross_entropy(
                score_logits.float().reshape(-1, vocab_size),
                labels.reshape(-1),
                reduction="sum",
            )
            window_nll = float(window_nll_tensor.double().item())
            nll_min = torch.tensor(window_nll, dtype=torch.float64, device=device)
            nll_max = nll_min.clone()
            dist.all_reduce(nll_min, op=dist.ReduceOp.MIN)
            dist.all_reduce(nll_max, op=dist.ReduceOp.MAX)
            rank_delta_per_token = float((nll_max - nll_min).item()) / target_count
            if rank_delta_per_token > 1e-4:
                raise RuntimeError(
                    f"window {display_index} rank NLL mismatch: "
                    f"delta/token={rank_delta_per_token:.6g}"
                )
            total_nll += window_nll
            scored_by_windows += target_count
            windows.append(
                {
                    "index": window.index,
                    "token_start": window.token_start,
                    "token_end": window.token_end,
                    "score_start": window.score_start,
                    "score_end": window.score_end,
                    "input_token_count": len(input_slice),
                    "scored_token_count": target_count,
                    "nll": window_nll,
                }
            )
            del outputs, logits, score_logits, labels, window_nll_tensor, input_ids

    torch.cuda.synchronize(device)
    scoring_seconds = benchmark_runner._max_across_ranks(
        time.perf_counter() - scoring_start, torch, dist, device
    )
    if scored_by_windows != scored_token_count:
        raise RuntimeError(
            f"scored {scored_by_windows} tokens, expected {scored_token_count}"
        )

    mean_nll = total_nll / scored_token_count
    if not math.isfinite(mean_nll):
        raise RuntimeError(f"mean NLL is not finite: {mean_nll}")
    try:
        ppl = math.exp(mean_nll)
    except OverflowError as error:
        raise RuntimeError(f"PPL overflows float64 at mean NLL={mean_nll}") from error
    if not math.isfinite(ppl):
        raise RuntimeError(f"PPL is not finite: {ppl}")

    result: dict[str, Any] = {}
    if rank == 0:
        result = {
            "schema": RESULT_SCHEMA,
            "status": "PASS",
            "backend": "transformers",
            "model": args.model,
            "dtype": "bfloat16",
            "tp_size": args.tp_size,
            "attention": resolved_attention,
            "corpus_manifest": args.token_manifest,
            "corpus_manifest_sha256": corpus["manifest_sha256"],
            "corpus_token_ids_sha256": corpus["token_ids_sha256"],
            "corpus_token_count": corpus["token_count"],
            "window_size": args.window,
            "stride": args.stride,
            "scoring_method": scoring_method,
            "first_scored_token_index": first_scored_token_index,
            "last_scored_token_index_exclusive": (
                last_scored_token_index_exclusive
            ),
            "scored_token_indices_sha256": scored_token_indices_sha256,
            "scored_token_count": scored_token_count,
            "total_nll": total_nll,
            "mean_nll": mean_nll,
            "ppl": ppl,
            "windows": windows,
            "window_count": len(windows),
            "scoring_seconds": scoring_seconds,
            "scored_tokens_per_second": scored_token_count / scoring_seconds,
            "model_load_seconds": load_seconds,
            "vocab_size": vocab_size,
            "full_vocab_logits_validated_every_window": True,
        }
        if args.json_output:
            _write_json_atomic(args.json_output, result)
        print(
            "PYTORCH_QWEN3_235B_PPL_RESULT "
            + json.dumps(result, ensure_ascii=False, sort_keys=True),
            flush=True,
        )
        print(
            f"Transformers true PPL: {ppl:.6f} "
            f"(mean NLL={mean_nll:.6f}, tokens={scored_token_count})",
            flush=True,
        )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _run_worker(args: argparse.Namespace) -> int:
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    caught: BaseException | None = None
    caught_traceback: Any = None
    teardown_errors: list[str] = []
    try:
        _run_worker_impl(args)
    except BaseException as error:
        caught = error
        caught_traceback = error.__traceback__
    finally:
        initialized = dist.is_initialized()
        if initialized:
            if caught is None:
                try:
                    dist.barrier()
                except BaseException as error:
                    caught = error
                    caught_traceback = error.__traceback__
                    teardown_errors.append(
                        f"barrier: {type(error).__name__}: {error}"
                    )
            try:
                dist.destroy_process_group()
            except BaseException as error:
                if caught is None:
                    caught = error
                    caught_traceback = error.__traceback__
                teardown_errors.append(
                    f"destroy_process_group: {type(error).__name__}: {error}"
                )
        status = "PASS" if caught is None and not teardown_errors else "ERROR"
        if rank == 0:
            completion: dict[str, Any] = {
                "schema": RESULT_SCHEMA,
                "status": status,
                "exit_code": 0 if status == "PASS" else 1,
                "distributed_teardown_complete": not dist.is_initialized(),
            }
            if caught is not None:
                completion["error"] = {
                    "type": type(caught).__name__,
                    "message": str(caught),
                }
            if teardown_errors:
                completion["teardown_errors"] = teardown_errors
            print(
                "PYTORCH_QWEN3_235B_PPL_COMPLETE "
                + json.dumps(completion, ensure_ascii=False, sort_keys=True),
                flush=True,
            )
    if caught is not None:
        raise caught.with_traceback(caught_traceback)
    return 0


def main() -> int:
    args = _parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if "LOCAL_RANK" not in os.environ and world_size == 1:
        # Fail on corpus/schema errors before reserving all eight devices.
        load_manifest(args.token_manifest)
        benchmark_runner._require_idle_gpu()
        benchmark_runner._launch_torchrun(args)
        raise AssertionError("os.execvpe returned unexpectedly")
    return _run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
