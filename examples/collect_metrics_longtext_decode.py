#!/usr/bin/env python3
"""
Collect long-context + decode metrics for metrics_longtext_mem.md.

**OOM-safe workflow:** run each case in a **fresh Python process** so CUDA allocations
are released between runs:

  ./run_longtext_metrics_cases.sh

Or manually:

  python3 collect_metrics_longtext_decode.py --case hf:16384 --append-jsonl profiling_runs/longtext_decode_rows.jsonl

See also docstring at top of previous revisions for GPU selection (CUDA_VISIBLE_DEVICES + NVML_GPU_INDEX).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python"))
try:
    from infllmv2_loader import preload_infllmv2_if_available

    preload_infllmv2_if_available()
except Exception:
    pass


def _poll_gpu_mem_mib(stop: threading.Event, gpu_index: int, out: List[int]) -> None:
    while not stop.is_set():
        try:
            r = subprocess.run(
                [
                    "nvidia-smi",
                    "-i",
                    str(gpu_index),
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.returncode == 0 and r.stdout.strip().isdigit():
                out.append(int(r.stdout.strip()))
        except Exception:
            pass
        if stop.wait(timeout=1.0):
            break


def _with_mem_poll(gpu_index: int, fn: Callable[[], Any]) -> Tuple[Any, Optional[int]]:
    samples: List[int] = []
    stop = threading.Event()
    th = threading.Thread(target=_poll_gpu_mem_mib, args=(stop, gpu_index, samples), daemon=True)
    th.start()
    err: Optional[BaseException] = None
    result: Any = None
    try:
        result = fn()
    except BaseException as e:
        err = e
    finally:
        stop.set()
        th.join(timeout=3.0)
    peak = max(samples) if samples else None
    if err is not None:
        raise err
    return result, peak


def _row_dict(
    date: str,
    backend: str,
    target: int,
    actual: int,
    max_new: int,
    peak: Optional[int],
    gpu_smi: int,
    r: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "date": date,
        "backend": backend,
        "target_input_tokens": target,
        "actual_input_tokens": actual,
        "max_new_tokens": max_new,
        "peak_mem_mib": peak,
        "gpu_smi_index": gpu_smi,
        "total_time_ms": r.get("total_time_ms"),
        "prefill_ttft_ms": r.get("prefill_ttft_ms"),
        "prefill_throughput_tok_s": r.get("prefill_throughput_tok_s"),
        "decode_itl_ms": r.get("decode_itl_ms"),
        "decode_throughput_tok_s": r.get("decode_throughput_tok_s"),
        "engine_reported_generation_ms": r.get("engine_reported_generation_ms"),
        "error": r.get("error"),
    }


def run_single_case(
    case: str,
    *,
    model_path: str,
    gpu_smi: int,
    date: str,
) -> Dict[str, Any]:
    """Run one measurement; returns a row dict (may contain error key)."""
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, examples_dir)
    os.chdir(examples_dir)

    from transformers import AutoTokenizer

    from compare_inference_speed import (
        _make_prompt_with_target_tokens,
        run_hf_decode_loop,
        run_hf_forward_prefill,
        run_infinilm_inprocess,
    )

    parts = case.strip().split(":")
    kind = parts[0].lower()
    if kind == "hf":
        # Backward compatible:
        #   hf:<target>         -> max_new=1 (forward-prefill only)
        #   hf:<target>:<max>  -> max_new=<max> (decode-loop timing)
        if len(parts) == 2:
            target = int(parts[1])
            max_new = 1
        elif len(parts) == 3:
            target = int(parts[1])
            max_new = int(parts[2])
        else:
            raise ValueError("--case hf:<target_tokens>[:<max_new_tokens>] (e.g. hf:16384 or hf:16384:32)")
    elif kind == "infinilm_rec":
        if len(parts) != 3:
            raise ValueError("--case infinilm_rec:<target>:<max_new> (e.g. infinilm_rec:32768:32)")
        target = int(parts[1])
        max_new = int(parts[2])
    else:
        raise ValueError(
            f"Unknown case kind {kind!r}; use hf: or infinilm_rec:"
        )

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt, actual = _make_prompt_with_target_tokens(tok, "How are you", target)

    if kind == "hf":

        def go() -> Dict[str, Any]:
            # Always use hf decode-loop so total_time_ms can be end-to-end
            # (prefill + decode), matching the InfiniLM generate semantics.
            return run_hf_decode_loop(
                model_path,
                prompt,
                max_new,
                device="cuda",
                attn_implementation="flash_attention_2",
                use_cache=True,
                warmup=1,
                iters=1,
            )

        try:
            r, peak = _with_mem_poll(gpu_smi, go)
            r = dict(r)
            return _row_dict(date, "hf (decode_loop)", target, actual, max_new, peak, gpu_smi, r)
        except Exception as e:
            return _row_dict(
                date,
                "hf (decode_loop)",
                target,
                actual,
                max_new,
                None,
                gpu_smi,
                {"error": str(e)},
            )

    recurrent = kind == "infinilm_rec"
    if max_new == 1:
        label = "infinilm (static_fit, recurrent GLA decode)"
    else:
        label = f"infinilm (static_fit, recurrent GLA, +{max_new} decode)"

    saved_lightning = os.environ.get("INFINI_LIGHTNING_GLA_RECURRENT_DECODE")
    saved_skip = os.environ.get("INFINI_SKIP_LAST_LOGITS_CPU")
    try:
        if recurrent:
            os.environ["INFINI_LIGHTNING_GLA_RECURRENT_DECODE"] = "1"
        else:
            os.environ.pop("INFINI_LIGHTNING_GLA_RECURRENT_DECODE", None)
        os.environ["INFINI_SKIP_LAST_LOGITS_CPU"] = "1"

        def go_inf() -> Dict[str, Any]:
            return run_infinilm_inprocess(
                model_path,
                prompt,
                max_new,
                cache_mode="static_fit",
                paged_block_size=256,
                attn_backend="default",
            )

        r, peak = _with_mem_poll(gpu_smi, go_inf)
        return _row_dict(date, label, target, actual, max_new, peak, gpu_smi, dict(r))
    except Exception as e:
        return _row_dict(date, label, target, actual, max_new, None, gpu_smi, {"error": str(e)})
    finally:
        if saved_lightning is None:
            os.environ.pop("INFINI_LIGHTNING_GLA_RECURRENT_DECODE", None)
        else:
            os.environ["INFINI_LIGHTNING_GLA_RECURRENT_DECODE"] = saved_lightning
        if saved_skip is None:
            os.environ.pop("INFINI_SKIP_LAST_LOGITS_CPU", None)
        else:
            os.environ["INFINI_SKIP_LAST_LOGITS_CPU"] = saved_skip


def print_markdown_table(rows: List[Dict[str, Any]]) -> None:
    def fmt(x: Any) -> str:
        if x is None:
            return "—"
        if isinstance(x, float):
            s = f"{x:.2f}"
            return s.rstrip("0").rstrip(".")
        return str(x)

    gpu_smi = rows[0].get("gpu_smi_index", 0) if rows else 0
    print("\n### Markdown table (paste into metrics_longtext_mem.md)\n")
    hdr = (
        "| date | backend | target_in | max_new | peak_mem_mib | total_ms | prefill_ttft_ms | "
        "prefill_tok_s | decode_itl_ms | decode_tok_s | gpu |"
    )
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    print(hdr)
    print(sep)
    for row in rows:
        if row.get("error"):
            print(
                f"| {row['date']} | {row['backend']} | {row['target_input_tokens']} | "
                f"{row['max_new_tokens']} | {fmt(row.get('peak_mem_mib'))} | OOM/err | — | — | — | — | {gpu_smi} |"
            )
            continue
        dec_itl = fmt(row.get("decode_itl_ms")) if row["max_new_tokens"] > 1 else "—"
        dec_tps = fmt(row.get("decode_throughput_tok_s")) if row["max_new_tokens"] > 1 else "—"
        ptt = row.get("prefill_ttft_ms")
        # Only forward-prefill runs use total_time_ms as a prefill-time proxy.
        if ptt is None and row.get("backend") == "hf (forward_prefill)":
            ptt = row.get("total_time_ms")
        print(
            f"| {row['date']} | {row['backend']} | {row['target_input_tokens']} | {row['max_new_tokens']} | "
            f"{fmt(row.get('peak_mem_mib'))} | {fmt(row.get('total_time_ms'))} | {fmt(ptt)} | "
            f"{fmt(row.get('prefill_throughput_tok_s'))} | {dec_itl} | {dec_tps} | {gpu_smi} |"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Long-context + decode metrics (OOM-safe --case mode)")
    ap.add_argument(
        "--case",
        type=str,
        default=None,
        help="Single case: hf:16384 | infinilm_rec:32768:32",
    )
    ap.add_argument(
        "--append-jsonl",
        type=str,
        default=None,
        help="Append one JSON line (--case mode only)",
    )
    ap.add_argument(
        "--from-jsonl",
        type=str,
        default=None,
        help="Load rows from jsonl and print markdown table",
    )
    ap.add_argument(
        "--all-in-process",
        action="store_true",
        help="Run full matrix in one process (may OOM between cases)",
    )
    args = ap.parse_args()

    model_path = os.environ.get(
        "MODEL_PATH", "/data-aisoft/zenghua/models/OpenBMB/MiniCPM-SALA"
    )
    gpu_smi = int(os.environ.get("NVML_GPU_INDEX", os.environ.get("CUDA_VISIBLE_DEVICES", "0")))
    date = os.environ.get("METRICS_DATE", "2026-03-23")
    decode_steps = int(os.environ.get("METRICS_DECODE_STEPS", "32"))
    targets = [int(x) for x in os.environ.get("METRICS_TARGETS", "16384,32768,65536").split(",")]

    examples_dir = os.path.dirname(os.path.abspath(__file__))

    if args.from_jsonl:
        rows = []
        with open(args.from_jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        print_markdown_table(rows)
        return

    if args.case:
        row = run_single_case(args.case, model_path=model_path, gpu_smi=gpu_smi, date=date)
        print(json.dumps(row, ensure_ascii=False))
        if args.append_jsonl:
            ap = os.path.abspath(args.append_jsonl)
            ad = os.path.dirname(ap)
            if ad:
                os.makedirs(ad, exist_ok=True)
            with open(ap, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    if not args.all_in_process:
        print(
            "Specify --case CASE, --from-jsonl FILE, or --all-in-process.\n"
            "For OOM safety use: ./run_longtext_metrics_cases.sh",
            file=sys.stderr,
        )
        sys.exit(2)

    # Legacy: all targets × all backends in one process
    rows: List[Dict[str, Any]] = []
    for t in targets:
        row = run_single_case(f"hf:{t}", model_path=model_path, gpu_smi=gpu_smi, date=date)
        rows.append(row)
    for t in targets:
        rows.append(
            run_single_case(f"infinilm_rec:{t}:1", model_path=model_path, gpu_smi=gpu_smi, date=date)
        )
    for t in targets:
        rows.append(
            run_single_case(
                f"infinilm_rec:{t}:{decode_steps}",
                model_path=model_path,
                gpu_smi=gpu_smi,
                date=date,
            )
        )

    out_path = os.path.join(examples_dir, "profiling_runs", "longtext_decode_metrics.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"gpu_smi_index": gpu_smi, "decode_steps": decode_steps, "rows": rows}, f, indent=2)
    print(f"Wrote {out_path}")
    print_markdown_table(rows)


if __name__ == "__main__":
    main()
