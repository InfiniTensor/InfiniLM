#!/usr/bin/env python3
"""
E2E sweep: ``INFINILM_FORCE_MOE_BACKEND`` (baseline vs vllm_fused) × prompt token count.

Each combination runs ``bench_balanced.py`` in a **fresh subprocess** so the C++ model picks the
correct MoE block at construction time (see ``MiniCPM5MoeDecoderLayer``).
With ``--attn flash-attn``, paged KV is enabled automatically (same as typical ``bench_balanced`` smoke).

Outputs one JSON array of rows with TTFT + ``bench_balanced`` step breakdown fields
(``INFINILM_PROFILE_STEP_TIMING`` / ``gpu_forward_ms`` — full forward, not MoE-only).

For Nsight Systems, wrap this script or ``bench_balanced.py``:

  nsys profile -o moe_sweep --trace-fork-before-exec=true \\
    python3 sweep_moe_e2e_backend_tokens.py --nvidia --model-path ... --json-out ...

Nsight: look for NVTX ranges ``moe_vllm_fused::router_d2h_cpu_topk_pack_h2d`` vs
``moe_vllm_fused::fused_experts_dispatch`` inside ``MiniCPM5MoeVllmFusedSparseMoeBlock::forward``.

Pass ``-v`` / ``--verbose`` for stderr progress (each ``bench_balanced`` subprocess, timings, parsed TTFT).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any


def _vlog(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"[sweep] {msg}", file=sys.stderr, flush=True)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument(
        "--backends",
        type=str,
        default="baseline,vllm_fused",
        help="Comma list passed as INFINILM_FORCE_MOE_BACKEND (default baseline,vllm_fused).",
    )
    ap.add_argument(
        "--prompt-tokens-sweep",
        type=str,
        default="64,256,1024",
        help="Comma-separated --prompt-tokens values for bench_balanced.",
    )
    ap.add_argument("--max-new-tokens", type=int, default=1, help="Keep small to stress prefill / MoE token count.")
    ap.add_argument("--warmup-steps", type=int, default=2)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--timing-discard-runs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--attn", type=str, default="flash-attn")
    ap.add_argument("--enable-paged-attn", action="store_true")
    ap.add_argument("--paged-kv-block-size", type=int, default=256)
    ap.add_argument("--json-out", type=str, required=True)
    ap.add_argument("--nvidia", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--print-row", action="store_true")
    ap.add_argument(
        "--print-summary",
        action="store_true",
        help="After sweep, print TTFT / gpu_forward / cpu_prep by backend × prompt_tokens (stderr).",
    )
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log each subprocess (backend × prompt_tokens), timings, and stream bench_balanced prints (stderr).",
    )
    args = ap.parse_args()

    if not args.nvidia and not args.cpu:
        raise SystemExit("Pass --nvidia or --cpu (same flags as bench_balanced).")

    repo = _repo_root()
    bench = os.path.join(repo, "InfiniLM", "examples", "bench_balanced.py")
    env_base = os.environ.copy()
    env_base["INFINILM_PROFILE_STEP_TIMING"] = "1"
    if not args.verbose:
        env_base["INFINILM_SUPPRESS_BENCH_PRINTS"] = "1"
    py_path = os.pathsep.join(
        [os.path.join(repo, "InfiniLM", "python"), os.path.join(repo, "InfiniCore", "python"), env_base.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    env_base["PYTHONPATH"] = py_path

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    tokens_list = [int(x.strip()) for x in args.prompt_tokens_sweep.split(",") if x.strip()]

    _vlog(
        args.verbose,
        f"plan backends={backends!r} prompt_tokens={tokens_list!r} "
        f"flash_auto_paged={bool(args.enable_paged_attn) or args.attn == 'flash-attn'} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}",
    )

    rows: list[dict[str, Any]] = []
    for backend in backends:
        for pt in tokens_list:
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
                jpath = tf.name

            try:
                sub_env = env_base.copy()
                sub_env["INFINILM_FORCE_MOE_BACKEND"] = backend

                cmd = [
                    sys.executable,
                    bench,
                    *(["--nvidia"] if args.nvidia else ["--cpu"]),
                    "--model-path",
                    args.model_path,
                    "--tp",
                    str(args.tp),
                    "--attn",
                    args.attn,
                    "--prompt-tokens",
                    str(pt),
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--warmup-steps",
                    str(args.warmup_steps),
                    "--runs",
                    str(args.runs),
                    "--timing-discard-runs",
                    str(args.timing_discard_runs),
                    "--seed",
                    str(args.seed),
                    "--json-out",
                    jpath,
                ]
                # flash-attn + eager cache hits block_tables assertion without paged KV (see bench_balanced).
                use_paged = bool(args.enable_paged_attn) or args.attn == "flash-attn"
                if use_paged:
                    cmd.append("--enable-paged-attn")
                    cmd.extend(["--paged-kv-block-size", str(args.paged_kv_block_size)])

                _vlog(args.verbose, f"start bench_balanced INFINILM_FORCE_MOE_BACKEND={backend!r} prompt_tokens={pt} -> {jpath}")
                t0 = time.perf_counter()
                r = subprocess.run(cmd, cwd=os.path.join(repo, "InfiniLM", "examples"), env=sub_env, check=False)
                elapsed = time.perf_counter() - t0
                _vlog(
                    args.verbose,
                    f"done bench_balanced backend={backend!r} prompt_tokens={pt} exit={r.returncode} wall_s={elapsed:.2f}",
                )
                if r.returncode != 0:
                    _vlog(args.verbose, f"bench_balanced FAILED backend={backend!r} prompt_tokens={pt} see stdout/stderr above")
                    rows.append(
                        {
                            "error": f"bench_balanced exit {r.returncode}",
                            "backend": backend,
                            "prompt_tokens": pt,
                            "cmd": cmd,
                        }
                    )
                    continue

                with open(jpath, encoding="utf-8") as f:
                    bench_row = json.load(f)
                bench_row["moe_backend"] = backend
                bench_row["sweep_prompt_tokens"] = pt
                bench_row["sweep_subprocess_wall_s"] = round(elapsed, 3)
                rows.append(bench_row)
                if args.verbose:
                    _vlog(
                        args.verbose,
                        f"parsed ttft_ms={bench_row.get('ttft_ms')} "
                        f"gpu_fwd={bench_row.get('ttft_gpu_forward_ms')} load_weights_s={bench_row.get('load_weights_s')}",
                    )
                if args.print_row:
                    print(json.dumps(bench_row), flush=True)
            finally:
                try:
                    os.unlink(jpath)
                except OSError:
                    pass

    out_abs = os.path.abspath(args.json_out)
    out_dir = os.path.dirname(out_abs)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    _vlog(args.verbose, f"wrote {len(rows)} row(s) -> {out_abs}")

    if args.print_summary:
        by_pt: dict[int, dict[str, dict[str, Any]]] = {}
        for row in rows:
            if "error" in row:
                continue
            pt = int(row.get("sweep_prompt_tokens", row.get("prompt_tokens_actual", -1)))
            be = str(row.get("moe_backend", ""))
            by_pt.setdefault(pt, {})[be] = row
        print("\n# sweep summary (prefill TTFT ms; gpu_forward from RankWorker events)\n", file=sys.stderr)
        for pt in sorted(by_pt.keys()):
            blk = by_pt[pt]
            parts = [f"prompt_tokens={pt}"]
            for be in backends:
                r = blk.get(be)
                if not r:
                    parts.append(f" {be}=<missing>")
                    continue
                parts.append(
                    f" {be}: ttft={r.get('ttft_ms', 0):.1f} "
                    f"gpu_fwd={r.get('ttft_gpu_forward_ms', 0):.1f} "
                    f"cpu_prep={r.get('ttft_cpu_prep_ms', 0):.1f}"
                )
            print(" ".join(parts), file=sys.stderr)


if __name__ == "__main__":
    main()
