#!/usr/bin/env python3
"""
Kernel-only microbench: vendored ``fused_experts`` (InfiniLM / ``torch.ops.infinilm``) vs PyPI vLLM.

Run as **two processes** with the same ``--seed`` and shape flags so tensors match; use the same
``--tuned-config-dir`` for both so ``INFINILM_TUNED_CONFIG_FOLDER`` and ``VLLM_TUNED_CONFIG_FOLDER``
point at identical JSON (copy from upstream ``vllm/.../fused_moe/configs`` — see vendor NOTICE).

InfiniLM side (system / HF parity interpreter — same libtorch as InfiniCore ``--aten=y`` build):

  REPO=...; export PYTHONPATH=$REPO/InfiniLM/python:$REPO/InfiniCore/python
  python3 $REPO/InfiniLM/examples/microbench_fused_moe_kernel.py --impl infinilm --nvidia ...

vLLM side (``.venv-vllm`` — different torch ABI; **do not** load ``_infinicore`` here):

  source $REPO/.venv-vllm/bin/activate
  python $REPO/InfiniLM/examples/microbench_fused_moe_kernel.py --impl vllm --nvidia ...

**Vendor vs upstream stack gap (same seed, container):** run
``InfiniLM/examples/run_moe_fused_stack_microbench_gap.sh`` (writes ``bench_artifacts/microbench_moe_fused_stack_gap.json``; see ``minicpm5_moe_inference_profiling.md``) instead of hand-rolling two invocations. Full ladder (microbench + smoke + e2e compare): ``run_moe_fused_stack_compare_ladder.sh``.

**Nsight Systems (kernel attribution, vendor vs upstream):** pass ``--nvtx`` so warmup and each ``fused_experts`` call are wrapped in ``torch.cuda.nvtx`` ranges (``microbench::infinilm::fused_experts``, ``microbench::vllm::fused_experts``). Then wrap the process, e.g.::

  nsys profile -o /tmp/moe_vendor.nsys-rep --trace=cuda,nvtx,osrt --force-overwrite true \\
    python3 microbench_fused_moe_kernel.py --nvtx --impl infinilm --nvidia ... --warmup 2 --iters 12

  nsys stats --report cuda_gpu_kern_sum /tmp/moe_vendor.nsys-rep --timeunit ms | head -50

Or use ``InfiniLM/examples/run_microbench_fused_moe_nsys.sh`` (same ``LD_LIBRARY_PATH`` convention as ``run_moe_fused_stack_microbench_gap.sh``). Compare the **kernel name** column: vendor Triton MoE matmuls + many small elementwise kernels vs upstream fewer fused kernels (e.g. ``silu_and_mul`` / ``swiglu*`` custom ops).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import AbstractContextManager, nullcontext
from typing import Any


def _mb_log(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"[microbench] {msg}", file=sys.stderr, flush=True)


def _nvtx_range(name: str, *, enabled: bool) -> AbstractContextManager[None]:
    """Return ``torch.cuda.nvtx.range`` when ``enabled`` (for Nsight Systems), else a no-op context."""
    if not enabled:
        return nullcontext()
    import torch

    if not torch.cuda.is_available():
        return nullcontext()
    return torch.cuda.nvtx.range(name)


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _prepend_pythonpath() -> None:
    repo = _repo_root()
    for p in (os.path.join(repo, "InfiniLM", "python"), os.path.join(repo, "InfiniCore", "python")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _load_moe_dims(model_path: str | None, overrides: dict[str, int]) -> tuple[int, int, int, int]:
    if model_path:
        cfg_path = os.path.join(os.path.expanduser(model_path), "config.json")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        n_exp = cfg.get("n_routed_experts") or cfg.get("num_local_experts")
        inter = cfg.get("moe_intermediate_size") or cfg.get("intermediate_size")
        hidden = cfg.get("hidden_size")
        top_k = cfg.get("num_experts_per_tok")
        if None in (n_exp, inter, hidden, top_k):
            raise SystemExit(f"config.json at {model_path} missing MoE dims (need experts, moe_intermediate_size, hidden, num_experts_per_tok)")
        return int(n_exp), int(inter), int(hidden), int(top_k)
    e = overrides["num_experts"]
    h = overrides["hidden"]
    i = overrides["intermediate"]
    k = overrides["top_k"]
    if min(e, h, i, k) <= 0:
        raise SystemExit("Set --model-path or positive --num-experts/--hidden/--intermediate/--top-k")
    return e, i, h, k


def _apply_tuned_config_dir(path: str | None) -> None:
    if not path:
        return
    ap = os.path.abspath(os.path.expanduser(path))
    os.environ["INFINILM_TUNED_CONFIG_FOLDER"] = ap
    os.environ["VLLM_TUNED_CONFIG_FOLDER"] = ap


def _make_tensors(
    *,
    device: "torch.device",
    dtype: "torch.dtype",
    num_tokens: int,
    num_experts: int,
    intermediate: int,
    hidden: int,
    top_k: int,
    seed: int,
):
    import torch

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    # Same draws for both processes when seed + shapes match.
    hidden_states = torch.randn(num_tokens, hidden, device=device, dtype=dtype, generator=g)
    w1 = torch.randn(num_experts, 2 * intermediate, hidden, device=device, dtype=dtype, generator=g)
    w2 = torch.randn(num_experts, hidden, intermediate, device=device, dtype=dtype, generator=g)
    w1 = w1.contiguous()
    w2 = w2.contiguous()
    logits = torch.randn(num_tokens, num_experts, device=device, dtype=torch.float32, generator=g)
    topk_w, topk_ids = torch.topk(logits, top_k, dim=-1)
    topk_w = torch.softmax(topk_w, dim=-1).to(dtype)
    topk_ids = topk_ids.to(torch.int32)
    return hidden_states, w1, w2, topk_w, topk_ids


def _run_infinilm(
    hidden_states,
    w1,
    w2,
    topk_w,
    topk_ids,
    *,
    activation: str,
    warmup: int,
    iters: int,
    verbose: bool = False,
    nvtx: bool = False,
) -> dict[str, Any]:
    import torch

    import infinicore.vendor.vllm_fused_moe  # noqa: F401
    from infinicore.vendor.vllm_fused_moe.fused_moe import MoEActivation, fused_experts

    act = MoEActivation.from_str(activation)
    _mb_log(verbose, f"infinilm warmup {warmup} iters …")
    t_w = time.perf_counter()
    with _nvtx_range("microbench::infinilm::warmup", enabled=nvtx):
        for _ in range(warmup):
            with _nvtx_range("microbench::infinilm::fused_experts", enabled=nvtx):
                out = fused_experts(
                    hidden_states,
                    w1,
                    w2,
                    topk_w,
                    topk_ids,
                    inplace=False,
                    activation=act,
                    apply_router_weight_on_input=False,
                )
    torch.cuda.synchronize()
    _mb_log(verbose, f"infinilm warmup done wall_s={time.perf_counter() - t_w:.3f}")
    _mb_log(verbose, f"infinilm timed iters={iters} …")
    t0 = time.perf_counter()
    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)
    cuda_ms: list[float] = []
    for i in range(iters):
        ev_start.record()
        with _nvtx_range("microbench::infinilm::fused_experts", enabled=nvtx):
            out = fused_experts(
                hidden_states,
                w1,
                w2,
                topk_w,
                topk_ids,
                inplace=False,
                activation=act,
                apply_router_weight_on_input=False,
            )
        ev_end.record()
        torch.cuda.synchronize()
        cuda_ms.append(ev_start.elapsed_time(ev_end))
        if verbose and (i == 0 or i == iters - 1):
            _mb_log(verbose, f"infinilm iter {i + 1}/{iters} cuda_ms={cuda_ms[-1]:.4f}")
    wall_ms = (time.perf_counter() - t0) * 1000.0 / max(iters, 1)
    mean_ms = sum(cuda_ms) / len(cuda_ms)
    _mb_log(verbose, f"infinilm timed done cuda_ms_mean={mean_ms:.4f} wall_ms_per_iter≈{wall_ms:.3f}")
    return {
        "out_last_elem": float(out.reshape(-1)[-1].detach().float().cpu()),
        "cuda_ms_mean": mean_ms,
        "cuda_ms_min": min(cuda_ms),
        "cuda_ms_max": max(cuda_ms),
        "wall_ms_per_iter": wall_ms,
    }


def _run_vllm(
    hidden_states,
    w1,
    w2,
    topk_w,
    topk_ids,
    *,
    activation: str,
    warmup: int,
    iters: int,
    verbose: bool = False,
    nvtx: bool = False,
) -> dict[str, Any]:
    import torch

    from vllm.model_executor.layers.fused_moe.fused_moe import MoEActivation, fused_experts

    act = MoEActivation.from_str(activation)
    _mb_log(verbose, f"vllm warmup {warmup} iters …")
    t_w = time.perf_counter()
    with _nvtx_range("microbench::vllm::warmup", enabled=nvtx):
        for _ in range(warmup):
            with _nvtx_range("microbench::vllm::fused_experts", enabled=nvtx):
                out = fused_experts(
                    hidden_states,
                    w1,
                    w2,
                    topk_w,
                    topk_ids,
                    inplace=False,
                    activation=act,
                    apply_router_weight_on_input=False,
                )
    torch.cuda.synchronize()
    _mb_log(verbose, f"vllm warmup done wall_s={time.perf_counter() - t_w:.3f}")
    _mb_log(verbose, f"vllm timed iters={iters} …")
    t0 = time.perf_counter()
    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)
    cuda_ms: list[float] = []
    for i in range(iters):
        ev_start.record()
        with _nvtx_range("microbench::vllm::fused_experts", enabled=nvtx):
            out = fused_experts(
                hidden_states,
                w1,
                w2,
                topk_w,
                topk_ids,
                inplace=False,
                activation=act,
                apply_router_weight_on_input=False,
            )
        ev_end.record()
        torch.cuda.synchronize()
        cuda_ms.append(ev_start.elapsed_time(ev_end))
        if verbose and (i == 0 or i == iters - 1):
            _mb_log(verbose, f"vllm iter {i + 1}/{iters} cuda_ms={cuda_ms[-1]:.4f}")
    wall_ms = (time.perf_counter() - t0) * 1000.0 / max(iters, 1)
    mean_ms = sum(cuda_ms) / len(cuda_ms)
    _mb_log(verbose, f"vllm timed done cuda_ms_mean={mean_ms:.4f} wall_ms_per_iter≈{wall_ms:.3f}")
    return {
        "out_last_elem": float(out.reshape(-1)[-1].detach().float().cpu()),
        "cuda_ms_mean": mean_ms,
        "cuda_ms_min": min(cuda_ms),
        "cuda_ms_max": max(cuda_ms),
        "wall_ms_per_iter": wall_ms,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", choices=("infinilm", "vllm"), required=True)
    ap.add_argument("--nvidia", action="store_true", help="Use CUDA device 0.")
    ap.add_argument("--device", type=str, default="cuda:0", help="Torch device string (default cuda:0).")

    ap.add_argument("--model-path", type=str, default=None, help="HF folder with config.json for MoE dims.")
    ap.add_argument("--num-experts", type=int, default=0)
    ap.add_argument("--intermediate", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=0)
    ap.add_argument("--top-k", type=int, default=0)

    ap.add_argument("--num-tokens", type=int, default=128)
    ap.add_argument(
        "--num-tokens-sweep",
        type=str,
        default=None,
        help="Comma-separated M values (overrides --num-tokens when set), e.g. 1,8,32,128,512.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=("bfloat16", "float16", "float32"))
    ap.add_argument(
        "--activation",
        type=str,
        default="silu",
        choices=("silu", "swigluoai", "swiglustep"),
        help="MoE expert activation passed to fused_experts (vendor vs vLLM enum value).",
    )

    ap.add_argument(
        "--tuned-config-dir",
        type=str,
        default=None,
        help="Directory with E=...,N=...,device_name=....json for both InfiniLM and vLLM tuning env vars.",
    )
    ap.add_argument("--jsonl-out", type=str, default=None, help="Append one JSON object per line per M in sweep.")
    ap.add_argument("--print-json", action="store_true", help="Print JSON rows to stdout.")
    ap.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log progress to stderr ([microbench] …): device, shapes, warmup/timed phases.",
    )
    ap.add_argument(
        "--nvtx",
        action="store_true",
        help="Emit torch.cuda.nvtx ranges for Nsight Systems (warmup + each fused_experts).",
    )
    args = ap.parse_args()

    if args.nvidia:
        dev_s = "cuda:0"
    else:
        dev_s = args.device

    _apply_tuned_config_dir(args.tuned_config_dir)

    if args.impl == "infinilm":
        _prepend_pythonpath()

    import torch

    dtype = getattr(torch, args.dtype)
    device = torch.device(dev_s)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise SystemExit("CUDA required for this microbench.")

    dims_over = {
        "num_experts": int(args.num_experts),
        "intermediate": int(args.intermediate),
        "hidden": int(args.hidden),
        "top_k": int(args.top_k),
    }
    num_experts, intermediate, hidden, top_k = _load_moe_dims(args.model_path, dims_over)

    if args.num_tokens_sweep:
        sweep = [int(x.strip()) for x in args.num_tokens_sweep.split(",") if x.strip()]
    else:
        sweep = [int(args.num_tokens)]

    _mb_log(
        args.verbose,
        f"impl={args.impl} device={device} E={num_experts} inter={intermediate} H={hidden} top_k={top_k} "
        f"activation={args.activation} dtype={args.dtype} sweep_M={sweep!r} warmup={args.warmup} iters={args.iters} "
        f"nvtx={args.nvtx} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}",
    )

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    rows: list[dict[str, Any]] = []
    for num_tokens in sweep:
        _mb_log(args.verbose, f"allocate tensors num_tokens={num_tokens} …")
        hidden_states, w1, w2, topk_w, topk_ids = _make_tensors(
            device=device,
            dtype=dtype,
            num_tokens=num_tokens,
            num_experts=num_experts,
            intermediate=intermediate,
            hidden=hidden,
            top_k=top_k,
            seed=args.seed,
        )
        _mb_log(
            args.verbose,
            f"shapes hs={tuple(hidden_states.shape)} w1={tuple(w1.shape)} w2={tuple(w2.shape)} "
            f"topk_w={tuple(topk_w.shape)} topk_ids={tuple(topk_ids.shape)}",
        )
        if args.impl == "infinilm":
            stats = _run_infinilm(
                hidden_states,
                w1,
                w2,
                topk_w,
                topk_ids,
                activation=args.activation,
                warmup=args.warmup,
                iters=args.iters,
                verbose=args.verbose,
                nvtx=args.nvtx,
            )
        else:
            stats = _run_vllm(
                hidden_states,
                w1,
                w2,
                topk_w,
                topk_ids,
                activation=args.activation,
                warmup=args.warmup,
                iters=args.iters,
                verbose=args.verbose,
                nvtx=args.nvtx,
            )

        row = {
            "impl": args.impl,
            "moe_fused_stack_analog": "vendor" if args.impl == "infinilm" else "upstream",
            "activation": args.activation,
            "nvtx": bool(args.nvtx),
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(device),
            "num_tokens": num_tokens,
            "num_experts": num_experts,
            "intermediate": intermediate,
            "hidden": hidden,
            "top_k": top_k,
            "dtype": args.dtype,
            "seed": args.seed,
            "warmup": args.warmup,
            "iters": args.iters,
            "tuned_config_dir": os.path.abspath(args.tuned_config_dir) if args.tuned_config_dir else None,
            **stats,
        }
        rows.append(row)
        if args.print_json:
            print(json.dumps(row), flush=True)

    if args.jsonl_out:
        out_abs = os.path.abspath(args.jsonl_out)
        out_dir = os.path.dirname(out_abs)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.jsonl_out, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
