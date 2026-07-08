#!/usr/bin/env python3
# Copyright (c) 2025, InfiniCore
"""PRD-04 M3: C++ InferEngine vs TorchCompileRunner weight-shared prefill parity."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import torch

DEFAULT_SEQ_LENS = (512, 4096)

_CPP_WORKER = """
import json, sys, time, os
import infinicore, torch
from infinilm.cache import StaticKVCacheConfig
from infinilm.distributed import DistConfig
from infinilm.infer_engine import InferEngine
from infinilm.modeling_utils import load_model_state_dict_by_file

seq_len = int(sys.argv[1])
model_path = sys.argv[2]
seed = int(sys.argv[3])
out_path = sys.argv[4]

gen = torch.Generator(device="cpu")
gen.manual_seed(seed + seq_len)
ids = torch.randint(1, 73448, (1, seq_len), generator=gen, dtype=torch.long)

engine = InferEngine(
    model_path,
    device=infinicore.device("cuda", 0),
    distributed_config=DistConfig(1),
    enable_graph_compiling=False,
)
load_model_state_dict_by_file(engine, model_path, dtype=engine.dtype)
engine.reset_cache(StaticKVCacheConfig(max_batch_size=1, max_cache_len=seq_len))

t0 = time.perf_counter()
out = engine.forward(
    infinicore.from_list(ids.tolist(), dtype=infinicore.int64),
    position_ids=infinicore.from_list([list(range(seq_len))], dtype=infinicore.int64),
    past_kv_lengths=infinicore.from_list([0], dtype=infinicore.int32),
    total_kv_lengths=infinicore.from_list([seq_len], dtype=infinicore.int32),
    input_offsets=infinicore.from_list([0, seq_len], dtype=infinicore.int32),
    cu_seqlens=infinicore.from_list([0, seq_len], dtype=infinicore.int32),
    return_logits=True,
    temperature=1.0,
    top_k=1,
    top_p=1.0,
)
last = infinicore.to_torch(out).float().cpu()[0, -1, :].clone()
torch.save(last, out_path)
sys.stdout.write(json.dumps({"ms": (time.perf_counter() - t0) * 1000.0, "argmax": int(last.argmax().item())}) + "\\n")
sys.stdout.flush()
os._exit(0)
"""


@dataclass
class ParityResult:
    seq_len: int
    passed: bool
    token_match: bool
    max_abs_diff: float
    mean_abs_diff: float
    cpp_argmax: int
    torch_argmax: int
    cpp_ms: float = 0.0
    torch_ms: float = 0.0
    torch_mode: str = "eager_cpp_attn"
    mup_scales: Optional[dict] = None
    error: Optional[str] = None
    triangulation_mode: str = "full"


def _make_input_ids(seq_len: int, vocab_size: int, device: torch.device, seed: int):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed + seq_len)
    ids = torch.randint(1, vocab_size, (1, seq_len), generator=gen, dtype=torch.long)
    return ids.to(device)


def _run_triangulation(
    mode: str,
    *,
    model_path: str,
    device: torch.device,
    seq_lens: List[int],
    seed: int,
    json_out: str,
) -> int:
    layer_argv = [
        "--model-path",
        model_path,
        "--device",
        str(device),
        "--seed",
        str(seed),
        "--seq-lens",
        ",".join(str(x) for x in seq_lens),
    ]
    if json_out:
        layer_argv.extend(["--json-out", json_out])

    if mode == "rope":
        layer_argv.extend(["--mode", "rope"])
    elif mode == "attn":
        layer_argv.extend(["--mode", "attn"])
    else:
        layer_argv.extend(["--mode", "layer_bisect"])

    script = os.path.join(
        os.path.dirname(__file__), "torch_compile_layer_parity.py"
    )
    proc = subprocess.run(["python3", script, *layer_argv], check=False)
    return int(proc.returncode)


def _compare_one_seq_len(
    seq_len: int,
    *,
    args,
    torch_device: torch.device,
    cpp_last: torch.Tensor,
    cpp_ms: float,
    runner,
    torch_model,
    mup_scales_dict: Optional[dict],
) -> ParityResult:
    import torch as _torch

    vocab_size = int(torch_model.config.vocab_size)
    input_ids = _make_input_ids(seq_len, vocab_size, torch_device, args.seed)
    try:
        if torch_device.type == "cuda":
            _torch.cuda.synchronize()
        t_torch = time.perf_counter()
        with _torch.inference_mode():
            if runner is not None:
                torch_last = runner.run_prefill_last_logits(input_ids).float().cpu()
            else:
                logits = torch_model.forward_prefill_compile(input_ids)
                torch_last = logits[0, input_ids.shape[1] - 1, :].float().cpu()
        if torch_device.type == "cuda":
            _torch.cuda.synchronize()
        torch_ms = (time.perf_counter() - t_torch) * 1000.0

        diff = (cpp_last - torch_last).abs()
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())
        cpp_argmax = int(cpp_last.argmax(dim=-1).item())
        torch_argmax = int(torch_last.argmax(dim=-1).item())
        token_match = cpp_argmax == torch_argmax
        logits_ok = _torch.allclose(cpp_last, torch_last, rtol=args.rtol, atol=args.atol)
        logits_ok = logits_ok or (max_abs <= 0.2 and token_match)
        passed = logits_ok and token_match
        return ParityResult(
            seq_len=seq_len,
            passed=passed,
            token_match=token_match,
            max_abs_diff=max_abs,
            mean_abs_diff=mean_abs,
            cpp_argmax=cpp_argmax,
            torch_argmax=torch_argmax,
            cpp_ms=cpp_ms,
            torch_ms=torch_ms,
            torch_mode=args.torch_mode,
            mup_scales=mup_scales_dict,
            triangulation_mode=args.triangulation_mode,
        )
    except Exception as exc:  # noqa: BLE001
        return ParityResult(
            seq_len=seq_len,
            passed=False,
            token_match=False,
            max_abs_diff=float("nan"),
            mean_abs_diff=float("nan"),
            cpp_argmax=-1,
            torch_argmax=-1,
            triangulation_mode=args.triangulation_mode,
            error=str(exc),
        )


def _run_cpp_subprocess(
    seq_len: int,
    *,
    model_path: str,
    seed: int,
    out_path: str,
) -> tuple[torch.Tensor, float, int]:
    proc = subprocess.run(
        ["python3", "-c", textwrap.dedent(_CPP_WORKER), str(seq_len), model_path, str(seed), out_path],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 and not os.path.isfile(out_path):
        raise RuntimeError(
            f"C++ worker failed seq_len={seq_len}: rc={proc.returncode} "
            f"stderr={proc.stderr[-800:]}"
        )
    if not proc.stdout.strip():
        payload = {"ms": float("nan"), "argmax": int(torch.load(out_path, weights_only=True).argmax().item())}
    else:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    last = torch.load(out_path, weights_only=True).float()
    return last, float(payload["ms"]), int(payload["argmax"])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/models/9g_8b_thinking")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seq-lens",
        default=",".join(str(x) for x in DEFAULT_SEQ_LENS),
    )
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--cache-root", default="")
    parser.add_argument(
        "--torch-mode",
        choices=("eager_cpp_attn", "compiled_flash"),
        default=os.environ.get("INFINI_TORCH_M3_MODE", "compiled_flash"),
        help="compiled_flash: TorchCompileRunner + splitting flash (default M3 gate); "
        "eager_cpp_attn: HF eager matmul attn for triangulation",
    )
    parser.add_argument(
        "--triangulation-mode",
        choices=("full", "rope", "attn", "layer-bisect"),
        default="full",
        help="full: end-to-end logits parity; rope/attn/layer-bisect: isolate components",
    )
    parser.add_argument(
        "--worker-seq-len",
        type=int,
        default=0,
        help="Run a single seq_len worker (subprocess harness); 0 = all seq_lens in-process",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    os.environ.setdefault("INFINI_TORCH_COMPILE", "1")
    os.environ.setdefault("INFINI_TORCH_COMPILE_SHARE_WEIGHTS", "1")
    os.environ.setdefault("INFINI_RETURN_LOGITS", "1")
    os.environ.setdefault("INFINI_ATTENTION_BACKEND", "flash")

    seq_lens = [int(x.strip()) for x in args.seq_lens.split(",") if x.strip()]
    if args.worker_seq_len > 0:
        seq_lens = [int(args.worker_seq_len)]

    torch_device = torch.device(args.device)

    if args.triangulation_mode != "full":
        tri_json = args.json_out or ""
        rc = _run_triangulation(
            args.triangulation_mode,
            model_path=args.model_path,
            device=torch_device,
            seq_lens=seq_lens,
            seed=args.seed,
            json_out=tri_json,
        )
        os._exit(rc)

    print(
        f"[M3] model={args.model_path} device={torch_device} "
        f"seq_lens={seq_lens} torch_mode={args.torch_mode} "
        f"attention_backend={os.environ.get('INFINI_ATTENTION_BACKEND', '')}"
    )

    t0 = time.perf_counter()
    cpp_last_by_len: Dict[int, torch.Tensor] = {}
    cpp_times: Dict[int, float] = {}
    input_by_len: Dict[int, torch.Tensor] = {}

    with tempfile.TemporaryDirectory(prefix="m3_cpp_") as tmp:
        for seq_len in seq_lens:
            print(f"\n[M3] C++ prefill seq_len={seq_len} ...", flush=True)
            out_path = os.path.join(tmp, f"cpp_last_{seq_len}.pt")
            last, ms, argmax = _run_cpp_subprocess(
                seq_len,
                model_path=args.model_path,
                seed=args.seed,
                out_path=out_path,
            )
            cpp_last_by_len[seq_len] = last
            cpp_times[seq_len] = ms
            input_by_len[seq_len] = _make_input_ids(
                seq_len, int(last.numel() and 73448), torch_device, args.seed
            )
            print(f"[M3] C++ done seq_len={seq_len} argmax={argmax} ms={ms:.1f}")

    load_s = time.perf_counter() - t0
    print(f"[M3] C++ phase complete in {load_s:.1f}s")

    import infinicore
    from infinilm.distributed import DistConfig
    from infinilm.infer_engine import InferEngine
    from infinilm.modeling_utils import load_model_state_dict_by_file

    infini_device = infinicore.device("cuda", torch_device.index or 0)
    engine = InferEngine(
        args.model_path,
        device=infini_device,
        distributed_config=DistConfig(1),
        enable_graph_compiling=False,
        attention_backend="default",
    )
    load_model_state_dict_by_file(engine, args.model_path, dtype=engine.dtype)
    cpp_state_dict = engine._cpp_state_dict_for_compile()
    del engine
    if torch_device.type == "cuda":
        torch.cuda.empty_cache()

    mup_scales_dict: Optional[dict] = None
    torch_model = None
    runner = None

    if args.torch_mode == "compiled_flash":
        from infinilm.compile.config import TorchCompileConfig
        from infinilm.compile.env import compile_max_seq_len, torch_compile_cache_root
        from infinilm.compile.runner import TorchCompileRunner

        cache_root = args.cache_root or torch_compile_cache_root()
        cfg = TorchCompileConfig(
            model_path=args.model_path,
            max_seq_len=compile_max_seq_len(),
            cache_root=cache_root,
            prefix="weight_parity",
        )
        runner = TorchCompileRunner(
            cfg,
            device=torch_device,
            cpp_state_dict=cpp_state_dict,
        )
        print("[M3] warming up torch runner (compiled_flash) ...", flush=True)
        runner.warmup()
        torch_model = runner.model
    else:
        from infinilm.torch_llama.model import load_torch_llama

        print("[M3] loading torch eager_cpp_attn backbone ...", flush=True)
        torch_model = load_torch_llama(
            args.model_path,
            device=torch_device,
            attn_implementation="eager",
            splitting_flash_boundary=False,
            cpp_state_dict=cpp_state_dict,
        )

    mup = getattr(torch_model, "_fm9g_mup_scales", None)
    if mup is not None:
        mup_scales_dict = asdict(mup)
        print(f"[M3] applied MuP scales: {mup_scales_dict}", flush=True)
    else:
        print("[M3] MuP scales: none (llama / no MuP config)", flush=True)

    vocab_size = int(torch_model.config.vocab_size)
    for seq_len in seq_lens:
        input_by_len[seq_len] = _make_input_ids(seq_len, vocab_size, torch_device, args.seed)

    results: List[ParityResult] = []
    all_pass = True

    for seq_len in seq_lens:
        print(f"\n[M3] torch compare seq_len={seq_len} ...", flush=True)
        result = _compare_one_seq_len(
            seq_len,
            args=args,
            torch_device=torch_device,
            cpp_last=cpp_last_by_len[seq_len],
            cpp_ms=cpp_times[seq_len],
            runner=runner,
            torch_model=torch_model,
            mup_scales_dict=mup_scales_dict,
        )

        results.append(result)
        if not result.passed:
            all_pass = False
        status = "PASS" if result.passed else "FAIL"
        err_suffix = f" error={result.error}" if result.error else ""
        print(
            f"[M3] {status} seq_len={seq_len} "
            f"max_abs_diff={result.max_abs_diff:.6f} "
            f"argmax={result.cpp_argmax}/{result.torch_argmax} "
            f"token_match={result.token_match} "
            f"cpp_ms={result.cpp_ms:.1f} torch_ms={result.torch_ms:.1f}"
            f"{err_suffix}"
        )

    summary = {
        "passed": all_pass,
        "model_path": args.model_path,
        "device": str(torch_device),
        "torch_mode": args.torch_mode,
        "triangulation_mode": args.triangulation_mode,
        "attention_backend": os.environ.get("INFINI_ATTENTION_BACKEND"),
        "mup_scales": mup_scales_dict,
        "load_seconds": load_s,
        "seq_lens": [asdict(r) for r in results],
    }
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[M3] wrote {args.json_out}")

    print(f"\n[M3] OVERALL: {'PASS' if all_pass else 'FAIL'}")
    os._exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
