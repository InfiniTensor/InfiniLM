# Copyright (c) 2025, InfiniCore
"""Unified process entry: offline AOT compile and/or HTTP serve.

Phases:
  compile — plan + AOT for missing packages; exit (no InferEngine / HTTP).
  serve   — same as ``inference_server.main`` (register-only AOT at engine init).
  all     — compile missing packages (skip entirely on full cache hit), then serve.

Cold kickoff (InfiniOrchestrator Qwen): ``python -m infinilm.server.entry --phase all ...``

CUDA-graph policy (``--cudagraph-policy`` / ``INFINI_CUDAGRAPH_POLICY``):
  eager              — no decode/prefill CUDA-graph capture (default).
  full_and_piecewise — vLLM dual-mode: FULL uniform decode + PIECEWISE
                       homogeneous prefill (MIXED→eager until mixed
                       PIECEWISE exists). MetaX: FA host-break + MoE
                       host-break; ``INFINI_FA_FORCE_CAPTURE`` /
                       ``INFINI_MOE_TRITON_CAPTURE`` diagnose-only.
Policy applies to serve / all; compile-only phase ignores graph capture knobs
(AOT packages do not need FA/MoE stream-capture policy).
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
from typing import Optional, Sequence

logger = logging.getLogger(__name__)


def _peel_entry_args(argv: Optional[Sequence[str]] = None):
    """Parse ``--phase`` / ``--force`` / ``--cudagraph-policy``; leave rest for ``BaseConfig``."""
    parser = argparse.ArgumentParser(
        prog="infinilm.server.entry",
        description="InfiniLM server entry (compile | serve | all)",
        add_help=False,
    )
    parser.add_argument(
        "--phase",
        choices=("compile", "serve", "all"),
        default="serve",
        help="compile: offline AOT only; serve: HTTP only; all: compile-prep then serve",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompile planned packages even when segment.pt2 already exists",
    )
    parser.add_argument(
        "--cudagraph-policy",
        choices=("eager", "full_and_piecewise"),
        default="eager",
        help=(
            "CUDA-graph policy (sets INFINI_CUDAGRAPH_POLICY; CLI wins over prior env). "
            "eager: no CG capture. full_and_piecewise: FULL uniform decode + "
            "PIECEWISE homogeneous prefill (MIXED→eager); MetaX FA/MoE host-break. "
            "FA_FORCE / MOE_TRITON_CAPTURE diagnose-only. Do not pass track_b."
        ),
    )
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        help="Show help (entry + BaseConfig flags)",
    )
    args, remaining = parser.parse_known_args(list(argv) if argv is not None else None)
    return args, remaining


def _apply_entry_cudagraph_policy(policy: str) -> str:
    """CLI wins: set env from ``--cudagraph-policy`` then expand companion knobs."""
    from infinilm.compile.env import apply_cudagraph_policy_env

    os.environ["INFINI_CUDAGRAPH_POLICY"] = policy
    return apply_cudagraph_policy_env(policy)


def _run_compile(cfg, *, force: bool) -> dict:
    from infinilm.compile.piecewise_bootstrap_plan import (
        build_piecewise_bootstrap_plan,
        compile_planned_packages,
    )
    from infinilm.compile.env import piecewise_inductor_segment_enabled

    if not piecewise_inductor_segment_enabled():
        logger.warning(
            "INFINI_PIECEWISE_INDUCTOR_SEGMENT is off; compile phase is a no-op"
        )
        return {"compiled": 0, "skipped": 0, "noop": True}

    plan = build_piecewise_bootstrap_plan(
        model_path=cfg.model,
        tp_size=cfg.tp,
    )
    if plan.full_cache_exists() and not force:
        logger.info(
            "piecewise AOT cache hit: all %s planned packages present under %s; "
            "skip compile",
            plan.expected_count,
            plan.cache_root,
        )
        return {"compiled": 0, "skipped": plan.expected_count, "cache_hit": True}

    return compile_planned_packages(plan, force=force)


def _release_compile_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    entry_args, remaining = _peel_entry_args(argv)

    # Apply policy before BaseConfig / engine so C++ sees expanded envs at init.
    applied = _apply_entry_cudagraph_policy(entry_args.cudagraph_policy)
    if entry_args.phase == "compile":
        logger.info(
            "entry --phase compile: cudagraph-policy=%s is a no-op for AOT "
            "(graph capture knobs apply to serve)",
            applied,
        )

    # BaseConfig reads sys.argv; temporarily replace so entry flags are not warned.
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *remaining]
        if entry_args.help or "-h" in remaining or "--help" in remaining:
            print(
                "usage: python -m infinilm.server.entry --phase {compile,serve,all} "
                "[--force] [--cudagraph-policy {eager,full_and_piecewise}] "
                "[BaseConfig flags...]\n"
                "\n"
                "  compile  Offline AOT for planned packages; exit (no HTTP).\n"
                "  serve    Start inference server (register-only AOT).\n"
                "  all      Compile missing packages (skip on full cache hit), then serve.\n"
                "\n"
                "  --cudagraph-policy  eager (default; no CG) | full_and_piecewise\n"
                "                      (FULL uniform decode + PIECEWISE prefill;\n"
                "                      MIXED→eager; MetaX FA/MoE host-break).\n"
                "                      Sets INFINI_CUDAGRAPH_POLICY; CLI wins.\n"
                "                      FA_FORCE / MOE_TRITON_CAPTURE diagnose-only.\n"
                "                      Ignored for graph capture during --phase compile.\n"
            )
            from infinilm.base_config import BaseConfig

            stub = BaseConfig.__new__(BaseConfig)
            stub.parser = argparse.ArgumentParser(description="InfiniLM Unified Config")
            BaseConfig._add_common_args(stub)
            stub.parser.print_help()
            return 0

        from infinilm.base_config import BaseConfig
        from infinilm.server.inference_server import (
            run_server_from_config,
            setup_logging,
        )

        cfg = BaseConfig()
        setup_logging(cfg.log_level)
        from infinilm.compile.env import prefill_native_cg_enabled

        logger.info(
            "entry cudagraph_policy=%s prefill_native_cg=%s "
            "INFINI_DECODE_GRAPH_ONLY=%s INFINI_DECODE_CG_BATCHES=%s "
            "INFINI_NATIVE_CG_CAPTURE_BUCKETS=%s INFINI_FA_FORCE_CAPTURE=%s "
            "INFINI_MOE_TRITON_CAPTURE=%s",
            applied,
            prefill_native_cg_enabled(),
            os.environ.get("INFINI_DECODE_GRAPH_ONLY", ""),
            os.environ.get("INFINI_DECODE_CG_BATCHES", ""),
            os.environ.get("INFINI_NATIVE_CG_CAPTURE_BUCKETS", ""),
            os.environ.get("INFINI_FA_FORCE_CAPTURE", "0"),
            os.environ.get("INFINI_MOE_TRITON_CAPTURE", "unset"),
        )

        phase = entry_args.phase
        force = bool(entry_args.force)

        if phase == "compile":
            _run_compile(cfg, force=force)
            _release_compile_memory()
            return 0

        if phase == "all":
            _run_compile(cfg, force=force)
            _release_compile_memory()
            logger.info("entry --phase all: starting serve after compile-prep")
            run_server_from_config(cfg)
            return 0

        # serve
        run_server_from_config(cfg)
        return 0
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    sys.exit(main())
