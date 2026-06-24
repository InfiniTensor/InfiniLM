#!/usr/bin/env python3
"""Offline LLMEngine bench via LLM.chat() — ATU repro and scheduler-path validation.

Uses the same engine stack as the HTTP server (V1 scheduler, BlockManager.append_slot,
RoPE decode) without HTTP overhead.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import traceback

from infinilm.base_config import BaseConfig
from infinilm.llm.llm import LLM

BISECT_LADDER = (8192, 16384, 32768, 36864, 40960)
ATU_MARKERS = (
    "ropeThreadPerItem",
    "ATU Fault",
    "atu address translation",
    "hcErrorIllegalAddress",
)


def _repo_root() -> str:
    return os.environ.get("REPO", "/workspace")


def make_chat_prompt(model_path: str, target_tokens: int) -> str:
    """Generate user-text prompt with exact post-chat-template prefill length."""
    script = os.path.join(_repo_root(), "scripts", "gen_chunk_smoke_prompt.py")
    proc = subprocess.run(
        [
            sys.executable,
            script,
            "--model",
            model_path,
            "--target-tokens",
            str(target_tokens),
            "--via-processor",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"gen_chunk_smoke_prompt failed (rc={proc.returncode}): {proc.stderr}"
        )
    if proc.stderr:
        print(proc.stderr.rstrip(), file=sys.stderr)
    return proc.stdout


def _build_llm(cfg: BaseConfig) -> LLM:
    model_path = os.path.expanduser(cfg.model)
    device_str = cfg.get_device_str(cfg.device)

    return LLM(
        model_path=model_path,
        device=device_str,
        dtype=cfg.dtype,
        tensor_parallel_size=cfg.tp,
        cache_type="paged" if cfg.enable_paged_attn else "static",
        max_batch_size=max(cfg.batch_size, 1),
        max_tokens=cfg.max_new_tokens,
        num_blocks=cfg.num_blocks,
        block_size=cfg.block_size,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
        top_p=cfg.top_p,
        enable_graph=cfg.enable_graph,
        attn_backend=cfg.attn,
        skip_load=cfg.skip_load,
    )


def _run_iterations(
    llm: LLM,
    prompt: str,
    *,
    num_iters: int,
    batch_size: int,
    max_new_tokens: int,
) -> tuple[int, str | None]:
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    completed = 0
    last_error: str | None = None

    for i in range(num_iters):
        conversations = [conversation for _ in range(batch_size)]
        t0 = time.perf_counter()
        try:
            outputs = llm.chat(messages=conversations)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            print(f"[iter {i + 1}/{num_iters}] FAILED: {last_error}")
            traceback.print_exc()
            break

        elapsed = time.perf_counter() - t0
        n_out = len(outputs[0].outputs[0].token_ids) if outputs else 0
        print(
            f"[iter {i + 1}/{num_iters}] elapsed={elapsed:.1f}s "
            f"tokens={n_out} batch={batch_size} max_new={max_new_tokens}"
        )
        completed += 1

    return completed, last_error


def run_bench(cfg: BaseConfig) -> int:
    if cfg.bisect:
        return _run_bisect(cfg)

    model_path = os.path.expanduser(cfg.model)
    print(
        f"[llm_engine_bench] model={model_path} target_tokens={cfg.target_tokens} "
        f"iters={cfg.num_iters} batch={cfg.batch_size} max_new={cfg.max_new_tokens}"
    )

    prompt = make_chat_prompt(model_path, cfg.target_tokens)
    llm = _build_llm(cfg)
    completed, last_error = _run_iterations(
        llm,
        prompt,
        num_iters=cfg.num_iters,
        batch_size=cfg.batch_size,
        max_new_tokens=cfg.max_new_tokens,
    )

    return _finish_gate(cfg, completed, last_error)


def _run_bisect(cfg: BaseConfig) -> int:
    model_path = os.path.expanduser(cfg.model)
    total_completed = 0
    total_expected = len(BISECT_LADDER) * cfg.num_iters
    last_error: str | None = None

    llm = _build_llm(cfg)
    for target in BISECT_LADDER:
        print(f"\n[llm_engine_bench] bisect target_tokens={target} iters={cfg.num_iters}")
        prompt = make_chat_prompt(model_path, target)
        completed, err = _run_iterations(
            llm,
            prompt,
            num_iters=cfg.num_iters,
            batch_size=cfg.batch_size,
            max_new_tokens=cfg.max_new_tokens,
        )
        total_completed += completed
        if err:
            last_error = err
            break

    cfg.min_completed_iters = total_expected
    return _finish_gate(cfg, total_completed, last_error)


def _finish_gate(cfg: BaseConfig, completed: int, last_error: str | None) -> int:
    expected = cfg.min_completed_iters
    print(f"[llm_engine_bench] completed={completed}/{expected}")

    atu_found = _scan_atu_markers()
    if atu_found:
        print(f"[llm_engine_bench] ATU marker detected: {atu_found}")
        if cfg.expect_atu:
            return 0
        return 1

    if cfg.expect_atu:
        print("[llm_engine_bench] expect-atu set but no ATU marker found")
        return 1

    if completed < expected:
        if last_error:
            print(f"[llm_engine_bench] last_error: {last_error}")
        return 1

    return 0


def _scan_atu_markers() -> str | None:
    log_paths = [
        os.environ.get("INFINI_AGENT_DEBUG_LOG", ""),
        "/workspace/.cursor/debug-073e37.log",
    ]
    for path in log_paths:
        if not path or not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
            for marker in ATU_MARKERS:
                if marker in text:
                    return marker
        except OSError:
            pass
    return None


if __name__ == "__main__":
    cfg = BaseConfig()
    raise SystemExit(run_bench(cfg))
