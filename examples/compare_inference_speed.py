#!/usr/bin/env python3
"""
Compare MiniCPM-SALA inference speed across HF, InfiniLM, and (optionally) SGLang.

Usage:
  # HF + InfiniLM only (InfiniLM runs in subprocess with same env as jiuge):
  python compare_inference_speed.py --model_path /path/to/MiniCPM-SALA [--prompt "How are you"] [--max_new_tokens 32]

  # Include SGLang (server must already be running with MiniCPM-SALA):
  python compare_inference_speed.py --model_path /path/to/MiniCPM-SALA --sglang_url http://127.0.0.1:30000

  # Optional: write JSON
  python compare_inference_speed.py --model_path /path/to/MiniCPM-SALA --output results.json

Requires: transformers, torch; for InfiniLM subprocess: PYTHONPATH and LD_LIBRARY_PATH as in jiuge.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Optional, Tuple, Literal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python"))

try:
    # Best-effort InfLLM-v2 preload to avoid requiring LD_PRELOAD in
    # profiling tools like nsys. Safe when infllm_v2 is absent.
    from infllmv2_loader import preload_infllmv2_if_available as _preload_infllmv2_if_available
except Exception:  # pragma: no cover - defensive import guard
    _preload_infllmv2_if_available = None

if _preload_infllmv2_if_available is not None:
    _preload_infllmv2_if_available()

def _build_chat_input_ids(tokenizer, prompt: str):
    conversation = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    return ids


def _make_prompt_with_target_tokens(tokenizer, base_prompt: str, target_input_tokens: int) -> Tuple[str, int]:
    """
    Build a prompt (user content) such that the *chat-templated* input_ids length is >= target_input_tokens.
    Returns (prompt, actual_input_tokens).
    """
    if target_input_tokens <= 0:
        raise ValueError("--target_input_tokens must be > 0")

    # Ensure boundaries don't merge tokens weirdly.
    chunk = (base_prompt.strip() + "\n") if base_prompt.strip() else "hello\n"

    # Exponential growth to find an upper bound.
    rep = 1
    while True:
        prompt = chunk * rep
        ids = _build_chat_input_ids(tokenizer, prompt)
        if len(ids) >= target_input_tokens:
            break
        rep *= 2
        if rep > 1_000_000:
            raise RuntimeError("Failed to build prompt to target length (rep too large)")

    # Binary search for smallest rep that reaches target.
    lo, hi = 1, rep
    best_prompt = prompt
    best_len = len(ids)
    while lo <= hi:
        mid = (lo + hi) // 2
        p = chunk * mid
        l = len(_build_chat_input_ids(tokenizer, p))
        if l >= target_input_tokens:
            best_prompt, best_len = p, l
            hi = mid - 1
        else:
            lo = mid + 1

    return best_prompt, best_len


def run_hf(
    model_path: str,
    prompt: str,
    max_new_tokens: int,
    device: str = "cuda",
    *,
    attn_implementation: Optional[str] = None,
):
    """Run HuggingFace generate and return metrics."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model_kwargs = {
        "torch_dtype": "auto",
        "trust_remote_code": True,
    }
    # Prefer flash-attn when available; fall back silently if not supported.
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation  # type: ignore[assignment]
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs,
        ).to(device)
    except TypeError:
        # Older transformers versions may not support attn_implementation kwarg.
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs,
        ).to(device)
    model.eval()

    conversation = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]

    start = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id or 0,
        )
    elapsed = time.perf_counter() - start
    output_len = out.shape[1] - input_len

    return {
        "backend": "hf",
        "total_time_ms": round(elapsed * 1000, 2),
        "input_tokens": input_len,
        "output_tokens": output_len,
        "prefill_ttft_ms": None,  # HF generate() doesn't expose TTFT without streaming
        "decode_throughput_tok_s": round(output_len / elapsed, 2) if elapsed > 0 else None,
        "total_throughput_tok_s": round((input_len + output_len) / elapsed, 2) if elapsed > 0 else None,
    }


def run_hf_forward_prefill(
    model_path: str,
    prompt: str,
    device: str = "cuda",
    *,
    attn_implementation: Optional[str] = None,
    use_cache: bool = True,
    warmup: int = 1,
    iters: int = 1,
):
    """
    Run HuggingFace *forward-only* prefill (no decode loop).
    Intended for kernel-level profiling to isolate prefill work.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model_kwargs = {
        "torch_dtype": "auto",
        "trust_remote_code": True,
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation  # type: ignore[assignment]
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).to(device)
    except TypeError:
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).to(device)
    model.eval()

    conversation = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]

    # Warmup (reduces first-iter compilation / cache effects for profiling).
    with torch.inference_mode():
        for _ in range(max(0, warmup)):
            # Prefer last-token logits only (reduces memory at long context).
            try:
                _ = model(**inputs, use_cache=use_cache, logits_to_keep=1)
            except TypeError:
                _ = model(**inputs, use_cache=use_cache)
        torch.cuda.synchronize()

    # Timed iters.
    times = []
    with torch.inference_mode():
        for _ in range(max(1, iters)):
            torch.cuda.synchronize()
            try:
                torch.cuda.nvtx.range_push("hf_forward_prefill")
            except Exception:
                pass
            start = time.perf_counter()
            try:
                _ = model(**inputs, use_cache=use_cache, logits_to_keep=1)
            except TypeError:
                _ = model(**inputs, use_cache=use_cache)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass
            times.append(elapsed)

    best = min(times) if times else 0.0
    return {
        "backend": "hf_forward_prefill",
        "total_time_ms": round(best * 1000, 2),
        "input_tokens": int(input_len),
        "output_tokens": 0,
        "use_cache": bool(use_cache),
        "warmup": int(warmup),
        "iters": int(iters),
        "prefill_throughput_tok_s": round(input_len / best, 2) if best > 0 else None,
    }


def run_hf_decode_loop(
    model_path: str,
    prompt: str,
    max_new_tokens: int,
    device: str = "cuda",
    *,
    attn_implementation: Optional[str] = None,
    use_cache: bool = True,
    warmup: int = 8,
    iters: int = 1,
):
    """
    Measure HF *decode-only* per-token latency using a manual loop with past_key_values.

    Protocol:
    - Prefill once on the full prompt (not included in decode timing).
    - Then decode `max_new_tokens` tokens with 1-token steps, timing the whole decode loop
      (optionally best-of `iters`).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if max_new_tokens <= 0:
        raise ValueError("--max_new_tokens must be > 0 for hf decode_loop")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model_kwargs = {
        "torch_dtype": "auto",
        "trust_remote_code": True,
    }
    if attn_implementation is not None:
        model_kwargs["attn_implementation"] = attn_implementation  # type: ignore[assignment]
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).to(device)
    except TypeError:
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).to(device)
    model.eval()

    conversation = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_len = int(inputs.input_ids.shape[1])

    # Prefill once to build cache.
    with torch.inference_mode():
        try:
            pre = model(**inputs, use_cache=use_cache, logits_to_keep=1)
        except TypeError:
            pre = model(**inputs, use_cache=use_cache)
        past = getattr(pre, "past_key_values", None)
        # Greedy next token from last logits.
        logits = pre.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)

    # Warmup decode steps (not timed) to reduce first-step effects.
    with torch.inference_mode():
        for _ in range(max(0, warmup)):
            try:
                out = model(input_ids=next_token, use_cache=use_cache, past_key_values=past, logits_to_keep=1)
            except TypeError:
                out = model(input_ids=next_token, use_cache=use_cache, past_key_values=past)
            past = getattr(out, "past_key_values", past)
            logits = out.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        torch.cuda.synchronize()

    # Timed decode loops (best-of iters).
    times = []
    with torch.inference_mode():
        for _ in range(max(1, iters)):
            # Re-prefill to avoid measuring a "warmed" cache from prior iteration.
            try:
                pre = model(**inputs, use_cache=use_cache, logits_to_keep=1)
            except TypeError:
                pre = model(**inputs, use_cache=use_cache)
            past = getattr(pre, "past_key_values", None)
            logits = pre.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

            torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                torch.cuda.nvtx.range_push("hf_decode_loop")
            except Exception:
                pass
            for _t in range(max_new_tokens):
                try:
                    out = model(input_ids=next_token, use_cache=use_cache, past_key_values=past, logits_to_keep=1)
                except TypeError:
                    out = model(input_ids=next_token, use_cache=use_cache, past_key_values=past)
                past = getattr(out, "past_key_values", past)
                logits = out.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass
            times.append(elapsed)

    best = min(times) if times else 0.0
    itl_ms = (best * 1000.0 / max_new_tokens) if best > 0 else None
    thr = (max_new_tokens / best) if best > 0 else None
    return {
        "backend": "hf_decode_loop",
        "total_time_ms": round(best * 1000, 2),
        "input_tokens": int(input_len),
        "output_tokens": int(max_new_tokens),
        "decode_itl_ms": round(itl_ms, 4) if itl_ms is not None else None,
        "decode_throughput_tok_s": round(thr, 2) if thr is not None else None,
        "use_cache": bool(use_cache),
        "warmup": int(warmup),
        "iters": int(iters),
    }


def run_infinilm_inprocess(
    model_path: str,
    prompt: str,
    max_new_tokens: int,
    *,
    cache_mode: Literal["static_fit", "static_maxpos", "paged"] = "paged",
    paged_block_size: int = 256,
    attn_backend: str = "flash-attn",
):
    """
    Run InfiniLM in-process (no 2048-token truncation). Parses InferEngine's timing prints.
    This expects PYTHONPATH to include InfiniLM/InfiniCore python packages (container runner does this).
    """
    import io
    import torch
    import contextlib

    import infinicore
    from transformers import AutoTokenizer

    from infinilm.cache import PagedKVCacheConfig, StaticKVCacheConfig
    from infinilm.distributed import DistConfig
    from infinilm.infer_engine import GenerationConfig, InferEngine
    from infinilm.modeling_utils import load_model_state_dict_by_file

    model_path = os.path.expanduser(model_path)
    # Prefer flash-attn when available; fall back to default.
    try:
        model = InferEngine(
            model_path,
            device=infinicore.device("cuda", 0),
            distributed_config=DistConfig(1),
            enable_graph_compiling=False,
            attention_backend=attn_backend,
        )
    except TypeError:
        # Older InferEngine builds may not accept attention_backend.
        model = InferEngine(
            model_path,
            device=infinicore.device("cuda", 0),
            distributed_config=DistConfig(1),
            enable_graph_compiling=False,
        )
    except Exception:
        try:
            model = InferEngine(
                model_path,
                device=infinicore.device("cuda", 0),
                distributed_config=DistConfig(1),
                enable_graph_compiling=False,
                attention_backend="default",
            )
        except TypeError:
            model = InferEngine(
                model_path,
                device=infinicore.device("cuda", 0),
                distributed_config=DistConfig(1),
                enable_graph_compiling=False,
            )
    load_model_state_dict_by_file(model, model_path, dtype=model.config.dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    input_ids = _build_chat_input_ids(tokenizer, prompt)
    input_ids_infini = infinicore.from_list([input_ids])

    initial_capacity = len(input_ids) + max_new_tokens
    if cache_mode == "paged":
        num_blocks = (initial_capacity + (paged_block_size - 1)) // paged_block_size
        cache_config = PagedKVCacheConfig(
            num_blocks=num_blocks,
            block_size=paged_block_size,
        )
    else:
        if cache_mode == "static_maxpos":
            max_pos = getattr(model.config, "max_position_embeddings", 4096)
            max_cache_len = max(initial_capacity, max_pos)
        else:
            # Fit cache to what we actually need for this run.
            max_cache_len = initial_capacity
        cache_config = StaticKVCacheConfig(max_batch_size=1, max_cache_len=max_cache_len)
    # Basic GPU memory stats around cache construction (CUDA device assumed to be index 0).
    mem_before_cache = torch.cuda.memory_allocated(0)
    max_mem_before_cache = torch.cuda.max_memory_allocated(0)

    model.reset_cache(cache_config)

    mem_after_cache = torch.cuda.memory_allocated(0)
    max_mem_after_cache = torch.cuda.max_memory_allocated(0)

    buf = io.StringIO()
    start = time.perf_counter()
    with contextlib.redirect_stdout(buf):
        try:
            torch.cuda.nvtx.range_push("infinilm_generate")
        except Exception:
            pass
        try:
            model.generate(
                input_ids_infini,
                GenerationConfig(
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,
                    top_k=1,
                    top_p=1.0,
                    # Profiling: avoid per-step EOS checks + early stop variability.
                    stop_on_eos=False,
                ),
                _measure_and_log_time=True,
            )
        finally:
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass
    elapsed = time.perf_counter() - start
    stdout = buf.getvalue()

    prefill_ttft_ms = None
    prefill_throughput = None
    decode_itl_ms = None
    decode_throughput = None
    gen_completed_ms = None
    for line in stdout.splitlines():
        if "Prefill TTFT:" in line:
            m = re.search(
                r"Prefill TTFT:\s*([\d.]+)\s*ms.*Throughput:\s*([\d.]+)\s*tok/s", line
            )
            if m:
                prefill_ttft_ms = float(m.group(1))
                prefill_throughput = float(m.group(2))
        if "Decode" in line and "ITL:" in line:
            m = re.search(
                r"Decode\s+Avg ITL:\s*([\d.]+)\s*ms.*Throughput:\s*([\d.]+)\s*tok/s", line
            )
            if m:
                decode_itl_ms = float(m.group(1))
                decode_throughput = float(m.group(2))
        if "Generation completed in" in line:
            m = re.search(r"Generation completed in\s*([\d.]+)\s*ms", line)
            if m:
                gen_completed_ms = float(m.group(1))

    return {
        "backend": "infinilm",
        "total_time_ms": round(elapsed * 1000, 2),
        "input_tokens": len(input_ids),
        "output_tokens": max_new_tokens,
        "prefill_ttft_ms": prefill_ttft_ms,
        "prefill_throughput_tok_s": prefill_throughput,
        "decode_itl_ms": decode_itl_ms,
        "decode_throughput_tok_s": decode_throughput,
        "engine_reported_generation_ms": gen_completed_ms,
        # Cache / attention configuration
        "cache_mode": cache_mode,
        "paged_block_size": paged_block_size if cache_mode == "paged" else None,
        "enable_paged_attn": getattr(model, "enable_paged_attn", False),
        "static_max_cache_len": max_cache_len if cache_mode != "paged" else None,
        "paged_num_blocks": num_blocks if cache_mode == "paged" else None,
        # Torch CUDA memory snapshots (bytes)
        "torch_memory_allocated_before_cache": int(mem_before_cache),
        "torch_memory_allocated_after_cache": int(mem_after_cache),
        "torch_max_memory_allocated_before_cache": int(max_mem_before_cache),
        "torch_max_memory_allocated_after_cache": int(max_mem_after_cache),
    }


def run_infinilm(model_path: str, prompt: str, max_new_tokens: int, env=None):
    """Run InfiniLM jiuge via subprocess and parse stdout for metrics."""
    run_env = {**os.environ, **(env or {})}
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    jiuge_py = os.path.join(examples_dir, "jiuge.py")
    cmd = [
        sys.executable,
        jiuge_py,
        "--nvidia",
        "--model_path", model_path,
        "--prompt", prompt,
        "--max_new_tokens", str(max_new_tokens),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            env=run_env,
            cwd=examples_dir,
        )
        stdout = result.stdout or ""
        if result.returncode != 0 and not stdout:
            return {"backend": "infinilm", "error": (result.stderr or f"exit code {result.returncode}")[:500]}
    except Exception as e:
        return {"backend": "infinilm", "error": str(e)}

    # Parse jiuge / InferEngine output
    prefill_ttft_ms = None
    prefill_throughput = None
    decode_itl_ms = None
    decode_throughput = None
    total_time_ms = None
    for line in stdout.splitlines():
        if "Prefill TTFT:" in line:
            m = re.search(r"Prefill TTFT:\s*([\d.]+)\s*ms.*Throughput:\s*([\d.]+)\s*tok/s", line)
            if m:
                prefill_ttft_ms = float(m.group(1))
                prefill_throughput = float(m.group(2))
        if "Decode" in line and "ITL:" in line:
            m = re.search(r"Decode\s+Avg ITL:\s*([\d.]+)\s*ms.*Throughput:\s*([\d.]+)\s*tok/s", line)
            if m:
                decode_itl_ms = float(m.group(1))
                decode_throughput = float(m.group(2))
        if "total_time:" in line:
            m = re.search(r"total_time:\s*([\d.]+)\s*ms", line)
            if m:
                total_time_ms = float(m.group(1))
        if "Generation completed in" in line:
            m = re.search(r"Generation completed in\s*([\d.]+)\s*ms", line)
            if m:
                total_time_ms = float(m.group(1))

    return {
        "backend": "infinilm",
        "total_time_ms": total_time_ms,
        "prefill_ttft_ms": prefill_ttft_ms,
        "prefill_throughput_tok_s": prefill_throughput,
        "decode_itl_ms": decode_itl_ms,
        "decode_throughput_tok_s": decode_throughput,
    }


def run_sglang_client(sglang_url: str, prompt: str, max_new_tokens: int):
    """Send one request to SGLang server and return metrics."""
    try:
        import requests
    except ImportError:
        return {"backend": "sglang", "error": "requests not installed"}

    url = sglang_url.rstrip("/") + "/generate"
    payload = {
        "text": prompt,
        "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0},
    }
    start = time.perf_counter()
    try:
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return {"backend": "sglang", "error": str(e)}
    elapsed_ms = (time.perf_counter() - start) * 1000

    # SGLang response may have "meta_info" with "completion_tokens" or we use prompt + output length
    output_text = (data.get("text") or data.get("choices", [{}])[0].get("text") or "")
    completion_tokens = data.get("meta_info", {}).get("completion_tokens") or data.get("usage", {}).get("completion_tokens")
    if completion_tokens is None and "usage" in data:
        completion_tokens = data["usage"].get("completion_tokens")
    if completion_tokens is None:
        completion_tokens = max_new_tokens  # fallback

    return {
        "backend": "sglang",
        "total_time_ms": round(elapsed_ms, 2),
        "output_tokens": completion_tokens,
        "total_throughput_tok_s": round(completion_tokens / (elapsed_ms / 1000), 2) if elapsed_ms > 0 else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare MiniCPM-SALA inference speed: HF, InfiniLM, SGLang")
    parser.add_argument("--model_path", required=True, help="Path to MiniCPM-SALA model dir")
    parser.add_argument("--prompt", default="How are you", help="Prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Max new tokens to generate")
    parser.add_argument(
        "--target_input_tokens",
        type=int,
        default=None,
        help="If set, synthesize a long prompt so chat-templated input tokens >= this value (e.g. 65536).",
    )
    parser.add_argument(
        "--infinilm_cache_mode",
        type=str,
        default="paged",
        choices=["paged", "static_fit", "static_maxpos"],
        help="InfiniLM KV cache mode when running long prompts in-process.",
    )
    parser.add_argument(
        "--infinilm_paged_block_size",
        type=int,
        default=256,
        help="Paged KV block size (tokens per block).",
    )
    parser.add_argument(
        "--infinilm_attn_backend",
        type=str,
        default="flash-attn",
        help="InfiniLM attention backend (e.g. flash-attn or default).",
    )
    parser.add_argument(
        "--hf_attn_implementation",
        type=str,
        default="flash_attention_2",
        help="HF attention implementation to request (e.g. flash_attention_2 or eager).",
    )
    parser.add_argument(
        "--hf_mode",
        type=str,
        default="generate",
        choices=["generate", "forward_prefill", "decode_loop"],
        help="HF run mode: generate() end-to-end, forward-only prefill, or manual decode_loop timing with KV cache.",
    )
    parser.add_argument(
        "--hf_forward_use_cache",
        action="store_true",
        help="In HF forward_prefill mode, pass use_cache=True (recommended).",
    )
    parser.add_argument(
        "--hf_forward_warmup",
        type=int,
        default=1,
        help="Warmup iterations for HF forward_prefill.",
    )
    parser.add_argument(
        "--hf_forward_iters",
        type=int,
        default=1,
        help="Measured iterations for HF forward_prefill (best-of).",
    )
    parser.add_argument(
        "--hf_decode_warmup",
        type=int,
        default=8,
        help="Warmup steps for HF decode_loop (not timed).",
    )
    parser.add_argument(
        "--hf_decode_iters",
        type=int,
        default=1,
        help="Measured iterations for HF decode_loop (best-of).",
    )
    parser.add_argument("--sglang_url", default=None, help="SGLang server URL (e.g. http://127.0.0.1:30000); if set, query SGLang")
    parser.add_argument("--backends", default="hf,infinilm", help="Comma-separated: hf,infinilm,sglang")
    parser.add_argument("--output", default=None, help="Write JSON results to this path")
    parser.add_argument("--no_hf", action="store_true", help="Skip HF (e.g. if no GPU memory for two models)")
    parser.add_argument("--no_infinilm", action="store_true", help="Skip InfiniLM")
    parser.add_argument(
        "--prefill_16k",
        action="store_true",
        help="Convenience flag: set --target_input_tokens=16384 and --max_new_tokens=1 (prefill-dominated).",
    )
    parser.add_argument(
        "--infinilm_inprocess",
        action="store_true",
        help="Run InfiniLM in-process (no jiuge subprocess). Use when PYTHONPATH/LD_LIBRARY_PATH are set in this process.",
    )
    args = parser.parse_args()

    backends = [b.strip() for b in args.backends.split(",")]
    results = []

    # Normalize convenience prefill-only configuration.
    if args.prefill_16k:
        if args.target_input_tokens is None:
            args.target_input_tokens = 16384
        # For prefill-dominated comparisons, prefer HF forward-only by default.
        if args.hf_mode == "generate":
            args.hf_mode = "forward_prefill"
        if args.max_new_tokens != 1:
            args.max_new_tokens = 1

    # If requested, build a long prompt once using HF tokenizer.
    if args.target_input_tokens is not None:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
            long_prompt, actual = _make_prompt_with_target_tokens(tok, args.prompt, args.target_input_tokens)
            args.prompt = long_prompt
            print(f"[prompt] synthesized chat input tokens: {actual} (target >= {args.target_input_tokens})")
        except Exception as e:
            print(f"[prompt] failed to synthesize long prompt: {e}")

    if "hf" in backends and not args.no_hf:
        try:
            import torch
            if args.hf_mode == "forward_prefill":
                r = run_hf_forward_prefill(
                    args.model_path,
                    args.prompt,
                    attn_implementation=args.hf_attn_implementation,
                    use_cache=args.hf_forward_use_cache,
                    warmup=args.hf_forward_warmup,
                    iters=args.hf_forward_iters,
                )
            elif args.hf_mode == "decode_loop":
                r = run_hf_decode_loop(
                    args.model_path,
                    args.prompt,
                    args.max_new_tokens,
                    attn_implementation=args.hf_attn_implementation,
                    use_cache=True,
                    warmup=args.hf_decode_warmup,
                    iters=args.hf_decode_iters,
                )
            else:
                r = run_hf(
                    args.model_path,
                    args.prompt,
                    args.max_new_tokens,
                    attn_implementation=args.hf_attn_implementation,
                )
            results.append(r)
        except Exception as e:
            results.append({"backend": "hf", "error": str(e)})

    if "infinilm" in backends and not args.no_infinilm:
        # In-process: when env is set in this process or --infinilm_inprocess, avoid jiuge subprocess.
        # Also use in-process for long prompts (target_input_tokens) to avoid 2048-token truncation.
        use_inprocess = args.infinilm_inprocess or args.target_input_tokens is not None
        if use_inprocess:
            try:
                r = run_infinilm_inprocess(
                    args.model_path,
                    args.prompt,
                    args.max_new_tokens,
                    cache_mode=args.infinilm_cache_mode,  # type: ignore[arg-type]
                    paged_block_size=args.infinilm_paged_block_size,
                    attn_backend=args.infinilm_attn_backend,
                )
            except Exception as e:
                r = {"backend": "infinilm", "error": str(e)}
        else:
            r = run_infinilm(args.model_path, args.prompt, args.max_new_tokens)
        results.append(r)

    if "sglang" in backends and args.sglang_url:
        r = run_sglang_client(args.sglang_url, args.prompt, args.max_new_tokens)
        results.append(r)
    elif "sglang" in backends and not args.sglang_url:
        results.append({"backend": "sglang", "error": "No --sglang_url provided; start SGLang server with MiniCPM-SALA first"})

    # Print table
    print("\n" + "=" * 60)
    print("MiniCPM-SALA inference speed comparison")
    print("=" * 60)
    print(f"  prompt = {repr(args.prompt[:500])}   max_new_tokens = {args.max_new_tokens}")
    print()
    for r in results:
        if "error" in r:
            print(f"  {r['backend']}: ERROR {r['error']}")
            continue
        print(f"  {r['backend']}:")
        for k, v in r.items():
            if k == "backend" or v is None:
                continue
            if isinstance(v, float):
                print(f"    {k}: {v}")
            else:
                print(f"    {k}: {v}")
        print()
    print("=" * 60)

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"prompt": args.prompt, "max_new_tokens": args.max_new_tokens, "results": results}, f, indent=2)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    import os
    main()
