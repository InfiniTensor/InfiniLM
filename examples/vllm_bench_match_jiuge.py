"""
Single-prompt vLLM timing aligned with `hf_bench_match_jiuge.py` tokenization (batch 1).

Uses `TokensPrompt(prompt_token_ids=...)` so prompt IDs match jiuge/HF bench.
Reports load time, TTFT (`RequestStateStats.first_token_latency`), mean inter-token
latency over decode steps (excluding the first generated token, same convention as jiuge), and throughput derived from those intervals.

Use an isolated ``$REPO/.venv-vllm`` only (do not install vLLM into the HF interpreter).
MiniCPM5 MoE goes through ``TransformersMoEForCausalLM``; keep **transformers>=5** inside this
venv per ``minicpm5_moe_inference_profiling.md``.

For HF-only bench / checkpoint-declared Transformers **4.57.1**, use ``$REPO/.venv-no-vllm``
(``setup_hf_parity_venv.sh``) and ``hf_bench_match_jiuge.py`` — not this venv.
"""

from __future__ import annotations

import os

# TorchDynamo mis-traces HF MiniCPM5 MoE when vLLM wraps experts; EngineCore workers inherit env.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import argparse
import importlib.util
import json
import sys
import time
import traceback

_ex_dir = os.path.dirname(os.path.abspath(__file__))
_vllm_patch_dir = os.path.join(_ex_dir, "vllm_patches")
_vllm_patch_sc = os.path.join(_vllm_patch_dir, "sitecustomize.py")
# vLLM EngineCore workers are fresh interpreters (often spawn): they must see this path
# first so `import sitecustomize` during `site` startup applies the MiniCPM5 MoE hooks.
_pp = os.environ.get("PYTHONPATH", "")
_parts = [p for p in _pp.split(os.pathsep) if p]
if _vllm_patch_dir not in _parts:
    os.environ["PYTHONPATH"] = (
        _vllm_patch_dir + (os.pathsep + _pp if _pp else "")
    )
if os.path.isfile(_vllm_patch_sc):
    _spec = importlib.util.spec_from_file_location("_minicpm5_vllm_sitecustomize", _vllm_patch_sc)
    if _spec and _spec.loader:
        _patch_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_patch_mod)

from packaging import version
from tokenizers import decoders as _dec
from transformers import AutoTokenizer
import transformers


def _maybe_fix_llama_tokenizer_decoder(tokenizer, model_type: str) -> None:
    if model_type != "llama":
        return
    backend = getattr(tokenizer, "backend_tokenizer", None)
    target = getattr(backend, "_tokenizer", backend)
    norm = getattr(target, "normalizer", None)
    dec = getattr(target, "decoder", None)
    sn = repr(norm)[:800] if norm is not None else ""
    sd = repr(dec)[:800] if dec is not None else ""
    if "Prepend" in sn and "Strip" in sd:
        target.decoder = _dec.Sequence(
            [
                _dec.Replace("\u2581", " "),
                _dec.ByteFallback(),
                _dec.Fuse(),
            ]
        )


def _encode_like_jiuge(tokenizer, prompt: str) -> list[int]:
    text = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    if version.parse(transformers.__version__) < version.parse("5.0.0"):
        return tokenizer.encode_plus(
            text, truncation=True, max_length=2048, add_special_tokens=True
        )["input_ids"]
    return tokenizer._encode_plus(
        text, truncation=True, max_length=2048, add_special_tokens=True
    )["input_ids"]


def _repeat_prompt_tokens(input_ids: list[int], target_len: int) -> list[int]:
    if target_len <= 0:
        return []
    if not input_ids:
        raise ValueError("Tokenized prompt is empty.")
    repeat_times = (target_len + len(input_ids) - 1) // len(input_ids)
    return (input_ids * repeat_times)[:target_len]


def _sampling_like_hf_bench(
    max_new_tokens: int,
    top_k: int,
    top_p: float,
    temperature: float,
    ignore_eos: bool,
):
    from vllm import SamplingParams

    # HF bench uses greedy (argmax) when top_k==1 and temperature==1.0.
    greedy = top_k <= 1 and abs(temperature - 1.0) < 1e-6 and top_p >= 1.0 - 1e-6
    if greedy:
        return SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            ignore_eos=ignore_eos,
        )
    return SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k if top_k > 0 else -1,
        ignore_eos=ignore_eos,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--prompt", type=str, default="Hi")
    ap.add_argument(
        "--prompt-tokens",
        type=int,
        default=None,
        help="If set, repeat/crop the tokenized prompt to exactly this many tokens "
        "(after applying the chat template), matching bench_balanced.py behavior.",
    )
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument(
        "--stop-on-eos",
        action="store_true",
        default=False,
        help="Stop decoding if EOS is generated (default: keep going).",
    )
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable torch.compile / cudagraphs (default: true; helps remote-code / MoE). "
        "Pass --no-enforce-eager to allow compilation.",
    )
    ap.add_argument("--json", action="store_true", help="Print one JSON object with metrics.")
    ap.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="If set, save metrics JSON to this path.",
    )
    args = ap.parse_args()

    if args.batch_size != 1:
        print("Only batch_size=1 is implemented.", file=sys.stderr)
        return 2

    model_path = os.path.expanduser(args.model_path)

    try:
        from vllm import LLM
        from vllm.inputs import TokensPrompt
    except ImportError as e:
        print(f"vLLM import failed: {e}", file=sys.stderr)
        return 2

    tok0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    cfg = transformers.AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    _maybe_fix_llama_tokenizer_decoder(tokenizer, getattr(cfg, "model_type", ""))
    base_prompt_ids = _encode_like_jiuge(tokenizer, args.prompt)
    if args.prompt_tokens is None:
        prompt_ids = base_prompt_ids
        prompt_tokens_target = None
    else:
        prompt_tokens_target = int(args.prompt_tokens)
        prompt_ids = _repeat_prompt_tokens(base_prompt_ids, prompt_tokens_target)
    tok_s = time.perf_counter() - tok0

    t_load0 = time.perf_counter()
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        disable_log_stats=False,
    )
    load_s = time.perf_counter() - t_load0

    sp = _sampling_like_hf_bench(
        args.max_new_tokens,
        args.top_k,
        args.top_p,
        args.temperature,
        ignore_eos=(not args.stop_on_eos),
    )
    prompt = TokensPrompt(prompt_token_ids=prompt_ids)

    t_gen0 = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params=sp, use_tqdm=False)
    gen_wall_s = time.perf_counter() - t_gen0

    if not outputs:
        print("No outputs from vLLM.", file=sys.stderr)
        return 1
    ro = outputs[0]
    metrics = ro.metrics
    if metrics is None:
        print(
            "RequestOutput.metrics is None (need disable_log_stats=False on LLM).",
            file=sys.stderr,
        )
        return 1

    n_prompt = len(prompt_ids)
    n_gen = metrics.num_generation_tokens
    ttft_s = metrics.first_token_latency
    ttft_ms = ttft_s * 1000.0

    decode_monotonic_s = metrics.last_token_ts - metrics.first_token_ts
    if n_gen > 1:
        avg_decode_itl_ms = (decode_monotonic_s / (n_gen - 1)) * 1000.0
    else:
        avg_decode_itl_ms = 0.0

    prefill_monotonic_s = metrics.first_token_ts - metrics.scheduled_ts
    prefill_ms = prefill_monotonic_s * 1000.0

    prefill_tok_per_s = (n_prompt / ttft_s) if ttft_s > 0 else 0.0
    decode_tok_per_s = (
        (n_gen - 1) / decode_monotonic_s if n_gen > 1 and decode_monotonic_s > 0 else 0.0
    )

    row = {
        "engine": "vllm",
        "model_path": model_path,
        "vllm_version": __import__("vllm").__version__,
        "transformers_version": transformers.__version__,
        "prompt": args.prompt,
        "prompt_tokens_target": prompt_tokens_target,
        "prompt_tokens": n_prompt,
        "max_new_tokens": args.max_new_tokens,
        "n_generated": n_gen,
        "tokenizer_setup_s": tok_s,
        "load_weights_s": load_s,
        "generate_wall_s": gen_wall_s,
        "ttft_s": ttft_s,
        "ttft_ms": ttft_ms,
        "prefill_engine_ms": prefill_ms,
        "decode_engine_ms": decode_monotonic_s * 1000.0,
        "avg_decode_itl_ms": avg_decode_itl_ms,
        "prefill_tok_per_s": prefill_tok_per_s,
        "decode_tok_per_s": decode_tok_per_s,
        "queued_engine_s": metrics.scheduled_ts - metrics.queued_ts,
    }

    if args.json_out:
        out_path = os.path.abspath(os.path.expanduser(args.json_out))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(row, f, indent=2)

    if args.json:
        print(json.dumps(row, indent=2))
        return 0

    print("== vLLM bench (jiuge-aligned tokenization, batch 1) ==")
    print(f"model_path:         {model_path}")
    print(f"vllm:               {row['vllm_version']}  transformers: {row['transformers_version']}")
    print(f"prompt:             {args.prompt!r}")
    print(f"prompt tokens:      {n_prompt}")
    print(f"max_new_tokens:     {args.max_new_tokens}")
    print(f"generated tokens:   {n_gen}")
    print(f"tokenizer setup:    {tok_s * 1000:.2f} ms")
    print(f"load_weights:       {load_s * 1000:.2f} ms  ({load_s:.3f} s)")
    print(f"TTFT (wall):        {ttft_ms:.2f} ms")
    print(
        "prefill (engine):   "
        f"{prefill_ms:.2f} ms  (monotonic, scheduled → first token)"
    )
    print(
        "decode total (eng): "
        f"{decode_monotonic_s * 1000:.2f} ms  ({n_gen} tokens, monotonic)"
    )
    print(f"decode avg/step:    {avg_decode_itl_ms:.2f} ms  (excl. first token, monotonic)")
    print(f"generate() wall:    {gen_wall_s * 1000:.2f} ms")
    print(f"prefill tok/s:      {prefill_tok_per_s:.2f}  (prompt_tokens / TTFT)")
    print(f"decode tok/s:       {decode_tok_per_s:.2f}  ((n_gen-1) / decode_engine_s)")
    if ro.outputs:
        toks = ro.outputs[0].token_ids
        print(f"output token ids:   {len(toks)} ids (first 16: {toks[:16]!r})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
