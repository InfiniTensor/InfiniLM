"""
HF timing with the same prompt tokenization and sampling knobs as jiuge.py (batch 1).

**Interpreter:** use ``$REPO/.venv-no-vllm/bin/python`` (HF parity: ``transformers==4.57.1``; see
``InfiniLM/examples/setup_hf_parity_venv.sh``). Do **not** reuse ``.venv-vllm`` here — that venv
pulls **transformers>=5** for vLLM / ``TransformersMoEForCausalLM`` and can hit LongRoPE / ``meta``
init issues with this checkpoint under plain HF ``from_pretrained``.

Measures:
  - model load (from_pretrained + .to(cuda) + eval)
  - prefill: one forward over the full prompt (use_cache=True)
  - decode: max_new_tokens greedy steps (top_k=1 / argmax), one CUDA sync per step
  - optional: transformers generate() wall time for cross-check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from packaging import version
from tokenizers import decoders as _dec
from transformers import AutoModelForCausalLM, AutoTokenizer
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--prompt", type=str, default="Hi")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--cuda-device", type=int, default=0)
    ap.add_argument("--json", action="store_true", help="Print one JSON object with metrics.")
    ap.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="If set, save metrics JSON to this path.",
    )
    args = ap.parse_args()

    if args.batch_size != 1:
        raise SystemExit("Only batch_size=1 is implemented (matches jiuge single-prompt path).")

    if version.parse(transformers.__version__) >= version.parse("5.0.0"):
        print(
            "hf_bench_match_jiuge: transformers>=5 is for the vLLM venv only; "
            "use: bash InfiniLM/examples/setup_hf_parity_venv.sh && "
            'source "$REPO/.venv-no-vllm/bin/activate"',
            file=sys.stderr,
        )

    model_path = os.path.expanduser(args.model_path)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    dev = torch.device(f"cuda:{args.cuda_device}")

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = model.to(dev).eval()
    if torch.cuda.is_available():
        torch.cuda.synchronize(dev)
    load_s = time.perf_counter() - t0

    _maybe_fix_llama_tokenizer_decoder(tokenizer, getattr(model.config, "model_type", ""))

    input_ids_list = [_encode_like_jiuge(tokenizer, args.prompt)]
    input_ids = torch.tensor([input_ids_list[0]], dtype=torch.long, device=dev)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    # --- Prefill ---
    with torch.no_grad():
        torch.cuda.synchronize(dev)
        t1 = time.perf_counter()
        out = model(input_ids=input_ids, use_cache=True)
        torch.cuda.synchronize(dev)
        prefill_s = time.perf_counter() - t1

    past = out.past_key_values
    # Greedy next token (same as top_k=1 sampling when taking the top-1 token).
    cur = out.logits[:, -1:, :].argmax(dim=-1)

    # --- Decode: max_new_tokens steps ---
    decode_step_s: list[float] = []
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            torch.cuda.synchronize(dev)
            t_step = time.perf_counter()
            out = model(
                input_ids=cur,
                past_key_values=past,
                use_cache=True,
            )
            torch.cuda.synchronize(dev)
            decode_step_s.append(time.perf_counter() - t_step)
            cur = out.logits[:, -1:, :].argmax(dim=-1)
            past = out.past_key_values

    decode_total_s = sum(decode_step_s)
    decode_avg_ms = (decode_total_s / len(decode_step_s)) * 1000.0 if decode_step_s else 0.0

    # --- transformers.generate() wall time (includes prefill internally) ---
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.top_k > 1 or args.temperature != 1.0,
        top_k=args.top_k if args.top_k > 0 else None,
        top_p=args.top_p,
        temperature=args.temperature,
        pad_token_id=pad_id,
        use_cache=True,
    )
    if gen_kwargs["do_sample"] is False:
        gen_kwargs.pop("top_k", None)
        gen_kwargs.pop("top_p", None)
        gen_kwargs.pop("temperature", None)

    with torch.no_grad():
        torch.cuda.synchronize(dev)
        tg0 = time.perf_counter()
        _ = model.generate(input_ids=input_ids, **gen_kwargs)
        torch.cuda.synchronize(dev)
        generate_api_s = time.perf_counter() - tg0

    n_prompt = int(input_ids.shape[1])
    row = {
        "engine": "hf_transformers",
        "model_path": model_path,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "prompt": args.prompt,
        "prompt_tokens": n_prompt,
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "cuda_device": args.cuda_device,
        "load_weights_s": load_s,
        "prefill_s": prefill_s,
        "prefill_ms": prefill_s * 1000.0,
        "decode_total_s": decode_total_s,
        "decode_avg_step_ms": decode_avg_ms,
        "prefill_decode_total_s": prefill_s + decode_total_s,
        "generate_api_s": generate_api_s,
        "prefill_tok_per_s": (n_prompt / prefill_s) if prefill_s > 0 else 0.0,
        "decode_tok_per_s": ((args.max_new_tokens - 1) / decode_total_s)
        if args.max_new_tokens > 1 and decode_total_s > 0
        else 0.0,
    }

    if args.json_out:
        out_path = os.path.abspath(os.path.expanduser(args.json_out))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(row, f, indent=2)

    if args.json:
        print(json.dumps(row, indent=2))
        return

    print("== HF bench (match jiuge tokenization / batch 1) ==")
    print(f"model_path: {model_path}")
    print(f"prompt:         {args.prompt!r}")
    print(f"prompt tokens:  {input_ids.shape[1]}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"dtype:          {args.dtype}")
    print(f"load_weights:   {load_s * 1000:.2f} ms  ({load_s:.3f} s)")
    print(f"prefill (1 fw): {prefill_s * 1000:.2f} ms")
    print(f"decode total:     {decode_total_s * 1000:.2f} ms  ({args.max_new_tokens} steps)")
    print(f"decode avg/step:{decode_avg_ms:.2f} ms")
    print(f"prefill+decode: {(prefill_s + decode_total_s) * 1000:.2f} ms")
    print(f"generate() API: {generate_api_s * 1000:.2f} ms  (sync wall)")


if __name__ == "__main__":
    main()
