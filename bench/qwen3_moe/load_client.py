#!/usr/bin/env python3
"""Dependency-free concurrent load client for the InfiniLM OpenAI server.

A stdlib-only alternative to `vllm bench serve` (no vllm / aiohttp needed), for
environments where vllm is not installed (e.g. MetaX containers). Sweeps the
concurrency × input-len × output-len matrix and writes one result file per
config in the exact label format `summarize.py` parses.

Because the server runs with `--ignore-eos`, each request generates exactly
`output_len` tokens, so output-token throughput is measured against the nominal
count (also reconciled with the response `usage` field when present).

Usage:
    python3 load_client.py --tag this
    python3 load_client.py --tag base --base-url http://127.0.0.1:8102 \
        --batch-sizes 1,8,32 --input-lens 32,256,2048 --output-lens 256,1024

Input length is approximated by repeating a CJK filler char (~1 token each);
this is fine for relative base-vs-this comparison. Pass --tokenizer to count
prompt tokens exactly if `transformers` is available.
"""

import argparse
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

FILLER = "的"  # ~1 Qwen token each; used to build an approximately input_len prompt.


def build_prompt(input_len, tokenizer=None):
    """Return a user message string of approximately `input_len` tokens."""
    if tokenizer is not None:
        # Grow the filler until the tokenizer reports >= input_len tokens.
        s = FILLER * input_len
        while len(tokenizer(s)["input_ids"]) < input_len:
            s += FILLER * 8
        return s
    return FILLER * input_len


def one_request(base_url, model, prompt, output_len, timeout):
    """Send one non-streaming chat request; return (ok, latency_s, completion_tokens)."""
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": output_len,
            "ignore_eos": True,
            "stream": False,
            "temperature": 1.0,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001 - report and keep going
        return False, time.perf_counter() - t0, 0, str(e)
    lat = time.perf_counter() - t0
    # Prefer server-reported usage; fall back to nominal output_len.
    comp = output_len
    usage = body.get("usage") or {}
    if (
        isinstance(usage.get("completion_tokens"), int)
        and usage["completion_tokens"] > 0
    ):
        comp = usage["completion_tokens"]
    return True, lat, comp, None


def run_config(base_url, model, bs, il, ol, repeat, timeout, tokenizer):
    num_prompts = bs * repeat
    prompt = build_prompt(il, tokenizer)
    results = []
    err_sample = None

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=bs) as pool:
        futs = [
            pool.submit(one_request, base_url, model, prompt, ol, timeout)
            for _ in range(num_prompts)
        ]
        for f in as_completed(futs):
            ok, lat, comp, err = f.result()
            results.append((ok, lat, comp))
            if not ok and err_sample is None:
                err_sample = err
    wall = time.perf_counter() - t_start

    ok_n = sum(1 for ok, _, _ in results if ok)
    out_tokens = sum(c for ok, _, c in results if ok)
    in_tokens = ok_n * il  # approximate (see build_prompt)
    lat_ok = [lat for ok, lat, _ in results if ok]
    mean_lat_ms = 1000.0 * sum(lat_ok) / len(lat_ok) if lat_ok else 0.0

    metrics = {
        "successful": ok_n,
        "num_prompts": num_prompts,
        "duration_s": wall,
        "req_s": ok_n / wall if wall > 0 else 0.0,
        "out_tok_s": out_tokens / wall if wall > 0 else 0.0,
        "total_tok_s": (in_tokens + out_tokens) / wall if wall > 0 else 0.0,
        "mean_latency_ms": mean_lat_ms,
        "err_sample": err_sample,
    }
    return metrics


def write_result(path, bs, il, ol, m):
    # Labels chosen to match summarize.py's regexes.
    lines = [
        "============ Serving Benchmark Result ============",
        f"config:                                  bs={bs} in={il} out={ol}",
        f"Successful requests:                     {m['successful']}/{m['num_prompts']}",
        f"Benchmark duration (s):                  {m['duration_s']:.2f}",
        f"Request throughput (req/s):              {m['req_s']:.4f}",
        f"Output token throughput (tok/s):         {m['out_tok_s']:.2f}",
        f"Total Token throughput (tok/s):          {m['total_tok_s']:.2f}",
        f"Mean E2E latency (ms):                   {m['mean_latency_ms']:.2f}",
    ]
    if m["err_sample"]:
        lines.append(f"error_sample:                            {m['err_sample']}")
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def parse_list(s):
    return [int(x) for x in s.replace(",", " ").split()]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:8102")
    ap.add_argument("--model", default="Qwen3-30B-A3B-Thinking-2507")
    ap.add_argument("--tag", default="this", help="result subdir under --outdir-root")
    ap.add_argument("--outdir-root", default="bench_results")
    ap.add_argument("--batch-sizes", type=parse_list, default=[1, 8, 32])
    ap.add_argument("--input-lens", type=parse_list, default=[32, 256, 2048])
    ap.add_argument("--output-lens", type=parse_list, default=[256, 1024])
    ap.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="requests per config = concurrency * repeat",
    )
    ap.add_argument("--timeout", type=float, default=1200.0)
    ap.add_argument(
        "--tokenizer",
        default=None,
        help="optional HF tokenizer path for exact prompt token counts",
    )
    args = ap.parse_args()

    tok = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        except Exception as e:  # noqa: BLE001
            print(f"(tokenizer load failed, using approximate prompt: {e})")

    outdir = os.path.join(args.outdir_root, args.tag)
    os.makedirs(outdir, exist_ok=True)
    print(f">>> tag={args.tag} url={args.base_url} model={args.model} -> {outdir}")
    print(
        f">>> concurrency={args.batch_sizes} input={args.input_lens} output={args.output_lens}"
    )

    for bs in args.batch_sizes:
        for il in args.input_lens:
            for ol in args.output_lens:
                path = os.path.join(outdir, f"bs{bs}_in{il}_out{ol}.txt")
                print(f">>> bs={bs} in={il} out={ol} ...", end="", flush=True)
                m = run_config(
                    args.base_url,
                    args.model,
                    bs,
                    il,
                    ol,
                    args.repeat,
                    args.timeout,
                    tok,
                )
                write_result(path, bs, il, ol, m)
                if m["successful"] == 0:
                    print(f" FAIL ({m['err_sample']})")
                else:
                    print(
                        f" ok  out={m['out_tok_s']:.1f} tok/s  total={m['total_tok_s']:.1f} tok/s"
                        f"  ({m['successful']}/{m['num_prompts']})"
                    )

    print(
        f">>> done. 汇总: python3 {os.path.join(os.path.dirname(__file__), 'summarize.py')} {outdir}"
    )


if __name__ == "__main__":
    main()
