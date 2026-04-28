"""
Drop-in replacement for `vllm bench serve` against an OpenAI-compatible
chat/completions endpoint, sized for the InfiniLM inference_server.

Usage:
    python bench_client.py \
        --tokenizer /data/...           # path to HF tokenizer dir
        --port 8102 \
        --model 9g_8b_thinking \
        --num-prompts 200 \
        --max-concurrency 16 \
        --random-input-len 256 \
        --random-output-len 256 \
        --seed 42

Reports: TTFT (mean/p50/p99), ITL (mean/p50/p99), TPOT, E2EL, request and
token throughput. Output as a single JSON line on stdout, plus a human
readable summary on stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import List

import httpx
from transformers import AutoTokenizer


@dataclass
class RequestStat:
    ttft: float = 0.0
    itl: List[float] = field(default_factory=list)
    e2el: float = 0.0
    output_tokens: int = 0
    success: bool = False


def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def gen_random_token_prompt(tokenizer, length: int, vocab_size: int, rng: random.Random) -> str:
    """Sample `length` random valid token ids and decode to string."""
    # Pick from the lower-half of vocab to avoid special/added tokens.
    cap = min(vocab_size, max(1024, vocab_size // 2))
    ids = [rng.randint(100, cap - 1) for _ in range(length)]
    return tokenizer.decode(ids, skip_special_tokens=True)


async def issue_one(
    client: httpx.AsyncClient,
    url: str,
    payload: dict,
    stat: RequestStat,
) -> None:
    t0 = time.perf_counter()
    last_token_t = None
    try:
        async with client.stream("POST", url, json=payload, timeout=600.0) as resp:
            if resp.status_code != 200:
                _ = await resp.aread()
                return
            async for raw in resp.aiter_lines():
                if not raw or not raw.startswith("data:"):
                    continue
                line = raw[5:].strip()
                if line == "[DONE]":
                    break
                now = time.perf_counter()
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                choice = (obj.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content is None and choice.get("finish_reason") is None:
                    continue
                if content is None:
                    break
                if not stat.ttft:
                    stat.ttft = now - t0
                else:
                    stat.itl.append(now - last_token_t)
                last_token_t = now
                stat.output_tokens += 1
            stat.e2el = time.perf_counter() - t0
            stat.success = True
    except Exception:
        return


async def run(args: argparse.Namespace) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rng = random.Random(args.seed)
    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)

    prompts = [
        gen_random_token_prompt(tokenizer, args.random_input_len, vocab_size, rng)
        for _ in range(args.num_prompts)
    ]
    url = f"http://{args.host}:{args.port}/v1/chat/completions"

    sem = asyncio.Semaphore(args.max_concurrency)
    stats: List[RequestStat] = [RequestStat() for _ in prompts]

    limits = httpx.Limits(max_connections=args.max_concurrency * 2,
                          max_keepalive_connections=args.max_concurrency)
    async with httpx.AsyncClient(limits=limits, timeout=None) as client:
        async def one(i: int):
            async with sem:
                payload = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompts[i]}],
                    "stream": True,
                    "max_tokens": args.random_output_len,
                    "temperature": 1.0,
                    "top_p": 0.8,
                    "ignore_eos": args.ignore_eos,
                    "include_stop_str_in_output": False,
                }
                await issue_one(client, url, payload, stats[i])

        wall_start = time.perf_counter()
        await asyncio.gather(*[one(i) for i in range(args.num_prompts)])
        wall_total = time.perf_counter() - wall_start

    successes = [s for s in stats if s.success]
    n_ok = len(successes)
    n_fail = len(stats) - n_ok
    total_out = sum(s.output_tokens for s in successes)
    ttfts = [s.ttft * 1000 for s in successes]
    itls_flat: List[float] = []
    for s in successes:
        itls_flat.extend(x * 1000 for x in s.itl)
    e2els = [s.e2el * 1000 for s in successes]
    tpots = [
        ((s.e2el - s.ttft) * 1000 / max(1, s.output_tokens - 1))
        for s in successes if s.output_tokens > 1
    ]

    def stat_block(values: List[float]) -> dict:
        if not values:
            return {"mean": 0, "p50": 0, "p99": 0, "min": 0, "max": 0}
        return {
            "mean": statistics.mean(values),
            "p50": percentile(values, 50),
            "p99": percentile(values, 99),
            "min": min(values),
            "max": max(values),
        }

    result = {
        "config": {
            "model": args.model,
            "num_prompts": args.num_prompts,
            "max_concurrency": args.max_concurrency,
            "random_input_len": args.random_input_len,
            "random_output_len": args.random_output_len,
            "seed": args.seed,
        },
        "successful_requests": n_ok,
        "failed_requests": n_fail,
        "wall_time_s": wall_total,
        "request_throughput_per_s": n_ok / wall_total if wall_total else 0,
        "output_token_throughput_per_s": total_out / wall_total if wall_total else 0,
        "total_output_tokens": total_out,
        "ttft_ms": stat_block(ttfts),
        "itl_ms": stat_block(itls_flat),
        "tpot_ms": stat_block(tpots),
        "e2el_ms": stat_block(e2els),
    }
    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--model", default="9g_8b_thinking")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8102)
    p.add_argument("--num-prompts", type=int, default=200)
    p.add_argument("--max-concurrency", type=int, default=16)
    p.add_argument("--random-input-len", type=int, default=256)
    p.add_argument("--random-output-len", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ignore-eos", action="store_true", default=True)
    args = p.parse_args()

    res = asyncio.run(run(args))
    json.dump(res, sys.stdout)
    sys.stdout.write("\n")

    cfg = res["config"]
    print(
        f"\n[bench] bs={cfg['max_concurrency']} in={cfg['random_input_len']} "
        f"out={cfg['random_output_len']} n={cfg['num_prompts']} "
        f"ok={res['successful_requests']}/{cfg['num_prompts']}",
        file=sys.stderr,
    )
    print(
        f"  req_throughput  : {res['request_throughput_per_s']:.2f} req/s",
        file=sys.stderr,
    )
    print(
        f"  out_token_thrpt : {res['output_token_throughput_per_s']:.1f} tok/s",
        file=sys.stderr,
    )
    print(
        f"  ttft_ms  mean={res['ttft_ms']['mean']:.1f}  p50={res['ttft_ms']['p50']:.1f}  p99={res['ttft_ms']['p99']:.1f}",
        file=sys.stderr,
    )
    print(
        f"  itl_ms   mean={res['itl_ms']['mean']:.2f}  p50={res['itl_ms']['p50']:.2f}  p99={res['itl_ms']['p99']:.2f}",
        file=sys.stderr,
    )
    print(
        f"  tpot_ms  mean={res['tpot_ms']['mean']:.2f}  p50={res['tpot_ms']['p50']:.2f}  p99={res['tpot_ms']['p99']:.2f}",
        file=sys.stderr,
    )
    print(
        f"  e2el_ms  mean={res['e2el_ms']['mean']:.0f}  p50={res['e2el_ms']['p50']:.0f}  p99={res['e2el_ms']['p99']:.0f}",
        file=sys.stderr,
    )
    return 0 if res["failed_requests"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
