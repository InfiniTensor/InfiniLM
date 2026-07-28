#!/usr/bin/env python3
"""Dependency-light equivalent of `vllm bench serve --backend openai-chat`.

It preserves vLLM's random dataset construction, burst submission, streaming
OpenAI request shape, and tokenizer-based output accounting without importing
vLLM's CUDA engine extensions.
"""

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass, field

import aiohttp
import numpy as np
from transformers import AutoTokenizer


@dataclass
class Result:
    success: bool = False
    text: str = ""
    latency: float = 0.0
    ttft: float = 0.0
    itl: list[float] = field(default_factory=list)
    error: str = ""
    prompt_len: int = 0
    output_tokens: int = 0


def make_prompts(tokenizer, count: int, input_len: int, seed: int):
    rng = np.random.default_rng(seed)
    special = int(tokenizer.num_special_tokens_to_add())
    inner_len = max(0, input_len - special)
    offsets = rng.integers(0, tokenizer.vocab_size, size=count)
    prompts = []
    for i, offset in enumerate(offsets):
        ids = ((int(offset) + i + np.arange(inner_len)) % tokenizer.vocab_size).tolist()
        prompt = tokenizer.decode(ids)
        ids = tokenizer.encode(prompt, add_special_tokens=False)[:inner_len]
        prompts.append((tokenizer.decode(ids), len(ids) + special))
    return prompts


async def one(session, sem, url, model, prompt, prompt_len, output_len):
    result = Result(prompt_len=prompt_len)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "temperature": 0.0,
        "max_completion_tokens": output_len,
        "max_tokens": output_len,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    async with sem:
        start = time.perf_counter()
        recent = start
        buffer = ""
        try:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    result.error = f"HTTP {response.status}: {await response.text()}"
                    return result
                async for raw in response.content.iter_any():
                    buffer += raw.decode("utf-8")
                    while "\n\n" in buffer:
                        message, buffer = buffer.split("\n\n", 1)
                        message = message.strip()
                        if not message or message.startswith(":"):
                            continue
                        chunk = message.removeprefix("data: ")
                        if chunk == "[DONE]":
                            continue
                        data = json.loads(chunk)
                        now = time.perf_counter()
                        if choices := data.get("choices"):
                            content = choices[0].get("delta", {}).get("content") or ""
                            if result.ttft == 0.0:
                                result.ttft = now - start
                            else:
                                result.itl.append(now - recent)
                            result.text += content
                            recent = now
                        elif usage := data.get("usage"):
                            result.output_tokens = usage.get("completion_tokens") or 0
                result.latency = recent - start
                result.success = result.ttft > 0.0
        except Exception as exc:
            result.error = repr(exc)
        return result


async def run(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    prompts = make_prompts(
        tokenizer, args.num_prompts, args.random_input_len, args.seed
    )
    connector = aiohttp.TCPConnector(
        limit=args.max_concurrency, limit_per_host=args.max_concurrency
    )
    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)
    sem = asyncio.Semaphore(args.max_concurrency)
    async with aiohttp.ClientSession(
        connector=connector, timeout=timeout, trust_env=False
    ) as session:
        # Match vLLM's readiness request, excluded from measured duration.
        warm = await one(
            session, sem, args.url, args.model, *prompts[0], args.random_output_len
        )
        if not warm.success:
            raise RuntimeError(f"warmup failed: {warm.error}")
        start = time.perf_counter()
        tasks = [
            one(session, sem, args.url, args.model, p, n, args.random_output_len)
            for p, n in prompts
        ]
        results = await asyncio.gather(*tasks)
        duration = time.perf_counter() - start

    actual = []
    for r in results:
        if r.success:
            actual.append(
                r.output_tokens
                or len(tokenizer(r.text, add_special_tokens=False).input_ids)
            )
        else:
            actual.append(0)
    good = [r for r in results if r.success]
    total_input = sum(r.prompt_len for r in good)
    total_output = sum(actual)
    ttfts = [r.ttft * 1000 for r in good]
    tpots = [
        (r.latency - r.ttft) / (n - 1) * 1000
        for r, n in zip(results, actual)
        if r.success and n > 1
    ]
    report = {
        "config": vars(args),
        "duration": duration,
        "completed": len(good),
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "request_throughput": len(good) / duration,
        "output_throughput": total_output / duration,
        "total_token_throughput": (total_input + total_output) / duration,
        "mean_ttft_ms": float(np.mean(ttfts or [0])),
        "p99_ttft_ms": float(np.percentile(ttfts or [0], 99)),
        "mean_tpot_ms": float(np.mean(tpots or [0])),
        "actual_output_lens": actual,
        "actual_prompt_lens": [r.prompt_len for r in results],
        "errors": [r.error for r in results if not r.success],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.save_result:
        with open(args.save_result, "w", encoding="utf-8") as f:
            json.dump(
                {**report, "results": [asdict(r) for r in results]},
                f,
                ensure_ascii=False,
                indent=2,
            )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8102/v1/chat/completions")
    p.add_argument("--model", default="Qwen3-8B")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--num-prompts", type=int, required=True)
    p.add_argument("--max-concurrency", type=int, required=True)
    p.add_argument("--random-input-len", type=int, required=True)
    p.add_argument("--random-output-len", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-result")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
