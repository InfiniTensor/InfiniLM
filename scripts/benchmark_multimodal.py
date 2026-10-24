#!/usr/bin/env python3
# Fixed-length OpenAI-compatible multimodal serving benchmark.

import argparse
import asyncio
import base64
import io
import json
import math
import random
import statistics
import time
from pathlib import Path

import aiohttp
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--text-len", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--responses-output")
    return parser.parse_args()


def load_font():
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, 104)
    return ImageFont.load_default()


def make_image(index, width, height, font):
    rng = random.Random(10000 + index)
    image = Image.new(
        "RGB",
        (width, height),
        (rng.randrange(48, 208), rng.randrange(48, 208), rng.randrange(48, 208)),
    )
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, width - 9, height - 9), outline=(16, 16, 16), width=8)
    number = f"{730 + index:03d}"
    box = draw.textbbox((0, 0), number, font=font)
    position = ((width - box[2] + box[0]) // 2, (height - box[3] + box[1]) // 2)
    draw.text(position, number, fill=(16, 16, 16), font=font)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", compresslevel=1)
    return number, base64.b64encode(buffer.getvalue()).decode("ascii")


def make_filler(tokenizer, target_tokens, seed):
    words = (
        "amber",
        "baseline",
        "copper",
        "delta",
        "east",
        "format",
        "grid",
        "harbor",
        "input",
        "jade",
        "kernel",
        "layout",
        "metric",
        "network",
        "output",
        "pattern",
        "query",
        "record",
    )
    rng = random.Random(seed)
    tokens = []
    while len(tokens) < target_tokens:
        word = words[rng.randrange(len(words))]
        tokens.extend(tokenizer.encode(" " + word, add_special_tokens=False))
    return tokenizer.decode(tokens[:target_tokens])


async def one_request(session, url, request, started):
    first_chunk = None
    finish = None
    output_parts = []
    error = None
    status = None
    try:
        async with session.post(url, json=request) as response:
            status = response.status
            if status != 200:
                error = (await response.text())[:1000]
            else:
                async for raw_line in response.content:
                    if not raw_line.startswith(b"data: "):
                        continue
                    payload = raw_line[6:].strip()
                    if payload == b"[DONE]":
                        break
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    content = (choices[0].get("delta") or {}).get("content")
                    if content:
                        if first_chunk is None:
                            first_chunk = time.perf_counter()
                        output_parts.append(content)
                finish = time.perf_counter()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    output = "".join(output_parts)
    if error is None and output.startswith("[Error:"):
        error = output
    return {
        "ok": status == 200 and error is None and bool(output),
        "status": status,
        "error": error,
        "ttft_ms": (first_chunk - started) * 1000 if first_chunk else None,
        "e2e_ms": (finish - started) * 1000 if finish else None,
        "output": output,
    }


async def run_case(args, font, warmup=False):
    concurrency = 1 if warmup else args.concurrency
    text_len = 4 if warmup else args.text_len
    output_len = 1 if warmup else args.output_len
    seed = args.seed + (100000 if warmup else 0)
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, trust_remote_code=True
        )

    requests = []
    expected = []
    for index in range(concurrency):
        number, encoded = make_image(index, args.image_width, args.image_height, font)
        filler = (
            " test" * text_len
            if tokenizer is None
            else make_filler(tokenizer, text_len, seed + index)
        )
        expected.append(number)
        requests.append(
            {
                "model": args.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encoded}"
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "The image contains a three-digit number. "
                                    f"Repeat that number first. Filler:{filler}"
                                ),
                            },
                        ],
                    }
                ],
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": output_len,
                "ignore_eos": True,
                "stream": True,
            }
        )

    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    started = time.perf_counter()
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=max(16, concurrency)),
        timeout=aiohttp.ClientTimeout(total=6 * 3600),
    ) as session:
        results = await asyncio.gather(
            *[one_request(session, url, request, started) for request in requests]
        )
    ended = time.perf_counter()

    successful = [result for result in results if result["ok"]]
    ttfts = [
        result["ttft_ms"] for result in successful if result["ttft_ms"] is not None
    ]
    latencies = [
        result["e2e_ms"] for result in successful if result["e2e_ms"] is not None
    ]
    tpots = [
        (result["e2e_ms"] - result["ttft_ms"]) / max(1, output_len - 1)
        for result in successful
        if result["ttft_ms"] is not None and result["e2e_ms"] is not None
    ]
    duration = ended - started
    return {
        "backend": args.backend,
        "concurrency": concurrency,
        "text_len": text_len,
        "output_len": output_len,
        "image": f"{args.image_width}x{args.image_height}",
        "seed": seed,
        "completed": len(successful),
        "failed": len(results) - len(successful),
        "duration_s": duration,
        "output_throughput": (
            len(successful) * output_len / duration if duration else None
        ),
        "mean_ttft_ms": statistics.fmean(ttfts) if ttfts else None,
        "p99_ttft_ms": (
            sorted(ttfts)[min(len(ttfts) - 1, math.ceil(0.99 * len(ttfts)) - 1)]
            if ttfts
            else None
        ),
        "mean_e2e_ms": statistics.fmean(latencies) if latencies else None,
        "mean_tpot_ms": statistics.fmean(tpots) if tpots else None,
        "expected_numbers": expected,
        "results": results,
    }


def append_jsonl(path, row):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    args = parse_args()
    font = load_font()
    if args.warmup:
        warmup = asyncio.run(run_case(args, font, warmup=True))
        print(
            json.dumps(
                {key: warmup[key] for key in ("completed", "failed", "duration_s")}
            )
        )
    row = asyncio.run(run_case(args, font))
    append_jsonl(args.output, row)
    if args.responses_output:
        append_jsonl(
            args.responses_output,
            {
                "expected": row["expected_numbers"],
                "outputs": [result.get("output", "") for result in row["results"]],
            },
        )
    print(
        json.dumps(
            {
                key: row[key]
                for key in (
                    "backend",
                    "concurrency",
                    "text_len",
                    "output_len",
                    "completed",
                    "failed",
                    "duration_s",
                    "output_throughput",
                    "mean_ttft_ms",
                    "mean_e2e_ms",
                    "mean_tpot_ms",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
