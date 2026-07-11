"""Service-style benchmark for chunked prefill scheduling.

This benchmark exercises the LLMEngine scheduler path, not the lower-level
InferEngine.generate path used by examples/bench.py. It launches several short
decode-heavy requests, inserts one long-prefill request after the short requests
have started decoding, and reports short-request tail ITL plus long-request TTFT.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from infinilm.llm import AsyncLLMEngine, SamplingParams
from infinilm.moe_config import configure_moe_ep_backend


DEVICE_STR_MAP = {
    "cpu": "cpu",
    "nvidia": "cuda",
    "qy": "cuda",
    "cuda": "cuda",
    "cambricon": "mlu",
    "ascend": "npu",
    "metax": "cuda",
    "moore": "musa",
    "iluvatar": "cuda",
    "kunlun": "kunlun",
    "hygon": "cuda",
    "ali": "cuda",
}


@dataclass
class RequestMetrics:
    request_id: str
    kind: str
    prompt_len: int
    output_len: int
    submit_time: float
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    token_times: list[float] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    finish_reason: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self, origin: float) -> dict[str, Any]:
        ttft_ms = None
        if self.first_token_time is not None:
            ttft_ms = (self.first_token_time - self.submit_time) * 1000.0

        total_ms = None
        if self.finish_time is not None:
            total_ms = (self.finish_time - self.submit_time) * 1000.0

        itl_ms = []
        for prev, cur in zip(self.token_times, self.token_times[1:]):
            itl_ms.append((cur - prev) * 1000.0)

        return {
            "request_id": self.request_id,
            "kind": self.kind,
            "prompt_len": self.prompt_len,
            "output_len": self.output_len,
            "submit_s": self.submit_time - origin,
            "first_token_s": None
            if self.first_token_time is None
            else self.first_token_time - origin,
            "finish_s": None
            if self.finish_time is None
            else self.finish_time - origin,
            "num_tokens": len(self.token_times),
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
            "itl_ms": itl_ms,
            "finish_reason": self.finish_reason,
            "error": self.error,
        }


def percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def summarize(values: list[float]) -> dict[str, Optional[float]]:
    return {
        "count": len(values),
        "avg_ms": sum(values) / len(values) if values else None,
        "p50_ms": percentile(values, 0.50),
        "p90_ms": percentile(values, 0.90),
        "p99_ms": percentile(values, 0.99),
        "max_ms": max(values) if values else None,
    }


def normalize_device(device: str) -> str:
    return DEVICE_STR_MAP.get(device.lower(), device)


def repeat_tokens(tokens: list[int], target_len: int) -> list[int]:
    if target_len <= 0:
        return []
    if not tokens:
        raise ValueError("Cannot repeat an empty prompt token list")
    repeat = (target_len + len(tokens) - 1) // len(tokens)
    return (tokens * repeat)[:target_len]


def build_prompt_tokens(engine: AsyncLLMEngine, prompt: str, target_len: int) -> list[int]:
    tokens = engine.engine.tokenize(prompt)
    return repeat_tokens(tokens, target_len)


async def collect_request(
    engine: AsyncLLMEngine,
    request,
    metrics: RequestMetrics,
    stream_timeout_s: float,
    request_timeout_s: float,
) -> RequestMetrics:
    try:
        async for token_output in engine.stream_request(
            request,
            timeout=stream_timeout_s,
            request_timeout=request_timeout_s,
        ):
            now = time.perf_counter()
            if metrics.first_token_time is None:
                metrics.first_token_time = now
            if token_output.token_id >= 0:
                metrics.token_times.append(now)
                metrics.token_ids.append(token_output.token_id)
            if token_output.finished:
                metrics.finish_reason = (
                    token_output.finish_reason.value
                    if hasattr(token_output.finish_reason, "value")
                    else str(token_output.finish_reason)
                )
                break
        metrics.finish_time = time.perf_counter()
    except Exception as exc:  # noqa: BLE001 - benchmark should report failures.
        metrics.error = repr(exc)
        metrics.finish_time = time.perf_counter()
    return metrics


async def wait_for_short_progress(
    short_metrics: list[RequestMetrics],
    tokens: int,
    timeout_s: float,
) -> bool:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if all(len(metric.token_times) >= tokens for metric in short_metrics):
            return True
        await asyncio.sleep(0.001)
    return False


def itls_after_insert(
    metrics: list[RequestMetrics], insert_time: float
) -> list[float]:
    values = []
    for metric in metrics:
        for prev, cur in zip(metric.token_times, metric.token_times[1:]):
            if cur >= insert_time:
                values.append((cur - prev) * 1000.0)
    return values


def all_itls(metrics: list[RequestMetrics]) -> list[float]:
    values = []
    for metric in metrics:
        for prev, cur in zip(metric.token_times, metric.token_times[1:]):
            values.append((cur - prev) * 1000.0)
    return values


def add_engine_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="nvidia")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=None)
    parser.add_argument("--moe-ep-backend", default="auto")
    parser.add_argument("--moe-ep-size", type=int, default=None)
    parser.add_argument("--skip-legacy-moe", action="store_true")
    parser.add_argument("--allreduce-backend", default="nccl", choices=["nccl", "auto", "custom"])
    parser.add_argument("--attn", default="flash-attn", choices=["default", "paged-attn", "flash-attn"])
    parser.add_argument("--enable-graph", action="store_true")
    parser.add_argument("--use-mla", action="store_true")
    parser.add_argument("--weight-load", dest="weight_load_mode", default="async", choices=["async", "sync"])
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--max-cache-len", type=int, default=4096)


def add_chunked_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--prefill-chunk-size", type=int, default=None)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--decode-priority", dest="decode_priority", action="store_true", default=None)
    group.add_argument("--no-decode-priority", dest="decode_priority", action="store_false")
    parser.add_argument("--max-num-partial-prefills", type=int, default=1)
    parser.add_argument("--max-long-partial-prefills", type=int, default=1)
    parser.add_argument("--long-prefill-token-threshold", type=int, default=None)
    parser.add_argument("--min-prefill-chunk-size", type=int, default=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_engine_args(parser)
    add_chunked_args(parser)
    parser.add_argument("--num-short-requests", type=int, default=4)
    parser.add_argument("--short-input-len", type=int, default=128)
    parser.add_argument("--short-output-len", type=int, default=64)
    parser.add_argument("--long-input-len", type=int, default=12800)
    parser.add_argument("--long-output-len", type=int, default=4)
    parser.add_argument("--insert-after-tokens", type=int, default=8)
    parser.add_argument("--prompt-file", default="examples/bench_prompt.md")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--stream-timeout-s", type=float, default=120.0)
    parser.add_argument("--request-timeout-s", type=float, default=1200.0)
    parser.add_argument("--progress-timeout-s", type=float, default=120.0)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def load_prompt(path: str) -> str:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "Explain the history and geography of Mount Tai in detail."


async def run_once(args: argparse.Namespace) -> dict[str, Any]:
    required_cache_len = max(
        args.short_input_len + args.short_output_len,
        args.long_input_len + args.long_output_len,
    )
    if args.max_cache_len < required_cache_len:
        args.max_cache_len = required_cache_len

    requested_ep = args.ep if args.ep is not None else args.moe_ep_size
    moe_ep_backend, moe_ep_size = configure_moe_ep_backend(
        args.tp, args.dp, requested_ep, args.moe_ep_backend, args.model
    )
    args.moe_ep_backend = moe_ep_backend
    args.moe_ep_size = moe_ep_size
    print(f"MoE EP backend: {moe_ep_backend}  TP={args.tp}  DP={args.dp}  EP={moe_ep_size}")

    engine = AsyncLLMEngine(
        model_path=args.model,
        device=normalize_device(args.device),
        dtype=args.dtype,
        tensor_parallel_size=args.tp,
        moe_ep_backend=moe_ep_backend,
        moe_ep_size=moe_ep_size,
        allreduce_backend=args.allreduce_backend,
        cache_type="paged",
        max_batch_size=args.max_batch_size,
        max_tokens=max(args.short_output_len, args.long_output_len),
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        max_cache_len=args.max_cache_len,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        enable_graph=args.enable_graph,
        attn_backend=args.attn,
        use_mla=args.use_mla,
        weight_load_mode=args.weight_load_mode,
        skip_legacy_moe=args.skip_legacy_moe,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=args.enable_chunked_prefill,
        prefill_chunk_size=args.prefill_chunk_size,
        decode_priority=args.decode_priority,
        max_num_partial_prefills=args.max_num_partial_prefills,
        max_long_partial_prefills=args.max_long_partial_prefills,
        long_prefill_token_threshold=args.long_prefill_token_threshold,
        min_prefill_chunk_size=args.min_prefill_chunk_size,
    )

    origin = time.perf_counter()
    tasks: list[asyncio.Task] = []
    short_metrics: list[RequestMetrics] = []
    long_metric: Optional[RequestMetrics] = None
    insert_time: Optional[float] = None
    progress_ready = False

    try:
        engine.start()
        prompt = load_prompt(args.prompt_file)
        short_tokens = build_prompt_tokens(engine, prompt, args.short_input_len)
        long_tokens = build_prompt_tokens(engine, prompt, args.long_input_len)

        if args.warmup:
            warmup_req = engine.add_request(
                messages=None,
                prompt_token_ids=short_tokens,
                sampling_params=SamplingParams(max_tokens=2, top_k=1, top_p=1.0, ignore_eos=True),
                request_id="warmup",
            )
            warmup_metrics = RequestMetrics("warmup", "warmup", len(short_tokens), 2, time.perf_counter())
            await collect_request(engine, warmup_req, warmup_metrics, args.stream_timeout_s, args.request_timeout_s)

        for idx in range(args.num_short_requests):
            req_id = f"short-{idx}"
            submit = time.perf_counter()
            req = engine.add_request(
                messages=None,
                prompt_token_ids=short_tokens,
                sampling_params=SamplingParams(
                    max_tokens=args.short_output_len,
                    top_k=1,
                    top_p=1.0,
                    temperature=1.0,
                    ignore_eos=True,
                ),
                request_id=req_id,
            )
            metric = RequestMetrics(req_id, "short", len(short_tokens), args.short_output_len, submit)
            short_metrics.append(metric)
            tasks.append(
                asyncio.create_task(
                    collect_request(
                        engine,
                        req,
                        metric,
                        args.stream_timeout_s,
                        args.request_timeout_s,
                    )
                )
            )

        progress_ready = await wait_for_short_progress(
            short_metrics,
            args.insert_after_tokens,
            args.progress_timeout_s,
        )

        insert_time = time.perf_counter()
        long_req = engine.add_request(
            messages=None,
            prompt_token_ids=long_tokens,
            sampling_params=SamplingParams(
                max_tokens=args.long_output_len,
                top_k=1,
                top_p=1.0,
                temperature=1.0,
                ignore_eos=True,
            ),
            request_id="long",
        )
        long_metric = RequestMetrics(
            "long",
            "long",
            len(long_tokens),
            args.long_output_len,
            insert_time,
        )
        tasks.append(
            asyncio.create_task(
                collect_request(
                    engine,
                    long_req,
                    long_metric,
                    args.stream_timeout_s,
                    args.request_timeout_s,
                )
            )
        )

        await asyncio.wait_for(asyncio.gather(*tasks), timeout=args.request_timeout_s)
    finally:
        engine.stop()

    short_after = itls_after_insert(short_metrics, insert_time or origin)
    short_all = all_itls(short_metrics)
    long_ttft = None
    long_total = None
    if long_metric and long_metric.first_token_time:
        long_ttft = (long_metric.first_token_time - long_metric.submit_time) * 1000.0
    if long_metric and long_metric.finish_time:
        long_total = (long_metric.finish_time - long_metric.submit_time) * 1000.0

    return {
        "config": vars(args),
        "progress_ready_before_insert": progress_ready,
        "insert_time_s": None if insert_time is None else insert_time - origin,
        "summary": {
            "short_itl_all": summarize(short_all),
            "short_itl_after_long_insert": summarize(short_after),
            "long_ttft_ms": long_ttft,
            "long_total_ms": long_total,
        },
        "cache_stats": engine.get_cache_stats(),
        "requests": [m.to_dict(origin) for m in short_metrics]
        + ([] if long_metric is None else [long_metric.to_dict(origin)]),
    }


def main() -> None:
    args = parse_args()
    result = asyncio.run(run_once(args))
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")


if __name__ == "__main__":
    main()
