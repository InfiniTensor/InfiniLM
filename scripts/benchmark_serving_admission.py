#!/usr/bin/env python3
"""Run the fixed KV admission manifest against an OpenAI-compatible server."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

try:
    from .kv_admission_manifest import AdmissionManifest, RequestSpec, load_manifest
except ImportError:
    from kv_admission_manifest import AdmissionManifest, RequestSpec, load_manifest


@dataclass
class RequestResult:
    request_id: str
    wave: str
    request_index: int
    started_at: float
    first_token_at: float | None
    finished_at: float
    finish_reason: str | None
    output_tokens: int
    content_chunks: int
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return (
            self.error is None
            and self.first_token_at is not None
            and self.finish_reason in {"length", "stop"}
            and self.output_tokens > 0
        )

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_at is None:
            return None
        return (self.first_token_at - self.started_at) * 1000.0

    @property
    def e2e_ms(self) -> float:
        return (self.finished_at - self.started_at) * 1000.0

    def json_dict(self, origin: float) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "started_s": self.started_at - origin,
                "first_token_s": (
                    None
                    if self.first_token_at is None
                    else self.first_token_at - origin
                ),
                "finished_s": self.finished_at - origin,
                "ttft_ms": self.ttft_ms,
                "e2e_ms": self.e2e_ms,
            }
        )
        for key in ("started_at", "first_token_at", "finished_at"):
            result.pop(key, None)
        return result


class FirstTokenBarrier:
    def __init__(self, target: int, total: int):
        if not 0 <= target <= total:
            raise ValueError("Barrier target must be between zero and `wave1` size.")
        self.target = target
        self.total = total
        self.count = 0
        self.finished_without_first_token = 0
        self.event = asyncio.Event()
        if target == 0:
            self.event.set()

    @property
    def reached(self) -> bool:
        return self.count >= self.target

    @property
    def impossible(self) -> bool:
        return self.total - self.finished_without_first_token < self.target

    def mark_first_token(self) -> None:
        self.count += 1
        if self.reached:
            self.event.set()

    def mark_finished_without_first_token(self) -> None:
        self.finished_without_first_token += 1
        if self.impossible:
            self.event.set()


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def metric_summary(values: Sequence[float]) -> dict[str, float | None]:
    return {
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "mean": sum(values) / len(values) if values else None,
    }


def summarize(results: Sequence[RequestResult]) -> dict[str, Any]:
    successful = [result for result in results if result.succeeded]
    if results:
        duration = max(result.finished_at for result in results) - min(
            result.started_at for result in results
        )
    else:
        duration = 0.0
    tokens = sum(result.output_tokens for result in successful)
    return {
        "attempted_requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "duration_s": duration,
        "output_tokens": tokens,
        "output_tokens_per_second": tokens / duration if duration else 0.0,
        "requests_per_second": len(successful) / duration if duration else 0.0,
        "ttft_ms": metric_summary(
            [result.ttft_ms for result in successful if result.ttft_ms is not None]
        ),
        "e2e_ms": metric_summary([result.e2e_ms for result in successful]),
    }


def predict_pressure(manifest: AdmissionManifest) -> dict[str, Any]:
    wave1 = manifest.wave1
    wave2 = manifest.wave2
    block_size = manifest.block_size
    prompt_blocks = [math.ceil(item.prompt_tokens / block_size) for item in wave1]
    target_blocks = [
        math.ceil((item.prompt_tokens + item.max_tokens) / block_size) for item in wave1
    ]
    exact = [
        max(target - prompt, 0) for target, prompt in zip(target_blocks, prompt_blocks)
    ]
    legacy = [math.ceil(max(item.max_tokens - 1, 0) / block_size) for item in wave1]
    free_after_prompt = max(manifest.num_blocks - sum(prompt_blocks), 0)

    def admitted(reserved: int) -> int:
        available = max(free_after_prompt - reserved, 0)
        count = 0
        for item in wave2:
            required = math.ceil((item.prompt_tokens + item.max_tokens) / block_size)
            if required > available:
                break
            available -= required
            count += 1
        return count

    return {
        "wave1_prompt_blocks": sum(prompt_blocks),
        "legacy_running_reserved_blocks": sum(legacy),
        "exact_running_reserved_blocks": sum(exact),
        "false_reserved_blocks": max(sum(legacy) - sum(exact), 0),
        "predicted_wave2_admissions_legacy": admitted(sum(legacy)),
        "predicted_wave2_admissions_exact": admitted(sum(exact)),
    }


async def issue_request(
    client,
    endpoint: str,
    model: str,
    spec: RequestSpec,
    request_index: int,
    barrier: FirstTokenBarrier | None = None,
    max_tokens: int | None = None,
) -> RequestResult:
    started = time.perf_counter()
    first_token = None
    finished = started
    finish_reason = None
    chunks = 0
    output_tokens = 0
    error = None
    requested_tokens = max_tokens or spec.max_tokens
    body = {
        "model": model,
        "messages": [{"role": "user", "content": spec.prompt}],
        "max_tokens": requested_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "stream": True,
    }
    try:
        async with client.stream("POST", endpoint, json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw == "[DONE]":
                    break
                chunk = json.loads(raw)
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                content = delta.get("content")
                if content is not None and content != "":
                    chunks += 1
                    output_tokens += 1
                    if first_token is None:
                        first_token = time.perf_counter()
                        if barrier is not None:
                            barrier.mark_first_token()
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
                    finished = time.perf_counter()
        if finish_reason is None:
            error = "stream ended without a finish reason"
            finished = time.perf_counter()
    except Exception as exc:  # Benchmark records failures instead of hiding them.
        error = f"{type(exc).__name__}: {exc}"
        finished = time.perf_counter()

    if finish_reason == "length":
        output_tokens = requested_tokens
    if first_token is None and barrier is not None:
        barrier.mark_finished_without_first_token()
    if first_token is None and error is None:
        error = "stream produced no content"

    return RequestResult(
        request_id=spec.request_id,
        wave=spec.wave,
        request_index=request_index,
        started_at=started,
        first_token_at=first_token,
        finished_at=finished,
        finish_reason=finish_reason,
        output_tokens=output_tokens,
        content_chunks=chunks,
        error=error,
    )


async def run_benchmark(
    manifest: AdmissionManifest,
    *,
    base_url: str,
    model: str,
    warmup_requests: int,
    wave2_start_after: int,
    barrier_timeout: float,
    request_timeout: float,
) -> dict[str, Any]:
    import httpx

    wave1 = manifest.wave1
    wave2 = manifest.wave2
    if not 0 <= wave2_start_after <= len(wave1):
        raise ValueError("`wave2_start_after` must be between zero and `wave1` size.")
    endpoint = base_url.rstrip("/") + "/chat/completions"
    limits = httpx.Limits(
        max_connections=len(wave1) + len(wave2),
        max_keepalive_connections=len(wave1) + len(wave2),
    )
    timeout = httpx.Timeout(request_timeout, connect=min(request_timeout, 30.0))
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        health = await client.get(base_url.rstrip("/") + "/models")
        health.raise_for_status()
        warmup_specs = wave1[: min(warmup_requests, len(wave1))]
        if warmup_specs:
            warmup_results = await asyncio.gather(
                *(
                    issue_request(
                        client,
                        endpoint,
                        model,
                        spec,
                        index,
                        max_tokens=min(spec.max_tokens, 8),
                    )
                    for index, spec in enumerate(warmup_specs)
                )
            )
            failures = [item for item in warmup_results if not item.succeeded]
            if failures:
                raise RuntimeError(f"Warmup failed: {failures[0].error}.")

        barrier = FirstTokenBarrier(wave2_start_after, len(wave1))
        benchmark_started = time.perf_counter()
        wave1_tasks = [
            asyncio.create_task(
                issue_request(client, endpoint, model, spec, index, barrier)
            )
            for index, spec in enumerate(wave1)
        ]
        try:
            await asyncio.wait_for(barrier.event.wait(), timeout=barrier_timeout)
        except asyncio.TimeoutError as exc:
            for task in wave1_tasks:
                task.cancel()
            await asyncio.gather(*wave1_tasks, return_exceptions=True)
            raise RuntimeError(
                f"Timed out at `wave2` barrier ({barrier.count}/{barrier.target})."
            ) from exc
        if not barrier.reached:
            for task in wave1_tasks:
                task.cancel()
            await asyncio.gather(*wave1_tasks, return_exceptions=True)
            raise RuntimeError(
                f"`wave2` barrier is impossible ({barrier.count}/{barrier.target})."
            )

        wave2_started = time.perf_counter()
        wave2_tasks = [
            asyncio.create_task(issue_request(client, endpoint, model, spec, index))
            for index, spec in enumerate(wave2)
        ]
        results = await asyncio.gather(*wave1_tasks, *wave2_tasks)

    wave1_results = [item for item in results if item.wave == "wave1"]
    wave2_results = [item for item in results if item.wave == "wave2"]
    summaries = {
        "overall": summarize(results),
        "wave1": summarize(wave1_results),
        "wave2": summarize(wave2_results),
    }
    return {
        "manifest_sha256": manifest.sha256,
        "configuration": {
            "base_url": base_url,
            "model": model,
            "wave1_requests": len(wave1),
            "wave2_requests": len(wave2),
            "wave2_start_after": wave2_start_after,
            "block_size": manifest.block_size,
            "num_blocks": manifest.num_blocks,
            "warmup_requests": len(warmup_specs),
        },
        "pressure_prediction": predict_pressure(manifest),
        "wave2_started_s": wave2_started - benchmark_started,
        "summary": summaries,
        "requests": [item.json_dict(benchmark_started) for item in results],
    }


def environment() -> dict[str, Any]:
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        gpu = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": gpu,
    }


def validate_tokenization(manifest: AdmissionManifest, model_path: str) -> None:
    """Verify that the serving tokenizer reproduces the manifest token ids."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    checked: dict[str, tuple[int, ...]] = {}
    for request in manifest.requests:
        token_ids = checked.get(request.prompt)
        if token_ids is None:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": request.prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            token_ids = tuple(
                int(token_id)
                for token_id in tokenizer(rendered, add_special_tokens=False)[
                    "input_ids"
                ]
            )
            checked[request.prompt] = token_ids
        if token_ids != request.prompt_token_ids:
            raise ValueError(
                f"Manifest tokenization mismatch for `{request.request_id}`: "
                f"manifest={len(request.prompt_token_ids)} actual={len(token_ids)}."
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--label", default="unlabeled")
    parser.add_argument("--server-revision", default=None)
    parser.add_argument("--server-max-batch-size", type=int, required=True)
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--wave2-start-after", type=int, default=None)
    parser.add_argument("--barrier-timeout", type=float, default=1200.0)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    validate_tokenization(manifest, args.model_path)
    model = args.model or manifest.model.rsplit("/", 1)[-1]
    barrier = (
        len(manifest.wave1)
        if args.wave2_start_after is None
        else args.wave2_start_after
    )
    payload = asyncio.run(
        run_benchmark(
            manifest,
            base_url=args.base_url,
            model=model,
            warmup_requests=args.warmup_requests,
            wave2_start_after=barrier,
            barrier_timeout=args.barrier_timeout,
            request_timeout=args.request_timeout,
        )
    )
    payload["environment"] = environment()
    payload["label"] = args.label
    payload["configuration"].update(
        {
            "model_path": args.model_path,
            "server_revision": args.server_revision,
            "server_max_batch_size": args.server_max_batch_size,
        }
    )
    payload["summary"]["manifest_sha256"] = manifest.sha256
    print(json.dumps(payload["pressure_prediction"], indent=2))
    for scope in ("overall", "wave1", "wave2"):
        summary = payload["summary"][scope]
        print(
            f"{scope}: ok={summary['successful_requests']}/{summary['attempted_requests']} "
            f"TTFT-p50={summary['ttft_ms']['p50']} ms "
            f"TTFT-p95={summary['ttft_ms']['p95']} ms "
            f"output={summary['output_tokens_per_second']:.2f} tok/s"
        )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
