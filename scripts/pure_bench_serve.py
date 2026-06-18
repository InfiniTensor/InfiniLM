#!/usr/bin/env python3
"""纯 Python 版 BurstGPT 在线压测客户端，可无缝替换 `vllm bench serve`。

设计目标：
  * 复刻 vLLM benchmark serve 的行为(BurstGPT 采样、gamma/泊松到达、并发上限、
    流式 TTFT/ITL/TPOT 统计);
  * 写出与 vLLM 完全相同字段的结果 JSON(供测试脚本 load_result 读取)。

只依赖标准库 + numpy + pandas + transformers，不需要 aiohttp / vllm。
"""
from __future__ import annotations

import argparse
import asyncio
import http.client
import json
import os
import sys
import time
import traceback
import urllib.parse
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MILLISECONDS = 1000.0


# --------------------------------------------------------------------------- #
# 命令行参数：与 vLLM benchmark serve 的子集保持同名，未用到的也照单接收以便兼容。
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str]) -> argparse.Namespace:
    # 兼容 `vllm bench serve ...` 这种前缀调用
    if argv[:2] == ["bench", "serve"]:
        argv = argv[2:]

    p = argparse.ArgumentParser(prog="pure bench serve")
    p.add_argument("--backend", default="openai-chat")
    p.add_argument("--base-url", required=True)
    p.add_argument("--endpoint", default="/v1/chat/completions")
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--dataset-name", default="burstgpt")
    p.add_argument("--dataset-path", required=True)
    p.add_argument("--num-prompts", type=int, default=100)
    p.add_argument("--request-rate", default="inf")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--burstiness", type=float, default=1.0)
    p.add_argument("--max-concurrency", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--ignore-eos", action="store_true")
    p.add_argument("--request-timeout", type=float, default=600.0)
    # 仅为兼容，不影响逻辑：
    p.add_argument("--disable-tqdm", action="store_true")
    p.add_argument("--save-result", action="store_true")
    p.add_argument("--save-detailed", action="store_true")
    p.add_argument("--result-dir", default=None)
    p.add_argument("--result-filename", default=None)
    p.add_argument("--metric-percentiles", default="99")
    # 兜底：吞掉任何其它未知参数，避免因 vLLM 新增 flag 而崩
    args, unknown = p.parse_known_args(argv)
    if unknown:
        print(f"[pure-bench] 忽略未识别参数: {unknown}")
    return args


def parse_rate(value: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float("inf") if value.lower() in {"inf", "infinity"} else float(value)


# --------------------------------------------------------------------------- #
# BurstGPT 数据集采样：严格对齐 vLLM BurstGPTDataset
# --------------------------------------------------------------------------- #
@dataclass
class SampleRequest:
    prompt: str
    prompt_len: int
    expected_output_len: int


def sample_burstgpt(dataset_path: str, num_requests: int, seed: int, tokenizer) -> list[SampleRequest]:
    df = pd.read_csv(dataset_path)
    gpt4 = df[df["Model"] == "GPT-4"]
    gpt4 = gpt4[gpt4["Response tokens"] > 0]
    if len(gpt4) == 0:
        raise RuntimeError(f"BurstGPT 过滤后为空: {dataset_path}")
    replace = num_requests > len(gpt4)
    data = gpt4.sample(n=num_requests, random_state=seed, replace=replace).values.tolist()

    vocab_size = tokenizer.vocab_size
    samples: list[SampleRequest] = []
    for i in range(num_requests):
        input_len = int(data[i][2])   # Request tokens
        output_len = int(data[i][3])  # Response tokens
        token_ids = [(i + j) % vocab_size for j in range(input_len)]
        prompt = tokenizer.decode(token_ids)
        samples.append(SampleRequest(prompt, input_len, output_len))
    return samples


# --------------------------------------------------------------------------- #
# 到达时刻：复刻 vLLM get_request 的 gamma 采样 + 归一化
# --------------------------------------------------------------------------- #
def build_delays(n: int, request_rate: float, burstiness: float) -> list[float]:
    delays: list[float] = []
    for _ in range(n):
        if request_rate == float("inf"):
            delays.append(0.0)
        elif burstiness == float("inf"):
            delays.append(1.0 / request_rate)
        else:
            theta = 1.0 / (request_rate * burstiness)
            delays.append(float(np.random.gamma(shape=burstiness, scale=theta)))
    for i in range(1, n):
        delays[i] += delays[i - 1]
    if request_rate != float("inf") and delays and delays[-1] != 0:
        target_total = n / request_rate
        factor = target_total / delays[-1]
        delays = [d * factor for d in delays]
    return delays


# --------------------------------------------------------------------------- #
# 单请求：流式 chat completions，测 TTFT / ITL / latency / output_tokens
# --------------------------------------------------------------------------- #
@dataclass
class RequestOutput:
    prompt_len: int = 0
    success: bool = False
    generated_text: str = ""
    output_tokens: int | None = None
    ttft: float = 0.0
    latency: float = 0.0
    itl: list[float] = field(default_factory=list)
    start_time: float = 0.0
    error: str = ""


def _do_request_blocking(
    base_url: str,
    endpoint: str,
    model: str,
    req: SampleRequest,
    temperature: float,
    ignore_eos: bool,
    timeout: float,
) -> RequestOutput:
    out = RequestOutput(prompt_len=req.prompt_len)
    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = (parsed.path.rstrip("/") + endpoint) if parsed.path not in ("", "/") else endpoint

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": req.prompt}],
        "temperature": temperature,
        "max_completion_tokens": req.expected_output_len,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(host, port, timeout=timeout)
    st = time.perf_counter()
    out.start_time = st
    most_recent = st
    generated = ""
    try:
        conn.request("POST", path, body=body, headers=headers)
        resp = conn.getresponse()
        if resp.status != 200:
            out.error = f"HTTP {resp.status} {resp.reason}"
            return out
        for raw in resp:  # 按 chunk/行迭代，数据到达即返回，可测 TTFT
            line = raw.strip()
            if not line or line.startswith(b":"):
                continue
            if not line.startswith(b"data:"):
                continue
            chunk = line[len(b"data:"):].strip()
            if chunk == b"[DONE]":
                continue
            now = time.perf_counter()
            data = json.loads(chunk)
            choices = data.get("choices")
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if out.ttft == 0.0:
                    out.ttft = now - st
                else:
                    out.itl.append(now - most_recent)
                generated += content or ""
                most_recent = now
            usage = data.get("usage")
            if usage and usage.get("completion_tokens") is not None:
                out.output_tokens = usage.get("completion_tokens")
        out.generated_text = generated
        out.latency = most_recent - st
        out.success = True
    except Exception:
        out.success = False
        out.error = "".join(traceback.format_exception(*sys.exc_info()))
    finally:
        conn.close()
    return out


# --------------------------------------------------------------------------- #
# 主压测循环：按到达计划派发，asyncio.Semaphore 控制并发上限
# --------------------------------------------------------------------------- #
async def run_benchmark(args, requests: list[SampleRequest], request_rate: float) -> tuple[list[RequestOutput], float]:
    n = len(requests)
    np.random.seed(args.seed)  # 与 vLLM 一致：到达采样前播种全局 numpy RNG
    delays = build_delays(n, request_rate, args.burstiness)

    sem = asyncio.Semaphore(args.max_concurrency) if args.max_concurrency else None
    temperature = 0.0 if args.temperature is None else args.temperature
    outputs: list[RequestOutput | None] = [None] * n
    done = 0

    async def one(i: int) -> None:
        nonlocal done
        async def call() -> RequestOutput:
            return await asyncio.to_thread(
                _do_request_blocking,
                args.base_url, args.endpoint, args.model, requests[i],
                temperature, args.ignore_eos, args.request_timeout,
            )
        if sem is not None:
            async with sem:
                outputs[i] = await call()
        else:
            outputs[i] = await call()
        done += 1
        # 机器可解析的进度行：父进程(测试脚本)心跳据此显示"已完成多少条"
        print(f"[pure-bench] PROGRESS {done}/{n}", flush=True)

    start = time.perf_counter()
    tasks: list[asyncio.Task] = []
    for i in range(n):
        wait = start + delays[i] - time.perf_counter()
        if wait > 0:
            await asyncio.sleep(wait)
        tasks.append(asyncio.create_task(one(i)))
    await asyncio.gather(*tasks)
    duration = time.perf_counter() - start
    return [o for o in outputs], duration  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# 指标：完全对齐 vLLM calculate_metrics 的口径
# --------------------------------------------------------------------------- #
def compute_result(args, requests, outputs: list[RequestOutput], duration: float, tokenizer, percentiles: list[float]) -> dict:
    actual_output_lens: list[int] = []
    total_input = 0
    completed = 0
    ttfts: list[float] = []
    tpots: list[float] = []
    itls: list[float] = []
    e2els: list[float] = []
    failed = 0

    for i, o in enumerate(outputs):
        if o.success:
            out_len = o.output_tokens
            if not out_len:
                out_len = len(tokenizer(o.generated_text, add_special_tokens=False).input_ids)
            actual_output_lens.append(out_len)
            total_input += requests[i].prompt_len
            if out_len > 1:
                tpots.append((o.latency - o.ttft) / (out_len - 1))
            itls += o.itl
            ttfts.append(o.ttft)
            e2els.append(o.latency)
            completed += 1
        else:
            actual_output_lens.append(0)
            failed += 1

    total_output = sum(actual_output_lens)
    dur = duration

    def stats(key: str, vals: list[float], result: dict) -> None:
        arr = vals or [0.0]
        result[f"mean_{key}_ms"] = float(np.mean(arr) * MILLISECONDS)
        result[f"median_{key}_ms"] = float(np.median(arr) * MILLISECONDS)
        result[f"std_{key}_ms"] = float(np.std(arr) * MILLISECONDS)
        for p in percentiles:
            pw = str(int(p)) if int(p) == p else str(p)
            result[f"p{pw}_{key}_ms"] = float(np.percentile(arr, p) * MILLISECONDS)

    result: dict = {
        "duration": dur,
        "completed": completed,
        "failed": failed,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "request_throughput": completed / dur if dur > 0 else 0.0,
        "output_throughput": total_output / dur if dur > 0 else 0.0,
        "total_token_throughput": (total_input + total_output) / dur if dur > 0 else 0.0,
        "input_lens": [o.prompt_len for o in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [o.ttft for o in outputs],
        "itls": [o.itl for o in outputs],
        "generated_texts": [o.generated_text for o in outputs],
        "errors": [o.error for o in outputs],
    }
    stats("ttft", ttfts, result)
    stats("tpot", tpots, result)
    stats("itl", itls, result)
    stats("e2el", e2els, result)
    return result


# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args(sys.argv[1:])
    request_rate = parse_rate(args.request_rate)
    percentiles = [float(x) for x in str(args.metric_percentiles).split(",") if x.strip()]

    from transformers import AutoTokenizer
    tok_src = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)

    print(f"[pure-bench] 数据集={args.dataset_path} 请求数={args.num_prompts} "
          f"速率={args.request_rate} burstiness={args.burstiness} "
          f"并发上限={args.max_concurrency} seed={args.seed}")
    requests = sample_burstgpt(args.dataset_path, args.num_prompts, args.seed, tokenizer)

    outputs, duration = asyncio.run(run_benchmark(args, requests, request_rate))
    result = compute_result(args, requests, outputs, duration, tokenizer, percentiles)

    print(f"[pure-bench] 完成 {result['completed']}/{args.num_prompts}，失败 {result['failed']}，"
          f"用时 {duration:.2f}s，请求吞吐 {result['request_throughput']:.3f} req/s，"
          f"输出吞吐 {result['output_throughput']:.3f} tok/s，"
          f"平均 TTFT {result['mean_ttft_ms']:.2f}ms，平均 TPOT {result['mean_tpot_ms']:.2f}ms")

    if args.save_result and args.result_dir and args.result_filename:
        os.makedirs(args.result_dir, exist_ok=True)
        out_path = os.path.join(args.result_dir, args.result_filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
        print(f"[pure-bench] 结果已写入 {out_path}")
    else:
        # 即便没要求保存，也按 result-dir/result-filename 写出，保证测试脚本能 load_result
        if args.result_dir and args.result_filename:
            os.makedirs(args.result_dir, exist_ok=True)
            out_path = os.path.join(args.result_dir, args.result_filename)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False)
            print(f"[pure-bench] 结果已写入 {out_path}")


if __name__ == "__main__":
    main()
