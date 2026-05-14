import argparse
import asyncio
import json
import random
import time
from typing import Optional

from openai import AsyncOpenAI

# How to read timing fields (also in argparse epilog for `test_perf.py --help`):
# - Per request, the client counts one "token" per non-empty SSE `delta.content` chunk (not HF BPE count).
# - avg_ms_per_token == avg_ms_per_stream_chunk: mean over requests of (full stream wall / chunk count);
#   wall starts before `chat.completions.create` and includes prefill + queueing; at concurrency>1, gaps
#   between chunks often reflect other requests in the same server batch.
# - avg_decode_ms_per_chunk: mean over requests of (elapsed - TTFT) / chunks — closer to decode-shaped
#   comparison vs in-proc `avg_decode_itl_ms` from bench_balanced / jiuge timing.

PROMPTS = [
    "如果猫能写诗，它们会写些什么？",
    "描述一个没有重力的世界。",
    "如果地球停止自转，会发生什么？",
    "假设你是一只会飞的鲸鱼，描述你的日常生活。",
    "如果人类可以与植物沟通，世界会变成什么样？",
    "描述一个由糖果构成的城市。",
    "如果时间旅行成为可能，你最想去哪个时代？",
    "想象一下，如果地球上只有蓝色，其他颜色都消失了。",
    "如果动物能上网，它们会浏览什么网站？",
    "描述一个没有声音的世界。",
    "如果人类可以在水下呼吸，城市会如何变化？",
    "想象一下，如果天空是绿色的，云是紫色的。",
    "如果你能与任何历史人物共进晚餐，你会选择谁？",
    "描述一个没有夜晚的星球。",
    "如果地球上只有一种语言，世界会如何运作？",
    "想象一下，如果所有的书都变成了音乐。",
    "如果你可以变成任何一种动物，你会选择什么？",
    "描述一个由机器人统治的未来世界。",
    "如果你能与任何虚构角色成为朋友，你会选择谁？",
    "想象一下，如果每个人都能读懂他人的思想。",
]

DEFAULT_NUM_REQUESTS = 64
DEFAULT_CONCURRENCY = 20
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "FM9G-7B"


async def _drain_warmup_stream(
    client: AsyncOpenAI,
    *,
    model: str,
    user_content: str,
    max_tokens: int,
) -> None:
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_content}],
        stream=True,
        max_tokens=max_tokens,
    )
    async for chunk in stream:
        if chunk.choices[0].finish_reason is not None:
            break


async def benchmark_user(
    client,
    semaphore,
    queue,
    results,
    user_id,
    verbose,
    model,
    max_tokens,
    fixed_prompt: Optional[str],
):
    while True:
        async with semaphore:
            task_id = await queue.get()
            if task_id is None:
                queue.task_done()
                break

            question = fixed_prompt if fixed_prompt is not None else random.choice(PROMPTS)
            try:
                print(f"🚀 User#{user_id} Sending request #{task_id}")

                start_time = time.time()
                stream = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": question}],
                    stream=True,
                    max_tokens=max_tokens,
                )

                first_token_time = None
                total_tokens = 0
                answer_chunks = []

                async for chunk in stream:
                    if first_token_time is None:
                        first_token_time = time.time()
                    delta = chunk.choices[0].delta.content
                    if delta:
                        answer_chunks.append(delta)
                        total_tokens += 1
                    if chunk.choices[0].finish_reason is not None:
                        break

                end_time = time.time()

                ttft = first_token_time - start_time if first_token_time else None
                elapsed_time = end_time - start_time if start_time else None
                ms_per_token = (
                    (elapsed_time / total_tokens * 1000)
                    if total_tokens > 0 and elapsed_time
                    else None
                )
                tokens_per_second = (
                    total_tokens / elapsed_time if elapsed_time > 0 else 0
                )

                answer = "".join(answer_chunks)

                results.append(
                    (total_tokens, elapsed_time, tokens_per_second, ttft, ms_per_token)
                )

                if verbose:
                    print(f"\n📝 Request #{task_id} (User #{user_id})")
                    if ttft is not None:
                        print(f"  ⏱ 首字延迟 TTFT: {ttft:.3f}s")
                    if elapsed_time is not None:
                        print(f"  ⏱ 总耗时: {elapsed_time:.3f}s")

                    print(f"  🔤 解码 token 总数: {total_tokens}")
                    if ms_per_token is not None:
                        print(f"  📏 平均 wall / SSE 文本块: {ms_per_token:.2f} ms/chunk")
                    else:
                        print(f"  📏 平均 wall / SSE 文本块: N/A (no chunk generated)")
                    print(f"  ❓ 提问: {question}")
                    print(f"  💬 回答: {answer}\n")

                queue.task_done()
            except Exception as e:
                if verbose:
                    print(f"\n⚠️ Request #{task_id} (User #{user_id}) FAILED:")
                    print(f"  ❌ Error: {e}\n")
                queue.task_done()


async def run_benchmark(
    *,
    base_url,
    model,
    num_requests,
    concurrency,
    max_tokens,
    fixed_prompt: Optional[str] = None,
    warmup_requests: int = 0,
    warmup_max_tokens: Optional[int] = None,
    verbose=False,
):
    client = AsyncOpenAI(base_url=base_url, api_key="default")

    warmup_wall_s = 0.0
    if warmup_requests > 0:
        wm = warmup_max_tokens if warmup_max_tokens is not None else max_tokens
        t0 = time.perf_counter()
        for i in range(int(warmup_requests)):
            user_content = (
                fixed_prompt if fixed_prompt is not None else random.choice(PROMPTS)
            )
            if verbose:
                print(f"🔥 Warmup {i + 1}/{warmup_requests} (max_tokens={wm}) ...", flush=True)
            await _drain_warmup_stream(
                client, model=model, user_content=user_content, max_tokens=int(wm)
            )
        warmup_wall_s = time.perf_counter() - t0
        if verbose:
            print(f"🔥 Warmup done in {warmup_wall_s:.2f}s\n", flush=True)

    semaphore = asyncio.Semaphore(concurrency)
    queue = asyncio.Queue()
    results = []
    for i in range(num_requests):
        await queue.put(i)
    for _ in range(concurrency):
        await queue.put(None)

    users = [
        asyncio.create_task(
            benchmark_user(
                client,
                semaphore,
                queue,
                results,
                user_id,
                verbose,
                model,
                max_tokens,
                fixed_prompt,
            )
        )
        for user_id in range(concurrency)
    ]

    start_time = time.time()
    await queue.join()
    await asyncio.gather(*users)
    end_time = time.time()

    total_elapsed_time = end_time - start_time
    tokens_list = [r[0] for r in results if r and r[0] is not None]
    latencies = [r[1] for r in results if r and r[1] is not None]
    tokens_per_second_list = [r[2] for r in results if r and r[2] is not None]
    ttft_list = [r[3] for r in results if r and r[3] is not None]
    ms_per_token_list = [r[4] for r in results if r and r[4] is not None]

    successful_requests = len(results)
    requests_per_second = (
        successful_requests / total_elapsed_time if total_elapsed_time > 0 else 0
    )
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    avg_tokens_per_second = (
        sum(tokens_per_second_list) / len(tokens_per_second_list)
        if tokens_per_second_list
        else 0
    )
    avg_ttft = sum(ttft_list) / len(ttft_list) if ttft_list else 0
    avg_ms_per_token = (
        sum(ms_per_token_list) / len(ms_per_token_list) if ms_per_token_list else None
    )

    # Decode-phase proxy: mean over requests of (wall after first chunk) / (# non-empty chunks).
    # Use with --prompt short + --concurrency 1 to compare against in-proc bench decode ITL.
    decode_ms_per_chunk_list: list[float] = []
    for r in results:
        if not r or len(r) < 5:
            continue
        total_tokens, elapsed, _tps, ttft, _mpt = r
        if ttft is None or total_tokens is None or total_tokens <= 0 or elapsed is None:
            continue
        decode_s = max(0.0, float(elapsed) - float(ttft))
        decode_ms_per_chunk_list.append(1000.0 * decode_s / float(total_tokens))

    decode_wall_s_list = [
        float(r[1]) - float(r[3])
        for r in results
        if r and len(r) >= 4 and r[1] is not None and r[3] is not None
    ]
    avg_decode_wall_s = (
        sum(decode_wall_s_list) / len(decode_wall_s_list) if decode_wall_s_list else 0.0
    )
    avg_decode_ms_per_chunk = (
        sum(decode_ms_per_chunk_list) / len(decode_ms_per_chunk_list)
        if decode_ms_per_chunk_list
        else None
    )

    width_label = 24
    sep = "-" * 60

    print(f"\n=== 📊 性能指标汇总 ({model}) ===")
    print(sep)
    if fixed_prompt is not None:
        print(f"{'Fixed prompt':<{width_label}}: {fixed_prompt[:80]}{'…' if len(fixed_prompt) > 80 else ''}")
    else:
        print(f"{'User prompt':<{width_label}}: random from PROMPTS pool")
    if warmup_requests > 0:
        print(f"{'Warmup (discarded)':<{width_label}}: {warmup_requests} req, {warmup_wall_s:.2f} s")
    print(f"{'并发数':<{width_label}}: {concurrency}")
    print(f"{'请求总数':<{width_label}}: {num_requests}")
    print(f"{'成功请求数':<{width_label}}: {successful_requests}")
    print(f"{'总耗时':<{width_label}}: {total_elapsed_time:.2f} s")
    print(f"{'总输出token数':<{width_label}}: {sum(tokens_list)}")
    print(f"{'请求速率 (RPS)':<{width_label}}: {requests_per_second:.2f} requests/s")
    print(sep)
    print(f"{'Average latency':<{width_label}}: {avg_latency:.2f} s")
    print(f"{'Average TTFT':<{width_label}}: {avg_ttft:.2f} s")
    _mpt = (
        f"{avg_ms_per_token:.2f} ms/chunk"
        if avg_ms_per_token is not None
        else "N/A"
    )
    print(
        f"{'Avg wall ms / stream chunk':<{width_label}}: {_mpt}  "
        f"(JSON avg_ms_per_stream_chunk; legacy avg_ms_per_token)"
    )
    print(
        f"{'Avg Token generation speed':<{width_label}}: {avg_tokens_per_second:.2f} tokens/s"
    )
    if avg_decode_ms_per_chunk is not None:
        print(
            f"{'Avg decode wall ms/chunk':<{width_label}}: {avg_decode_ms_per_chunk:.2f} ms/chunk"
        )
        print(f"{'Avg decode wall (post-TTFT)':<{width_label}}: {avg_decode_wall_s:.2f} s")

    return {
        "base_url": base_url,
        "model": model,
        "num_requests": num_requests,
        "concurrency": concurrency,
        "max_tokens": max_tokens,
        "fixed_prompt": fixed_prompt,
        "warmup_requests": int(warmup_requests),
        "warmup_wall_s": float(warmup_wall_s),
        "successful_requests": successful_requests,
        "total_elapsed_time_s": total_elapsed_time,
        "total_output_tokens": sum(tokens_list),
        "requests_per_second": requests_per_second,
        "avg_latency_s": avg_latency,
        "avg_ttft_s": avg_ttft,
        "avg_ms_per_token": avg_ms_per_token,
        "avg_ms_per_stream_chunk": avg_ms_per_token,
        "avg_decode_wall_s": avg_decode_wall_s,
        "avg_decode_ms_per_chunk": avg_decode_ms_per_chunk,
        "avg_tokens_per_second": avg_tokens_per_second,
    }


if __name__ == "__main__":
    _METRICS_EPILOG = """
Streaming timing fields (JSON from --json-out):
  avg_ms_per_stream_chunk: mean over requests of (wall seconds for the full OpenAI streaming
    completion / number of non-empty SSE text deltas). Same numeric value as avg_ms_per_token
    (legacy name). This is NOT HuggingFace tokenizer ITL; the numerator includes time before
    chat.completions.create returns, prefill, and—when --concurrency>1—other requests' GPU work
    between chunks on this stream.
  avg_decode_ms_per_chunk: mean over requests of ((elapsed - TTFT) / chunks); use with
    --concurrency 1 and a short --prompt to compare against in-proc avg_decode_itl_ms from
    bench_balanced.py / jiuge timing.
"""

    parser = argparse.ArgumentParser(
        description="Async OpenAI-compatible chat streaming load probe.",
        epilog=_METRICS_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_BASE_URL,
        help="OpenAI client base_url (include /v1), e.g. http://127.0.0.1:8001/v1",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--num-requests",
        type=int,
        default=DEFAULT_NUM_REQUESTS,
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max completion tokens per request (passed to chat.completions.create).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help=(
            "If set, use this user message for every request (including warmup); "
            "else each request samples randomly from built-in Chinese PROMPTS."
        ),
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=0,
        help="Number of streaming completions to run before the timed benchmark (discarded).",
    )
    parser.add_argument(
        "--warmup-max-tokens",
        type=int,
        default=None,
        help="max_tokens for warmup only (default: same as --max-tokens).",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="If set, write summary metrics JSON (see epilog for avg_ms_per_stream_chunk vs decode).",
    )
    args = parser.parse_args()
    summary = asyncio.run(
        run_benchmark(
            base_url=args.base_url,
            model=args.model,
            num_requests=args.num_requests,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            fixed_prompt=args.prompt,
            warmup_requests=args.warmup_requests,
            warmup_max_tokens=args.warmup_max_tokens,
            verbose=args.verbose,
        )
    )
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=2)
