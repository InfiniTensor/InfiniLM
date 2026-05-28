import asyncio
import time
from openai import AsyncOpenAI
import argparse
import random

PROMPTS = [

    # ~10000 tokens：极限长上下文，多文件代码重构
    "下面给出 4 个相关文件，请重构以消除重复逻辑并提取公共抽象：\n\n"
    + "# file: scheduler_v1.py\n" + "def schedule(reqs):\n    return sorted(reqs, key=lambda r: r.arrival)\n" * 100
    + "\n# file: scheduler_v2.py\n" + "def schedule(reqs):\n    return sorted(reqs, key=lambda r: -r.priority)\n" * 100
    + "\n# file: scheduler_v3.py\n" + "def schedule(reqs):\n    return sorted(reqs, key=lambda r: r.prompt_len)\n" * 100
    + "\n# file: scheduler_v4.py\n" + "def schedule(reqs):\n    return sorted(reqs, key=lambda r: r.slo_deadline)\n" * 100,

    

    "1+1=?",
    
]

NUM_REQUESTS = len(PROMPTS)
CONCURRENCY = 5
API_URL = "http://127.0.0.1:2333"
MODEL = "FM9G-7B"


async def benchmark_user(client, semaphore, queue, results, user_id, verbose):
    while True:
        async with semaphore:
            task_id = await queue.get()
            if task_id is None:
                queue.task_done()
                break

            question = PROMPTS[task_id]
            try:
                print(f"🚀 User#{user_id} Sending request #{task_id}")

                start_time = time.time()
                stream = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": question}],
                    stream=True,
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
                        print(f"  📏 平均 token 解码时间: {ms_per_token:.2f} ms/token")
                    else:
                        print(f"  📏 平均 token 解码时间: N/A (no token generated)")
                    print(f"  ❓ 提问: {question}")
                    print(f"  💬 回答: {answer}\n")

                queue.task_done()
            except Exception as e:
                if verbose:
                    print(f"\n⚠️ Request #{task_id} (User #{user_id}) FAILED:")
                    print(f"  ❌ Error: {e}\n")
                queue.task_done()


async def run_benchmark(verbose=False):
    client = AsyncOpenAI(base_url=API_URL, api_key="default")
    semaphore = asyncio.Semaphore(CONCURRENCY)
    queue = asyncio.Queue()
    results = []
    for i in range(NUM_REQUESTS):
        await queue.put(i)
    for _ in range(CONCURRENCY):
        await queue.put(None)

    users = [
        asyncio.create_task(
            benchmark_user(client, semaphore, queue, results, user_id, verbose)
        )
        for user_id in range(CONCURRENCY)
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

    width_label = 24
    sep = "-" * 60

    print(f"\n=== 📊 性能指标汇总 ({MODEL}) ===")
    print(sep)
    print(f"{'并发数':<{width_label}}: {CONCURRENCY}")
    print(f"{'请求总数':<{width_label}}: {NUM_REQUESTS}")
    print(f"{'成功请求数':<{width_label}}: {successful_requests}")
    print(f"{'总耗时':<{width_label}}: {total_elapsed_time:.2f} s")
    print(f"{'总输出token数':<{width_label}}: {sum(tokens_list)}")
    print(f"{'请求速率 (RPS)':<{width_label}}: {requests_per_second:.2f} requests/s")
    print(sep)
    print(f"{'Average latency':<{width_label}}: {avg_latency:.2f} s")
    print(f"{'Average TTFT':<{width_label}}: {avg_ttft:.2f} s")
    print(f"{'Avg time per token':<{width_label}}: {avg_ms_per_token:.2f} ms/token")
    print(
        f"{'Avg Token generation speed':<{width_label}}: {avg_tokens_per_second:.2f} tokens/s"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.verbose))

