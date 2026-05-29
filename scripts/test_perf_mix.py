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





'''# Chunked prefill / TTFT-biased workload.
#
# 设计目标：
# 1. 每个并发窗口都混入 1-2 个多 chunk 长 prompt 和多个极短 prompt；
# 2. 长 prompt 负责制造 prefill 压力，短 prompt 提供更敏感的 TTFT 样本；
# 3. 所有问题都要求短回答，尽量避免 decode 阶段掩盖 TTFT 差异。
PROMPTS = (
    # 0: 长日志诊断，主要制造多 chunk prefill。
    "你是推理服务性能分析员。请只输出 3 条结论，每条不超过 18 字。\n\n"
    + "\n".join(
        [
            (
                f"[Trace {i:03d}] model=FM9G-7B chunk_size=256 "
                f"prompt_len={4096 + (i % 7) * 768} queued_prefill={18 + i % 13} "
                f"decode_batch={2 + i % 5} ttft_ms={780 + (i % 17) * 91} "
                "现象：长 prompt prefill 与短请求 decode 竞争同一个调度窗口。"
            )
            for i in range(120)
        ]
    )
    + "\n\n问题：哪些迹象说明短请求被长 prefill 阻塞？",

    "只回答数字：17 * 23 = ?",

    "只回答一个英文短语：TTFT 的全称是什么？",

    # 3: 长代码审查，prompt 很长但输出极短。
    "阅读下面的调度器伪代码。只输出 2 个最可能影响 TTFT 的问题。\n\n```python\n"
    + "\n".join(
        [
            (
                f"def schedule_step_{i}(waiting, running, budget):\n"
                f"    long_prefill = [r for r in waiting if r.prompt_len > {2048 + i * 8}]\n"
                "    short_prefill = [r for r in waiting if r.prompt_len <= 128]\n"
                "    batch = long_prefill + short_prefill + running\n"
                "    return batch[:budget]\n"
            )
            for i in range(80)
        ]
    )
    + "```\n问题：这个策略为什么不利于短请求首 token？",

    "只回答星期几：2026-05-28 后三天是周几？",

    "只回答 yes 或 no：chunked prefill 会把长 prompt 拆成多个 prefill 片段吗？",

    # 6: RAG 长上下文，模拟检索拼接。
    "基于检索结果回答最后的问题。只输出一句话结论。\n\n"
    + "\n\n".join(
        [
            (
                f"[Doc {i:02d}] 在线推理系统中，超长 prompt 的 prefill 会占用连续计算预算。"
                "当调度器支持 chunked prefill 时，长请求的 KV cache 写入被拆成固定大小块，"
                "短请求可以穿插进入同一个批次，从而降低短请求排队到首 token 的时间。"
            )
            for i in range(48)
        ]
    )
    + "\n\n问题：为什么这个负载更容易体现 chunked prefill 对 TTFT 的收益？",

    "把 'batch scheduler' 翻译成中文，只给译文。",

    "只回答一个数字：2 的 10 次方是多少？",

    "只回答一句话：KV cache 的作用是什么？",

    # 10: 长用户会话，模拟客服/RAG 历史。
    "以下是用户会话历史。请只给出一个优先级最高的处理建议。\n\n"
    + "\n".join(
        [
            (
                f"用户{i:03d}：我的请求 prompt_len={512 + (i % 9) * 1024}，"
                f"排队后 TTFT 超过 {1.2 + (i % 6) * 0.7:.1f}s，"
                "短问答和长文档摘要同时进入服务端，怀疑 prefill 阶段产生队头阻塞。"
            )
            for i in range(110)
        ]
    )
    + "\n最后问题：应该优先检查哪个调度指标？",

    "只回答一个词：prefill 后逐 token 生成的阶段叫什么？",

    "只输出 JSON：{\"ttft_sensitive\": true}",

    # 13: 长表格分析，制造稳定长 prompt。
    "阅读下面的压测表格，只输出 3 个异常点编号。\n\n"
    + "\n".join(
        [
            (
                f"case={i:03d}, qps={4 + i % 8}, concurrency=5, "
                f"prompt_tokens={256 * (4 + i % 24)}, output_tokens={8 + i % 5}, "
                f"ttft_p50={380 + (i % 11) * 70}ms, ttft_p99={1200 + (i % 19) * 160}ms, "
                "note=短请求应当快速进入 decode，但长 prefill 持续占用预算。"
            )
            for i in range(100)
        ]
    )
    + "\n问题：哪些 case 最像 chunked prefill 关闭时的队头阻塞？",

    "只回答数字：4096 / 256 = ?",

    "用 8 个字以内解释：为什么短请求关注 TTFT？",

    # 16: 长合同/规则文本，非代码类长输入。
    "阅读以下服务等级条款，只输出对延迟指标最不利的 2 条。\n\n"
    + "\n".join(
        [
            (
                f"第{i:03d}条：当单个请求 prompt 长度超过 {1024 + (i % 12) * 512} tokens 时，"
                "系统可以暂缓短请求首 token 返回，直到当前 prefill 批次完成；"
                "若启用 chunked prefill，应允许短请求在下一调度片段中插入。"
            )
            for i in range(90)
        ]
    )
    + "\n问题：哪些条款会直接拉高短请求 TTFT？",

    "只回答一个单词：latency 的中文常用译法是什么？",

    "只回答 A/B：更适合测 TTFT 的输出长度是 A.很短 B.很长",

    "只回答一句话：chunk size 过大会怎样影响短请求？",

    # 20: 长混合材料，压住队列尾部。
    "下面混合了日志、设计说明和用户反馈。请只输出一句总体判断。\n\n"
    + "\n".join(
        [
            (
                f"[mix-{i:03d}] 设计：prefill_budget={256 * (1 + i % 6)}, "
                f"decode_budget={8 + i % 4}; 日志：queue_depth={32 + i % 64}, "
                f"long_prompt={4096 + (i % 10) * 1024}; "
                "反馈：短问题的首 token 等待时间比总生成时间更影响交互体验。"
            )
            for i in range(130)
        ]
    )
    + "\n问题：当前负载是否适合观察 chunked prefill 对 TTFT 的改善？",

    "只回答数字：5 个并发里 1 个长请求，短请求有几个？",

    "只回答英文缩写：time to first token 简写是什么？",

    "把这句话压缩到 12 个字以内：长 prompt 不应该长期阻塞短请求。",
)
'''
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
