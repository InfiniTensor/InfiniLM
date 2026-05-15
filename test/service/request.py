import argparse
import asyncio
import time

from openai import AsyncOpenAI


def get_args():
    # 1. 创建参数解析器（支持重复 --message 构建列表）
    parser = argparse.ArgumentParser(description="向推理服务发送 OpenAI 格式请求")

    # 核心：重复 --content 自动拼成列表
    parser.add_argument(
        "--system",
        type=str,
        default="",
        help="system prompt",
    )
    parser.add_argument(
        "--content",
        action="append",
        default=[],
        help="start with content type['text', 'image_url'] and colon, e.g. text:hello or image_url:http://example.com/image.jpg",
    )

    # 目标服务地址与端口
    parser.add_argument(
        "--port", type=int, default=8000, help="推理服务端口，默认 8000"
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="推理服务地址，默认 127.0.0.1"
    )

    # 解析参数
    return parser.parse_args()


def build_messages(content_args, system_prompt):
    contents = []
    for content in content_args:
        if ":" not in content:
            raise ValueError(
                f"Invalid content format: '{content}'. Expected format is 'type:value'."
            )
        ctype, cvalue = content.split(":", 1)

        if ctype == "text":
            contents.append({"type": "text", "text": cvalue})
        elif ctype == "image_url":
            contents.append({"type": "image_url", "image_url": {"url": cvalue}})
        else:
            raise ValueError(
                f"Unsupported content type: '{ctype}'. Supported types are 'text' and 'image_url'."
            )

    messages = (
        [] if not system_prompt else [{"role": "system", "content": system_prompt}]
    )
    messages.append({"role": "user", "content": contents})
    return messages


async def benchmark_user(client, messages):
    try:
        print(f"  ❓ 提问: {messages}")
        start_time = time.time()
        stream = await client.chat.completions.create(
            model="default",
            messages=messages,
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
        ms_per_token_decode = (
            ((elapsed_time - ttft) / (total_tokens - 1) * 1000)
            if total_tokens - 1 > 0 and elapsed_time
            else None
        )

        answer = "".join(answer_chunks)
        print(f"  💬 回答: {answer}\n")
        print(f"  总耗时: {elapsed_time:.3f}s")
        print(f"  首字延迟 TTFT: {ttft:.3f}s")
        print(f"  Token间延迟 ITL: {ms_per_token_decode:.2f} ms")
        print(
            f"  Decode吞吐: {1000 / ms_per_token_decode:.2f} tokens/s"
            if ms_per_token_decode
            else "  Decode吞吐: N/A"
        )

    except Exception as e:
        print(f"  ❌ Error: {e}\n")


def main():
    args = get_args()
    if not args.content:
        args.content = ["text:山东最高的山是？"]
    messages = build_messages(args.content, args.system)
    client = AsyncOpenAI(base_url=f"http://{args.host}:{args.port}", api_key="default")
    asyncio.run(benchmark_user(client, messages))


if __name__ == "__main__":
    main()
