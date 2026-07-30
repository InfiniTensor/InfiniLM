import argparse
import asyncio
import math
import random
import subprocess
import time
from pathlib import Path

from openai import AsyncOpenAI

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

IMAGE_PROMPTS = [
    "请描述一下图片里的内容。",
    "图片里有人吗？",
    "请结合图片，讲一个小故事。",
]

NUM_REQUESTS = 64
CONCURRENCY = 20
API_URL = "http://127.0.0.1:8000"
MODEL = ""
FIXED_PROMPT = None
MAX_TOKENS = None
REQUESTED_INPUT_TOKENS = None
ACTUAL_INPUT_TOKENS = None


class ImageCollector:
    def __init__(self, dir_path: str, port=None):
        self.dir_path = Path(dir_path).resolve()

        if not self.dir_path.is_dir():
            raise ValueError(f"Not a valid directory: {self.dir_path}")

        self.image_files = [
            file.resolve()
            for file in self.dir_path.rglob("*")
            if file.is_file() and file.suffix.lower() in [".jpg", ".jpeg"]
        ]

        assert len(self.image_files) > 0, "No image file found in provided directory!"

        self.host = "127.0.0.1"
        self.port = port
        self.server_process = None

        # Only start HTTP server if BOTH host and port are provided
        self.use_http = self.host is not None and self.port is not None

        if self.use_http:
            self._start_server()

    def _start_server(self):
        print(
            f"[ImageCollector] Starting image HTTP server...\n"
            f"  Directory: {self.dir_path}\n"
            f"  URL: http://{self.host}:{self.port}\n"
        )
        self.server_process = subprocess.Popen(
            [
                "python",
                "-m",
                "http.server",
                str(self.port),
                "--bind",
                self.host,
            ],
            cwd=str(self.dir_path),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        time.sleep(0.5)

    def stop_server(self):
        if self.server_process is not None:
            self.server_process.terminate()

            try:
                self.server_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.server_process.kill()

            self.server_process = None

    def __del__(self):
        self.stop_server()

    def random_image_url(self):
        image_path = random.choice(self.image_files)

        # Return local absolute path
        if not self.use_http:
            return str(image_path)

        # Return HTTP URL
        relative_path = image_path.relative_to(self.dir_path)

        return f"http://{self.host}:{self.port}/{relative_path.as_posix()}"


async def benchmark_user(
    client, semaphore, queue, results, user_id, verbose, image_collector=None
):
    while True:
        async with semaphore:
            task_id = await queue.get()
            if task_id is None:
                queue.task_done()
                break

            try:
                print(f"🚀 User#{user_id} Sending request #{task_id}")
                messages = None
                if image_collector is None:
                    messages = [
                        {
                            "role": "user",
                            "content": FIXED_PROMPT
                            if FIXED_PROMPT is not None
                            else random.choice(PROMPTS),
                        }
                    ]
                else:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_collector.random_image_url()
                                    },
                                },
                                {"type": "text", "text": random.choice(IMAGE_PROMPTS)},
                            ],
                        }
                    ]

                print(messages)

                start_time = time.time()
                request_args = {
                    "model": MODEL,
                    "messages": messages,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                if MAX_TOKENS is not None:
                    request_args["max_tokens"] = MAX_TOKENS
                stream = await client.chat.completions.create(**request_args)

                first_token_time = None
                total_tokens = 0
                usage_tokens = None
                answer_chunks = []

                async for chunk in stream:
                    if chunk.usage is not None:
                        usage_tokens = chunk.usage.completion_tokens
                    if not chunk.choices:
                        continue
                    if first_token_time is None:
                        first_token_time = time.time()
                    delta = chunk.choices[0].delta.content
                    if delta:
                        answer_chunks.append(delta)
                        total_tokens += 1
                    if chunk.choices[0].finish_reason is not None:
                        break

                end_time = time.time()
                if usage_tokens is not None:
                    total_tokens = usage_tokens

                ttft = first_token_time - start_time if first_token_time else None
                elapsed_time = end_time - start_time if start_time else None
                tpot_ms = (
                    ((elapsed_time - ttft) / (total_tokens - 1) * 1000)
                    if total_tokens > 0
                    and elapsed_time
                    and ttft is not None
                    and total_tokens > 1
                    else None
                )
                tokens_per_second = (
                    total_tokens / elapsed_time if elapsed_time > 0 else 0
                )

                answer = "".join(answer_chunks)

                results.append(
                    (total_tokens, elapsed_time, tokens_per_second, ttft, tpot_ms)
                )

                if verbose:
                    print(f"\n📝 Request #{task_id} (User #{user_id})")
                    if ttft is not None:
                        print(f"  ⏱ 首字延迟 TTFT: {ttft:.3f}s")
                    if elapsed_time is not None:
                        print(f"  ⏱ 总耗时: {elapsed_time:.3f}s")

                    print(f"  🔤 解码 token 总数: {total_tokens}")
                    if tpot_ms is not None:
                        print(f"  📏 TPOT（排除 TTFT）: {tpot_ms:.2f} ms/token")
                    else:
                        print("  📏 TPOT（排除 TTFT）: N/A")
                    print(f"  ❓ 提问: {messages}")
                    print(f"  💬 回答: {answer}\n")

                queue.task_done()
            except Exception as e:
                if verbose:
                    print(f"\n⚠️ Request #{task_id} (User #{user_id}) FAILED:")
                    print(f"  ❌ Error: {e}\n")
                queue.task_done()


async def run_benchmark(verbose=False, image_collector=None):
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
            benchmark_user(
                client, semaphore, queue, results, user_id, verbose, image_collector
            )
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
    if REQUESTED_INPUT_TOKENS is not None:
        print(f"{'目标输入 tokens':<{width_label}}: {REQUESTED_INPUT_TOKENS}")
        print(f"{'实际输入 tokens':<{width_label}}: {ACTUAL_INPUT_TOKENS}")
    print(f"{'最大输出 tokens':<{width_label}}: {MAX_TOKENS}")
    print(f"{'成功请求数':<{width_label}}: {successful_requests}")
    print(f"{'总耗时':<{width_label}}: {total_elapsed_time:.2f} s")
    print(f"{'总输出token数':<{width_label}}: {sum(tokens_list)}")
    print(
        f"{'聚合输出吞吐':<{width_label}}: "
        f"{sum(tokens_list) / total_elapsed_time:.2f} tokens/s"
    )
    print(f"{'请求速率 (RPS)':<{width_label}}: {requests_per_second:.2f} requests/s")
    print(sep)
    print(f"{'Average latency':<{width_label}}: {avg_latency:.2f} s")
    print(f"{'Average TTFT':<{width_label}}: {avg_ttft:.2f} s")
    if avg_ms_per_token is None:
        print(f"{'Average TPOT':<{width_label}}: N/A")
    else:
        print(f"{'Average TPOT':<{width_label}}: {avg_ms_per_token:.2f} ms/token")
    print(
        f"{'Avg Token generation speed':<{width_label}}: {avg_tokens_per_second:.2f} tokens/s"
    )


def build_prompt_for_token_length(tokenizer_path, target_tokens):
    """Build a natural-language prompt close to an exact chat-template token length."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    prefix = "请阅读以下背景信息，并用正常语言简要总结其要点："
    filler = "这是用于语言模型服务性能测试的一段背景信息，内容本身没有特殊含义。"

    def chat_length(content):
        messages = [{"role": "user", "content": content}]
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        input_ids = encoded["input_ids"]
        if hasattr(input_ids, "shape"):
            return input_ids.shape[-1]
        if input_ids and isinstance(input_ids[0], list):
            return len(input_ids[0])
        return len(input_ids)

    minimum = chat_length(prefix)
    if target_tokens < minimum:
        raise ValueError(
            f"--input-tokens={target_tokens} is smaller than chat-template "
            f"overhead ({minimum} tokens)"
        )

    low, high = 0, max(1, math.ceil(target_tokens / 2))
    best_prompt, best_length = prefix, minimum
    while low <= high:
        middle = (low + high) // 2
        candidate = prefix + filler * middle
        length = chat_length(candidate)
        if length <= target_tokens:
            best_prompt, best_length = candidate, length
            low = middle + 1
        else:
            high = middle - 1

    # Chinese characters are generally one token in Qwen tokenizers. Try a
    # short suffix search to close any remainder left by sentence granularity.
    for count in range(1, target_tokens - best_length + 9):
        candidate = best_prompt + "测" * count
        length = chat_length(candidate)
        if best_length < length <= target_tokens:
            best_prompt, best_length = candidate, length
        if length > target_tokens:
            break

    # Token merging can make a repeated-character suffix jump over the target.
    # Greedily try heterogeneous suffixes that commonly add one token.
    suffixes = ("。", "！", "？", "\n", " a", " 1", "x")
    while best_length < target_tokens:
        previous_length = best_length
        for suffix in suffixes:
            candidate = best_prompt + suffix
            length = chat_length(candidate)
            if best_length < length <= target_tokens:
                best_prompt, best_length = candidate, length
                if best_length == target_tokens:
                    break
        if best_length == previous_length:
            break

    return best_prompt, best_length


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--mm-port", type=str, default=None)
    parser.add_argument("--api-url", type=str, default="127.0.0.1:8000")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--num-requests", type=int, default=NUM_REQUESTS)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt-repeat", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--input-tokens", type=int, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    args = parser.parse_args()

    if args.num_requests <= 0:
        parser.error("--num-requests must be positive")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.prompt_repeat <= 0:
        parser.error("--prompt-repeat must be positive")
    if args.input_tokens is not None and args.input_tokens <= 0:
        parser.error("--input-tokens must be positive")
    if args.input_tokens is not None and args.prompt is not None:
        parser.error("--input-tokens and --prompt are mutually exclusive")
    if args.input_tokens is not None and not args.tokenizer:
        parser.error("--tokenizer is required with --input-tokens")

    API_URL = "http://" + args.api_url
    MODEL = args.model
    NUM_REQUESTS = args.num_requests
    CONCURRENCY = args.concurrency
    MAX_TOKENS = args.max_tokens
    REQUESTED_INPUT_TOKENS = args.input_tokens
    if args.input_tokens is not None:
        try:
            FIXED_PROMPT, ACTUAL_INPUT_TOKENS = build_prompt_for_token_length(
                args.tokenizer, args.input_tokens
            )
        except ValueError as error:
            parser.error(str(error))
    else:
        FIXED_PROMPT = (
            args.prompt * args.prompt_repeat if args.prompt is not None else None
        )

    image_collector = None
    if args.image_dir is not None:
        image_collector = ImageCollector(args.image_dir, port=args.mm_port)

    asyncio.run(run_benchmark(args.verbose, image_collector))
