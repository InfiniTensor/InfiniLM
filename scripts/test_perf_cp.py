"""
Chunked Prefill TTFT Benchmark

Test: send a long request, wait a short delay, then send a short request.
Measure the short request's TTFT and E2E.

With chunked prefill: short request inserts at next chunk boundary → lower TTFT
Without chunked prefill: short request waits for full long prefill → higher TTFT

Usage:
  python3 scripts/test_perf.py [--rounds 5] [--delay 0.1]
"""
import asyncio
import time
from openai import AsyncOpenAI
import argparse

API_URL = "http://127.0.0.1:2333"
MODEL = "jiuge"
MAX_TOKENS = 30 # decode的tokens数

_BASE_PARAGRAPHS = [
    '''人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。人工智能的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大。可以设想，未来人工智能带来的科技产品，将会是人类智慧的容器。''',
    '''1956年夏季，以麦卡赛、明斯基、罗切斯特和申农等为首的一批有远见卓识的年轻科学家在一起聚会，共同研究和探讨用机器模拟智能的一系列有关问题，并首次提出了人工智能这一术语，它标志着人工智能这门新兴学科的正式诞生。此后，IBM公司研制的专用计算机深蓝击败了国际象棋世界冠军卡斯帕罗夫。谷歌公司开发的AlphaGo程序战胜了围棋世界冠军李世石，这被认为是人工智能发展史上的一个重要里程碑。''',
    '''量子计算是一种利用量子力学原理进行信息处理的计算方式。与经典计算机使用比特作为信息的基本单位不同，量子计算机使用量子比特。量子比特具有叠加态的特性，即一个量子比特可以同时处于0和1的状态，这使得量子计算机在处理某些特定问题时具有经典计算机无法比拟的优势。量子纠缠是量子计算中另一个关键概念，当两个量子比特发生纠缠时，测量其中一个的状态会立即影响另一个的状态。''',
    '''根据联合国政府间气候变化专门委员会第六次评估报告，全球平均温度已经比工业化前水平上升了约1.1摄氏度。报告指出，人类活动是导致全球变暖的主要原因，其中化石燃料的燃烧、工业生产和土地利用变化是温室气体排放的主要来源。极端天气事件的频率和强度都在增加，包括热浪、干旱、暴雨和洪水。北极海冰面积持续缩小，格陵兰和南极冰盖加速融化。''',
    '''深度学习是机器学习的一个分支，其核心是利用多层神经网络从大量数据中自动学习特征表示。卷积神经网络在图像识别领域取得了巨大成功，循环神经网络和Transformer架构则在自然语言处理领域展现了强大能力。近年来，大语言模型如GPT、BERT、LLaMA等引领了自然语言处理的技术革新，这些模型通过在海量文本数据上进行预训练，获得了强大的语言理解和生成能力。''',
    '''在计算机体系结构领域，冯诺依曼架构仍然是现代计算机的基础。然而随着摩尔定律逐渐放缓，研究人员开始探索新型计算范式，包括神经形态计算、存内计算、光子计算等。GPU和TPU等专用加速器的发展极大推动了深度学习的进步。RISC-V开源指令集架构的兴起为芯片设计带来了新的可能性，而chiplet技术和先进封装则为突破制程限制提供了新的路径。''',
    '''可再生能源的成本大幅下降，太阳能和风能已经成为最便宜的新增发电来源。电动汽车市场快速增长，电池技术不断进步。碳捕获和储存技术、绿色氢能等前沿技术也在加速发展。然而，要实现全球碳中和目标，仍需要在能源系统、交通运输、工业生产、建筑等领域进行深刻的变革。智能电网、储能技术、虚拟电厂等概念正在从理论走向实践。''',
    '''生物信息学是一门利用计算机技术和数学方法研究生物学问题的交叉学科。基因组学、蛋白质组学、代谢组学等组学技术的发展产生了海量的生物数据。AlphaFold2在蛋白质结构预测方面取得了革命性突破，为药物研发和生命科学研究开辟了新的方向。CRISPR基因编辑技术的发展使得精准修改基因成为可能，为遗传疾病的治疗带来了希望。''',
]


def build_long_prompt(idx, target_chars=9000):
    parts = [f"(文档编号{idx}) 请仔细阅读以下学术材料并总结：\n\n"]
    i = idx
    while sum(len(p) for p in parts) < target_chars:
        parts.append(_BASE_PARAGRAPHS[i % len(_BASE_PARAGRAPHS)])
        parts.append("\n\n")
        i += 1
    parts.append(f"以上是第{idx}份材料，请给出详细分析。")
    return "".join(parts)


async def measure_one_round(client, round_idx, delay_sec):
    """
    1. Fire a long request (starts prefill immediately)
    2. After delay_sec, fire a short request
    3. Return both TTFT and E2E for both requests
    """
    long_prompt = build_long_prompt(round_idx, target_chars=9000)
    short_prompt = f"(编号{round_idx}) 1+1等于几？"

    long_result = {}
    short_result = {}

    async def do_request(prompt, result_dict, delay=0):
        if delay > 0:
            await asyncio.sleep(delay)
        t0 = time.time()
        stream = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_new_tokens=MAX_TOKENS,
            temperature=1.0,
            top_p=1.0,
            extra_body={"top_k": 1},
        )
        first_token_time = None
        total_tokens = 0
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.time()
                total_tokens += 1
            if chunk.choices[0].finish_reason is not None:
                break
        end_time = time.time()
        result_dict["ttft"] = (first_token_time - t0) if first_token_time else None
        result_dict["e2e"] = end_time - t0
        result_dict["tokens"] = total_tokens

    await asyncio.gather(
        do_request(long_prompt, long_result, delay=0),
        do_request(short_prompt, short_result, delay=delay_sec),
    )
    return long_result, short_result


async def run_benchmark(rounds, delay):
    client = AsyncOpenAI(base_url=API_URL, api_key="default")

    # Warmup
    print("Warmup...")
    await measure_one_round(client, 100, delay)
    print("Warmup done.\n")

    long_ttfts = []
    long_e2es = []
    short_ttfts = []
    short_e2es = []

    for i in range(rounds):
        lr, sr = await measure_one_round(client, i, delay) # lr = long request result, sr = short request result

        lt = lr["ttft"] * 1000 if lr["ttft"] else 0
        le = lr["e2e"] * 1000
        st = sr["ttft"] * 1000 if sr["ttft"] else 0
        se = sr["e2e"] * 1000

        print(f"  Round {i}: LONG  TTFT={lt:>7.1f}ms  E2E={le:>8.1f}ms  tokens={lr['tokens']}")
        print(f"           SHORT TTFT={st:>7.1f}ms  E2E={se:>8.1f}ms  tokens={sr['tokens']}")

        if lr["ttft"]:
            long_ttfts.append(lr["ttft"])
        long_e2es.append(lr["e2e"])
        if sr["ttft"]:
            short_ttfts.append(sr["ttft"])
        short_e2es.append(sr["e2e"])

    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Chunked Prefill TTFT Benchmark")
    print(f"{sep}")
    print(f"  Rounds: {rounds}")
    print(f"  Delay before short request: {delay}s")
    print(f"  Long prompt: ~9000 chars")
    print(f"  Max tokens: {MAX_TOKENS}")

    def print_stats(label, ttfts, e2es):
        if not ttfts:
            return
        print(f"\n  [{label}]")
        print(f"    Avg TTFT: {sum(ttfts)/len(ttfts)*1000:>8.1f} ms")
        print(f"    Min TTFT: {min(ttfts)*1000:>8.1f} ms")
        print(f"    Max TTFT: {max(ttfts)*1000:>8.1f} ms")
        print(f"    Avg E2E:  {sum(e2es)/len(e2es)*1000:>8.1f} ms")

    print_stats("LONG ", long_ttfts, long_e2es)
    print_stats("SHORT", short_ttfts, short_e2es)

    if short_ttfts:
        print(f"\n  >>> SHORT Avg TTFT = {sum(short_ttfts)/len(short_ttfts)*1000:.1f} ms <<<")
        print(f"  >>> SHORT Avg E2E  = {sum(short_e2es)/len(short_e2es)*1000:.1f} ms <<<")
    print(f"{sep}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.1,
                        help="Seconds to wait before sending short request (default: 0.1)")
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.rounds, args.delay))
