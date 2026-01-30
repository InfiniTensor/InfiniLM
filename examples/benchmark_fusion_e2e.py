#!/usr/bin/env python3
"""
Fusion Strategy End-to-End Comparison

端到端推理对比测试：
1. always_fuse: 始终融合
2. never_fuse: 始终不融合  
3. smart_schedule: 智能调度 (基于 profile 决策)

Usage:
    python examples/benchmark_fusion_e2e.py \
        --iluvatar \
        --model_path /data/liuxingyu/OpCompiler/TinyLlama-1.1B-Chat-v1.0 \
        --prompt "What is the capital of France?" \
        --max_new_tokens 50 \
        --runs 3
"""

import infinicore
from transformers import AutoTokenizer
from tokenizers import decoders as _dec
from infinilm.modeling_utils import load_model_state_dict_by_file
from infinilm.distributed import DistConfig
from infinilm.infer_engine import GenerationConfig, InferEngine
from infinilm.fused_infer_engine import FusedInferEngine
import argparse
import sys
import time
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python"))


def get_args():
    parser = argparse.ArgumentParser(description="Fusion Strategy E2E Comparison")
    
    # Device
    parser.add_argument("--cpu", action="store_true", help="Run on CPU")
    parser.add_argument("--nvidia", action="store_true", help="Run on NVIDIA GPU")
    parser.add_argument("--iluvatar", action="store_true", help="Run on ILUVATAR GPU")
    parser.add_argument("--metax", action="store_true", help="Run on MetaX")
    parser.add_argument("--moore", action="store_true", help="Run on Moore")
    
    # Model
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism")
    
    # Generation
    parser.add_argument("--max_new_tokens", type=int, default=50)
    
    # Benchmark
    parser.add_argument("--runs", type=int, default=2, help="Number of runs per prompt")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs")
    
    return parser.parse_args()


# 统一 max_tokens，确保 decode 时间一致，以便公平对比 prefill 性能
DEFAULT_MAX_TOKENS = 30

TEST_PROMPTS = [
    # ========== 极短 Prompt (seq_len < 16) ==========
    {
        "name": "tiny_qa",
        "prompt": "Hi",  # ~2 tokens
        "category": "tiny",
        "estimated_prefill_len": 2,
        "description": "极短输入 (<16 tokens)",
    },
    {
        "name": "short_question",
        "prompt": "What is 2+2?",  # ~6 tokens
        "category": "tiny",
        "estimated_prefill_len": 6,
        "description": "短问题 (<16 tokens)",
    },
    
    # ========== 短 Prompt (16 <= seq_len < 64) ==========
    {
        "name": "medium_short",
        "prompt": "Explain the concept of machine learning in simple terms that a beginner can understand.",  # ~20 tokens
        "category": "short",
        "estimated_prefill_len": 20,
        "description": "中短输入 (16-64 tokens)",
    },
    {
        "name": "code_request",
        "prompt": "Write a Python function to calculate the nth fibonacci number using dynamic programming.",  # ~18 tokens
        "category": "short",
        "estimated_prefill_len": 18,
        "description": "代码请求 (16-64 tokens)",
    },
    {
        "name": "multi_sentence",
        "prompt": "I want to learn programming. What programming language should I start with? Please give me some suggestions and explain why.",  # ~25 tokens
        "category": "short",
        "estimated_prefill_len": 25,
        "description": "多句问题 (16-64 tokens)",
    },
    
    # ========== 中等 Prompt (64 <= seq_len < 128) ==========
    {
        "name": "long_context",
        "prompt": """Here is a story: Once upon a time, in a small village nestled between rolling hills and a sparkling river, there lived a young girl named Aria. She was known throughout the village for her curiosity and kind heart. Every morning, she would wake before dawn. What should Aria do next?""",  # ~65 tokens
        "category": "medium",
        "estimated_prefill_len": 70,
        "description": "中等上下文 (64-128 tokens)",
    },
    {
        "name": "summarization",
        "prompt": """Please summarize the following text:

Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals.

Summary:""",  # ~75 tokens
        "category": "medium",
        "estimated_prefill_len": 75,
        "description": "摘要任务 (64-128 tokens)",
    },
    
    # ========== 长 Prompt (seq_len >= 128) ==========
    {
        "name": "very_long_context",
        "prompt": """Here is a detailed technical document about machine learning:

Machine learning is a subset of artificial intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.

The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide. The primary aim is to allow the computers to learn automatically without human intervention or assistance and adjust actions accordingly.

Machine learning algorithms are often categorized as supervised or unsupervised. What are the key differences between these approaches?""",  # ~150 tokens
        "category": "long",
        "estimated_prefill_len": 150,
        "description": "长输入 (128+ tokens)",
    },
]



def run_inference(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device,
    disable_eos: bool = True,  # 禁用 EOS 提前终止，强制生成完 max_tokens
) -> dict:
    """
    运行一次推理，分开测量 prefill 和 decode 时间

    Returns:
        {
            "output_text": str,
            "prefill_time_ms": float,
            "decode_time_ms": float,
            "total_time_ms": float,
            "prefill_len": int,
            "decode_steps": int,
        }
    """
    # Tokenize
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    input_ids_list = [tokenizer.encode(input_content)]
    prefill_len = len(input_ids_list[0])

    # Reset cache
    model.reset_cache(1, max_new_tokens + prefill_len)

    input_ids_infini = infinicore.from_list(input_ids_list, device=device)
    batch_size, seq_len = input_ids_infini.shape[:2]

    # 初始化 position_ids 和 cache_lengths
    position_ids = infinicore.from_list(
        [list(range(0, seq_len)) for _ in range(batch_size)],
        dtype=infinicore.int64,
        device=device,
    )
    cache_lengths = infinicore.from_list(
        [0],
        dtype=infinicore.int64,
        device=device,
    )

    output_tokens_list = []
    eos_token_id = model.config.eos_token_id
    eos_token_id_list = [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id

    # ====== Prefill 阶段 ======
    prefill_start = time.perf_counter()
    
    logits = model(
        input_ids=input_ids_infini,
        position_ids=position_ids,
        cache_lengths=cache_lengths,
    )
    infinicore.sync_device()
    
    prefill_end = time.perf_counter()
    prefill_time_ms = (prefill_end - prefill_start) * 1000.0

    # 获取第一个 token
    logits_np = logits.to_numpy()
    next_token_id = int(logits_np.argmax(axis=-1)[0, 0])
    output_tokens_list.append(next_token_id)

    # 更新 position_ids 和 cache_lengths
    seq_len = position_ids.shape[-1]
    position_ids = infinicore.from_list(
        [1] * batch_size,
        dtype=infinicore.int64,
        device=device,
    ).view((batch_size, 1)) + position_ids.narrow(1, seq_len - 1, 1)
    cache_lengths = cache_lengths + infinicore.from_list(
        [seq_len],
        dtype=infinicore.int64,
        device=device,
    )

    # ====== Decode 阶段 ======
    decode_start = time.perf_counter()
    decode_steps = 1  # 已经生成了一个 token

    for _ in range(max_new_tokens - 1):
        # 检查 EOS（除非禁用）
        if not disable_eos and next_token_id in eos_token_id_list:
            break

        # 准备下一轮输入
        input_ids_infini = infinicore.from_list(
            [[next_token_id] for _ in range(batch_size)],
            dtype=infinicore.int64,
            device=device,
        )

        # 调用 forward
        logits = model(
            input_ids=input_ids_infini,
            position_ids=position_ids,
            cache_lengths=cache_lengths,
        )
        infinicore.sync_device()

        # Greedy decoding
        logits_np = logits.to_numpy()
        next_token_id = int(logits_np.argmax(axis=-1)[0, 0])
        output_tokens_list.append(next_token_id)
        decode_steps += 1

        # 更新 position_ids 和 cache_lengths
        seq_len = position_ids.shape[-1]
        position_ids = infinicore.from_list(
            [1] * batch_size,
            dtype=infinicore.int64,
            device=device,
        ).view((batch_size, 1)) + position_ids.narrow(1, seq_len - 1, 1)
        cache_lengths = cache_lengths + infinicore.from_list(
            [seq_len],
            dtype=infinicore.int64,
            device=device,
        )

    decode_end = time.perf_counter()
    decode_time_ms = (decode_end - decode_start) * 1000.0

    # 解码输出
    output_text = tokenizer.decode(output_tokens_list, skip_special_tokens=True)

    return {
        "output_text": output_text.strip(),
        "prefill_time_ms": prefill_time_ms,
        "decode_time_ms": decode_time_ms,
        "total_time_ms": prefill_time_ms + decode_time_ms,
        "prefill_len": prefill_len,
        "decode_steps": decode_steps,
    }


def load_model_with_strategy(
    model_path: str,
    device,
    tp: int,
    strategy: str,
    profile_path: str = None,
    debug: bool = False,
) -> tuple:
    """
    根据策略加载模型 (使用 C++ infiniop 融合后端)
    
    Args:
        model_path: 模型路径
        device: 设备
        tp: 张量并行度
        strategy: 策略 - "always_fuse" | "never_fuse" | "smart_schedule"
        profile_path: profile 数据路径 (仅 smart_schedule 时使用)
        debug: 是否打印调试信息
    """
    model_path = os.path.expanduser(model_path)
    
    if strategy == "always_fuse":
        # 使用 FusedInferEngine，始终融合
        model = FusedInferEngine(
            model_path,
            device=device,
            distributed_config=DistConfig(tp),
            enable_fusion=True,
            fusion_mode="always",
            debug=debug,
        )
        
    elif strategy == "never_fuse":
        # 使用普通 InferEngine，不融合
        model = InferEngine(
            model_path,
            device=device,
            distributed_config=DistConfig(tp),
        )
        
    elif strategy == "smart_schedule":
        # 使用 FusedInferEngine，基于 profile 智能调度
        model = FusedInferEngine(
            model_path,
            device=device,
            distributed_config=DistConfig(tp),
            enable_fusion=True,
            fusion_mode="profile",
            profile_path=profile_path,
            debug=debug,
        )
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # 加载权重
    load_model_state_dict_by_file(model, model_path, dtype=model.config.dtype)
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 修复 LLaMA tokenizer
    if getattr(model.config, "model_type", "") == "llama":
        backend = getattr(tokenizer, "backend_tokenizer", None)
        target = getattr(backend, "_tokenizer", backend)
        norm = getattr(target, "normalizer", None)
        dec = getattr(target, "decoder", None)
        sn = repr(norm)[:800] if norm is not None else ""
        sd = repr(dec)[:800] if dec is not None else ""
        has_prepend = "Prepend" in sn
        has_strip = "Strip" in sd
        if has_prepend and has_strip:
            target.decoder = _dec.Sequence([
                _dec.Replace("▁", " "),
                _dec.ByteFallback(),
                _dec.Fuse(),
            ])
    
    return model, tokenizer


def benchmark_strategy(
    model_path: str,
    device,
    tp: int,
    prompt: str,
    max_new_tokens: int,
    strategy: str,
    runs: int,
    warmup: int,
) -> dict:
    """
    对单个策略进行多次测试
    """
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy}")
    print(f"{'='*60}")
    
    # 加载模型
    print(f"Loading model...")
    model, tokenizer = load_model_with_strategy(model_path, device, tp, strategy)
    
    prefill_times = []
    decode_times = []
    total_times = []
    
    # Warmup
    print(f"Warmup ({warmup} runs)...")
    for i in range(warmup):
        run_inference(model, tokenizer, prompt, max_new_tokens, device)
    
    # Timed runs
    print(f"Benchmark ({runs} runs)...")
    for i in range(runs):
        result = run_inference(model, tokenizer, prompt, max_new_tokens, device)
        prefill_times.append(result["prefill_time_ms"])
        decode_times.append(result["decode_time_ms"])
        total_times.append(result["total_time_ms"])
        print(f"  Run {i+1}: prefill={result['prefill_time_ms']:.2f}ms, decode={result['decode_time_ms']:.2f}ms, total={result['total_time_ms']:.2f}ms")
    
    # Show sample output
    print(f"Sample output: {result['output_text'][:100]}...")
    
    avg_prefill = sum(prefill_times) / len(prefill_times)
    avg_decode = sum(decode_times) / len(decode_times)
    avg_total = sum(total_times) / len(total_times)
    
    print(f"Results: avg_prefill={avg_prefill:.2f}ms, avg_decode={avg_decode:.2f}ms, avg_total={avg_total:.2f}ms")
    
    # 获取融合统计（如果有）
    fusion_stats = None
    if hasattr(model, 'get_stats'):
        fusion_stats = model.get_stats()
        print(f"Fusion stats: {fusion_stats}")
    
    return {
        "strategy": strategy,
        "avg_prefill_ms": avg_prefill,
        "avg_decode_ms": avg_decode,
        "avg_total_ms": avg_total,
        "prefill_len": result["prefill_len"],
        "decode_steps": result["decode_steps"],
        "fusion_stats": fusion_stats,
    }


def run_all_prompts_with_strategy(
    model,
    tokenizer,
    prompts: list,
    runs: int,
    warmup: int,
    device,
    max_tokens: int = DEFAULT_MAX_TOKENS,  # 统一的 max_tokens
) -> dict:
    """对一个策略运行所有 prompts，分开记录 prefill/decode 时间"""
    results = {}
    
    for p in prompts:
        name = p["name"]
        prompt = p["prompt"]
        
        prefill_times = []
        decode_times = []
        total_times = []
        prefill_len = 0
        decode_steps = 0
        
        # Warmup
        for _ in range(warmup):
            run_inference(model, tokenizer, prompt, max_tokens, device)
        
        # Timed runs
        for _ in range(runs):
            result = run_inference(model, tokenizer, prompt, max_tokens, device)
            prefill_times.append(result["prefill_time_ms"])
            decode_times.append(result["decode_time_ms"])
            total_times.append(result["total_time_ms"])
            prefill_len = result["prefill_len"]
            decode_steps = result["decode_steps"]
        
        results[name] = {
            "avg_prefill_ms": sum(prefill_times) / len(prefill_times),
            "avg_decode_ms": sum(decode_times) / len(decode_times),
            "avg_total_ms": sum(total_times) / len(total_times),
            "prefill_len": prefill_len,
            "decode_steps": decode_steps,
            "category": p["category"],
            "description": p["description"],
        }
    
    return results


def main():
    args = get_args()
    
    # 确定设备
    if args.nvidia:
        device_str = "cuda"
    elif args.iluvatar:
        device_str = "cuda"  # ILUVATAR 使用 cuda 接口
    elif args.cpu:
        device_str = "cpu"
    elif args.metax:
        device_str = "maca"
    elif args.moore:
        device_str = "musa"
    else:
        print("Please specify device: --cpu, --nvidia, --iluvatar, --metax, or --moore")
        sys.exit(1)
    
    device = infinicore.device(device_str, 0)
    
    print("=" * 80)
    print("Fusion Strategy E2E Comparison - Multi-Prompt Benchmark")
    print("=" * 80)
    print(f"Device: {device_str}")
    print(f"Model: {args.model_path}")
    print(f"Runs per prompt: {args.runs}, Warmup: {args.warmup}")
    print(f"Test prompts: {len(TEST_PROMPTS)}")
    
    # 测试三种策略
    strategies = ["never_fuse", "always_fuse", "smart_schedule"]
    all_results = {}
    
    for strategy in strategies:
        print(f"\n{'='*80}")
        print(f"📌 Strategy: {strategy}")
        print(f"{'='*80}")
        
        try:
            print("Loading model...")
            model, tokenizer = load_model_with_strategy(
                args.model_path, device, args.tp, strategy
            )
            
            print(f"Running {len(TEST_PROMPTS)} prompts...")
            results = run_all_prompts_with_strategy(
                model, tokenizer, TEST_PROMPTS, args.runs, args.warmup, device
            )
            
            all_results[strategy] = results
            
            # 显示该策略的结果
            print(f"\n{'Prompt':<20} {'Category':<10} {'PrefillLen':<10} {'Prefill(ms)':<12} {'Decode(ms)':<12} {'Total(ms)':<12}")
            print("-" * 76)
            for name, r in results.items():
                print(f"{name:<20} {r['category']:<10} {r['prefill_len']:<10} {r['avg_prefill_ms']:<12.2f} {r['avg_decode_ms']:<12.2f} {r['avg_total_ms']:<12.2f}")
            
            total_prefill = sum(r["avg_prefill_ms"] for r in results.values())
            total_decode = sum(r["avg_decode_ms"] for r in results.values())
            total = sum(r["avg_total_ms"] for r in results.values())
            print("-" * 76)
            print(f"{'TOTAL':<20} {'':<10} {'':<10} {total_prefill:<12.2f} {total_decode:<12.2f} {total:<12.2f}")
            
            # 释放模型内存
            print("Releasing model memory...")
            del model
            del tokenizer
            import gc
            gc.collect()
            infinicore.sync_device()
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[strategy] = {"error": str(e)}
    
    # ========== Detailed Comparison ==========
    print("\n" + "=" * 80)
    print("📊 PER-PROMPT COMPARISON")
    print("=" * 80)

    valid_strategies = [s for s in strategies if "error" not in all_results.get(s, {})]

    if len(valid_strategies) >= 2:
        # Header
        header = f"{'Prompt':<20}"
        for s in valid_strategies:
            header += f" {s:<12}"
        header += " Best"
        print(header)
        print("-" * (32 + 12 * len(valid_strategies)))

        prompt_winners = {"never_fuse": 0, "always_fuse": 0, "smart_schedule": 0}

        for p in TEST_PROMPTS:
            name = p["name"]

            row = f"{name:<20}"
            times = {}
            for s in valid_strategies:
                if name in all_results[s]:
                    t = all_results[s][name]["avg_time"]
                    times[s] = t
                    row += f" {t:<12.1f}"
                else:
                    row += f" {'N/A':<12}"

            if times:
                best = min(times, key=times.get)
                prompt_winners[best] = prompt_winners.get(best, 0) + 1
                row += f" {best:<12}"

            print(row)

        # Totals
        print("-" * (32 + 12 * len(valid_strategies)))
        row = f"{'TOTAL':<20}"
        totals = {}
        for s in valid_strategies:
            total = sum(all_results[s][p["name"]]["avg_time"] for p in TEST_PROMPTS if p["name"] in all_results[s])
            totals[s] = total
            row += f" {total:<12.1f}"

        best_total = min(totals, key=totals.get)
        row += f" {best_total:<12} ⭐"
        print(row)
        
        # Strategy Summary
        print("\n" + "=" * 80)
        print("📈 STRATEGY SUMMARY")
        print("=" * 80)
        
        baseline = max(totals.values())
        print(f"\n{'Strategy':<20} {'Total (ms)':<15} {'Speedup':<10} {'Wins':<10}")
        print("-" * 60)
        for s in valid_strategies:
            speedup = baseline / totals[s] if totals[s] > 0 else 0
            wins = prompt_winners.get(s, 0)
            marker = "⭐" if s == best_total else ""
            print(f"{s:<20} {totals[s]:<15.2f} {speedup:<10.2f}x {wins:<10} {marker}")
        
        # Category Analysis
        print("\n" + "=" * 80)
        print("📊 CATEGORY ANALYSIS")
        print("=" * 80)
        
        categories = ["decode_heavy", "balanced", "prefill_heavy"]
        for cat in categories:
            cat_prompts = [p for p in TEST_PROMPTS if p["category"] == cat]
            if not cat_prompts:
                continue
            
            print(f"\n【{cat}】({len(cat_prompts)} prompts)")
            cat_totals = {}
            for s in valid_strategies:
                total = sum(
                    all_results[s][p["name"]]["avg_time"] 
                    for p in cat_prompts 
                    if p["name"] in all_results[s]
                )
                cat_totals[s] = total
            
            cat_baseline = max(cat_totals.values())
            for s in valid_strategies:
                speedup = cat_baseline / cat_totals[s] if cat_totals[s] > 0 else 0
                best_marker = "⭐" if cat_totals[s] == min(cat_totals.values()) else ""
                print(f"  {s:<18}: {cat_totals[s]:.2f}ms ({speedup:.2f}x) {best_marker}")
    
    print("\n" + "=" * 80)
    print("✅ Benchmark Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()


