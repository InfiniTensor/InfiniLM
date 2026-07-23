#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import time

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
INFINILM_ROOT = os.path.abspath(os.path.join(BENCH_DIR, "../.."))
sys.path.insert(0, os.path.join(INFINILM_ROOT, "python"))
sys.path.insert(0, BENCH_DIR)

from backends import TransformersBenchmark
from run_transformers_small_bench import load_samples
from test_benchmark import evaluate_samples


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data1/DeepSeek-V4-Flash-BF16")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--subject", default="middle_school_mathematics")
    parser.add_argument("--split", default="val")
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cache-dir", default="/data1/pepe/datasets")
    return parser.parse_args()


def run_case(model, samples, benchmark, max_new_tokens, subject, batch_size):
    model.total_time = 0.0
    model.total_tokens = 0
    started = time.perf_counter()
    result = evaluate_samples(
        model,
        samples,
        benchmark,
        max_new_tokens,
        subject,
        batch_size,
    )
    wall_time = time.perf_counter() - started
    return {
        "batch_size": batch_size,
        "correct": result["correct"],
        "total": result["total"],
        "accuracy": result["accuracy"],
        "generation_time_s": model.total_time,
        "wall_time_s": wall_time,
        "tokens": model.total_tokens,
        "throughput_tok_s": model.total_tokens / model.total_time,
    }


def main():
    args = parse_args()
    samples_by_subject, benchmark = load_samples(args)
    samples = samples_by_subject[args.subject]

    load_started = time.perf_counter()
    model = TransformersBenchmark(
        args.model,
        device_type_str="cuda",
        tensor_parallel_size=args.tp,
        benchmark=benchmark,
    )
    first_constructor_s = time.perf_counter() - load_started

    print("WARMUP_START")
    evaluate_samples(
        model, samples[:1], benchmark, 4, args.subject, batch_size=1
    )
    print("WARMUP_END")

    sequential = run_case(
        model, samples, benchmark, args.max_new_tokens, args.subject, 1
    )
    batched = run_case(
        model,
        samples,
        benchmark,
        args.max_new_tokens,
        args.subject,
        args.batch_size,
    )

    reuse_started = time.perf_counter()
    reused = TransformersBenchmark(
        args.model,
        device_type_str="cuda",
        tensor_parallel_size=args.tp,
        benchmark=benchmark,
    )
    resident_constructor_s = time.perf_counter() - reuse_started

    summary = {
        "first_constructor_s": first_constructor_s,
        "resident_constructor_s": resident_constructor_s,
        "resident_model_reused": reused.model is model.model,
        "sequential": sequential,
        "batched": batched,
        "batch_speedup": sequential["generation_time_s"]
        / batched["generation_time_s"],
    }
    print("OPTIMIZATION_BENCHMARK_SUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    reused.destroy_model_instance()
    model.destroy_model_instance()


if __name__ == "__main__":
    main()
