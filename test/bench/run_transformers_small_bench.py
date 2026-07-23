#!/usr/bin/env python3
"""
Quick benchmark on a small C-Eval subset using the Transformers backend.

Uses the local C-Eval cache under /data1/pepe/datasets by default.

Usage:
  cd /workspace_infini/InfiniLM
  python test/bench/run_transformers_small_bench.py

  # Use a different small model
  python test/bench/run_transformers_small_bench.py --model /data1/TinyLlama-1.1B-Chat-v1.0

  # Evaluate another C-Eval subject
  python test/bench/run_transformers_small_bench.py \
      --subject high_school_physics --num-samples 10
"""

from __future__ import annotations

import argparse
import os
import sys

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
INFINILM_ROOT = os.path.abspath(os.path.join(BENCH_DIR, "../.."))
DEFAULT_CACHE_DIR = "/data1/pepe/datasets"
sys.path.insert(0, os.path.join(INFINILM_ROOT, "python"))
sys.path.insert(0, BENCH_DIR)

from backends import TransformersBenchmark
from test_benchmark import evaluate_samples, load_dataset_samples


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a small C-Eval benchmark with the Transformers backend"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/data1/Qwen3-0.6B",
        help="Path to local model directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda", "mlu", "musa", "npu"],
        help="Torch device type for Transformers backend",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        help="Tensor parallel size (Transformers uses device_map=auto when tp>1)",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="middle_school_mathematics",
        help="C-Eval subject name, comma-separated, or 'all'",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["test", "val", "all"],
        help="C-Eval dataset split",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to evaluate per subject",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of prompts generated in one Transformers call",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Maximum new tokens to generate per sample",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help="HuggingFace datasets cache directory",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional CSV output path",
    )
    return parser.parse_args()


class _DatasetArgs:
    """Minimal args object for load_dataset_samples()."""

    def __init__(self, args):
        self.bench = "ceval"
        self.subject = args.subject
        self.split = args.split
        self.num_samples = args.num_samples
        self.cache_dir = args.cache_dir


def load_samples(args):
    dataset_args = _DatasetArgs(args)
    dataset_args.cache_dir = os.path.expanduser(dataset_args.cache_dir)
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    subject_samples = load_dataset_samples(dataset_args)
    if not subject_samples:
        raise RuntimeError(
            f"No C-Eval samples loaded for subject={args.subject}, "
            f"cache_dir={dataset_args.cache_dir}"
        )
    return subject_samples, "ceval"


def write_csv(path, results, overall_correct, overall_total, overall_accuracy):
    import csv

    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Subject", "Correct", "Total", "Accuracy"])
        for result in results:
            writer.writerow(
                [
                    result["subject"],
                    result["correct"],
                    result["total"],
                    f"{result['accuracy']:.4f}",
                ]
            )
        writer.writerow(
            ["Overall", overall_correct, overall_total, f"{overall_accuracy:.4f}"]
        )
    print(f"CSV file written: {path}")


def main():
    args = parse_args()

    print("=" * 60)
    print("Transformers Small C-Eval Benchmark")
    print("=" * 60)
    print(f"Model:          {args.model}")
    print(f"Device:         {args.device}")
    print(f"TP:             {args.tp}")
    print(f"Subject:        {args.subject}")
    print(f"Split:          {args.split}")
    print(f"Cache dir:      {args.cache_dir}")
    print(f"Num samples:    {args.num_samples}")
    print(f"Batch size:     {args.batch_size}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print("=" * 60)

    print("\nSTEP 1: Loading C-Eval dataset")
    subject_samples, benchmark = load_samples(args)
    total_loaded = sum(len(samples) for samples in subject_samples.values())
    print(f"Loaded {total_loaded} samples from {len(subject_samples)} subject(s)")

    print("\nSTEP 2: Loading Transformers model")
    model = TransformersBenchmark(
        args.model,
        device_type_str=args.device,
        tensor_parallel_size=args.tp,
        benchmark=benchmark,
    )

    print("\nSTEP 3: Evaluating")
    all_results = []
    for subject_name, samples in subject_samples.items():
        print(f"\n{'=' * 60}")
        print(f"Evaluating subject: {subject_name}")
        print(f"{'=' * 60}")
        result = evaluate_samples(
            model,
            samples,
            benchmark,
            args.max_new_tokens,
            subject_name,
            args.batch_size,
        )
        all_results.append(result)

    model.destroy_model_instance()

    print(f"\n{'=' * 60}")
    print("OVERALL RESULTS")
    print(f"{'=' * 60}")
    for result in all_results:
        print(
            f"Subject '{result['subject']}': "
            f"{result['correct']}/{result['total']} = {result['accuracy']:.2%}"
        )

    overall_correct = sum(r["correct"] for r in all_results)
    overall_total = sum(r["total"] for r in all_results)
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0
    print(
        f"Overall: {overall_correct}/{overall_total} = {overall_accuracy:.2%}"
    )
    print(f"Total latency: {model.total_time:.2f} s")
    print(f"Total tokens: {model.total_tokens}")
    if model.total_time > 0:
        print(
            f"Overall throughput: {model.total_tokens / model.total_time:.2f} tok/s"
        )

    if args.output_csv:
        write_csv(
            args.output_csv,
            all_results,
            overall_correct,
            overall_total,
            overall_accuracy,
        )


if __name__ == "__main__":
    main()
