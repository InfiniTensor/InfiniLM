"""Reproducible single-GPU RWKV5 scheduler benchmark."""

import argparse
import json
import math
import os
import statistics
import time

import torch

from infinilm.llm.llm import LLM
from infinilm.llm.request import InferenceRequest
from infinilm.llm.sampling_params import SamplingParams


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def make_prompt_tokens(tokenizer, length: int) -> list[int]:
    seed = tokenizer.encode(
        "The quick brown fox jumps over the lazy dog. "
        "InfiniLM RWKV5 scheduler benchmark. "
    )
    return (seed * ((length + len(seed) - 1) // len(seed)))[:length]


def run_case(engine, tokenizer, batch_size: int, input_len: int, output_len: int):
    prompt_tokens = make_prompt_tokens(tokenizer, input_len)
    sampling = SamplingParams(
        max_tokens=output_len,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        ignore_eos=True,
    )
    requests = []
    for index in range(batch_size):
        request = InferenceRequest(
            request_id=f"rwkv5-bench-{batch_size}-{input_len}-{index}",
            prompt_token_ids=prompt_tokens,
            sampling_params=sampling,
            eos_token_ids=engine.eos_token_ids,
        )
        requests.append(request)
        engine.add_request(request)

    torch.cuda.synchronize()
    start = time.perf_counter()
    while any(request.get_num_generated_tokens() == 0 for request in requests):
        did_work, _ = engine.step()
        if not did_work:
            raise RuntimeError("scheduler made no progress during prefill")
    torch.cuda.synchronize()
    ttft = time.perf_counter() - start
    generated_at_ttft = [
        request.get_num_generated_tokens() for request in requests
    ]

    decode_start = time.perf_counter()
    while any(not request.is_finished() for request in requests):
        did_work, _ = engine.step()
        if not did_work:
            raise RuntimeError("scheduler made no progress during decode")
    torch.cuda.synchronize()
    decode_time = time.perf_counter() - decode_start

    generated = [request.get_num_generated_tokens() for request in requests]
    expected = [output_len] * batch_size
    if generated != expected:
        raise AssertionError(f"unexpected output lengths: {generated} != {expected}")

    remaining_tokens = [output_len - count for count in generated_at_ttft]
    decode_tokens = sum(remaining_tokens)
    decode_steps = max(remaining_tokens, default=0)
    return {
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
        "ttft_ms": ttft * 1000.0,
        "tokens_generated_before_all_ttft": sum(generated_at_ttft),
        "decode_time_ms": decode_time * 1000.0,
        "decode_itl_ms": decode_time * 1000.0 / max(decode_steps, 1),
        "decode_throughput_tok_s": (
            decode_tokens / decode_time if decode_tokens else 0.0
        ),
        "end_to_end_throughput_tok_s": batch_size * output_len / (ttft + decode_time),
    }


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percent * len(ordered)) - 1)
    return ordered[index]


def summarize(samples: list[dict]) -> dict:
    summary = {
        key: samples[0][key]
        for key in ("batch_size", "input_len", "output_len")
    }
    summary["runs"] = len(samples)
    for metric in (
        "ttft_ms",
        "decode_itl_ms",
        "decode_throughput_tok_s",
        "end_to_end_throughput_tok_s",
    ):
        values = [sample[metric] for sample in samples]
        summary[f"{metric}_p50"] = statistics.median(values)
        summary[f"{metric}_p90"] = percentile(values, 0.9)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default=os.environ.get("RWKV5_MODEL_PATH"), required=False
    )
    parser.add_argument("--batch-sizes", default="1,2,4", type=parse_int_list)
    parser.add_argument("--input-lens", default="32,128,512", type=parse_int_list)
    parser.add_argument("--output-lens", default="32,128", type=parse_int_list)
    parser.add_argument("--cache-type", choices=("paged", "static"), default="paged")
    parser.add_argument("--num-blocks", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--force-attention-kv",
        "--legacy-kv-reservation",
        dest="force_attention_kv",
        action="store_true",
        help="Ablate the pure-recurrent path by forcing attention KV scheduling",
    )
    args = parser.parse_args()
    if not args.model:
        parser.error("--model or RWKV5_MODEL_PATH is required")
    if args.cache_type == "static" and args.batch_sizes != [1]:
        parser.error(
            "static cache uses the single-request scheduler; use --batch-sizes 1"
        )
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.force_attention_kv and args.cache_type != "paged":
        parser.error("--force-attention-kv requires --cache-type paged")

    model = LLM(
        model_path=args.model,
        device="cuda",
        dtype="bfloat16",
        tensor_parallel_size=1,
        cache_type=args.cache_type,
        max_batch_size=max(args.batch_sizes),
        max_tokens=max(args.output_lens),
        num_blocks=args.num_blocks,
        block_size=16,
        enable_graph=False,
        attn_backend="paged-attn",
        weight_load_mode="sync",
        enable_prefix_caching=False,
    )
    try:
        engine = model.engine
        tokenizer = engine.tokenizer
        if args.force_attention_kv:
            engine.scheduler.cacheless_state_model = False
        for _ in range(args.warmup):
            run_case(engine, tokenizer, 1, min(args.input_lens), min(args.output_lens))

        print(
            json.dumps(
                {
                    "type": "metadata",
                    "cache_type": args.cache_type,
                    "num_blocks": args.num_blocks,
                    "force_attention_kv": args.force_attention_kv,
                    "runs": args.runs,
                }
            )
        )
        for batch_size in args.batch_sizes:
            for input_len in args.input_lens:
                for output_len in args.output_lens:
                    samples = []
                    for run_index in range(args.runs):
                        result = run_case(
                            engine, tokenizer, batch_size, input_len, output_len
                        )
                        samples.append(result)
                        print(
                            json.dumps(
                                {"type": "sample", "run": run_index + 1, **result}
                            ),
                            flush=True,
                        )
                    print(
                        json.dumps({"type": "summary", **summarize(samples)}),
                        flush=True,
                    )
    finally:
        model.close()


if __name__ == "__main__":
    main()
