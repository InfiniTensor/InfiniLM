#!/usr/bin/env python3
"""Executable multi-request PP smoke test with machine-readable output."""

import argparse
import json
import time

from infinilm.llm import LLM, SamplingParams


PROMPTS = [
    "The capital of France is",
    "A prime number after seven is",
    "Write one short sentence about CUDA",
    "The result of two plus three is",
    "Pipeline parallel inference means",
    "A useful property of unit tests is",
    "The opposite of cold is",
    "One byte contains this many bits:",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output-len", type=int, default=4)
    parser.add_argument("--num-blocks", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    prompts = [
        f"{PROMPTS[index % len(PROMPTS)]} [request {index}]"
        for index in range(args.batch_size)
    ]

    llm = LLM(
        model_path=args.model,
        device="cuda",
        tensor_parallel_size=1,
        pipeline_parallel_size=args.pp,
        cache_type="paged",
        max_batch_size=args.batch_size,
        max_tokens=args.output_len,
        num_blocks=args.num_blocks,
        block_size=256,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        enable_graph=False,
        attn_backend="paged-attn",
    )

    sampling = SamplingParams(
        max_tokens=args.output_len,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        ignore_eos=True,
    )
    started = time.perf_counter()
    outputs = llm.generate(
        prompts=prompts,
        sampling_params=sampling,
        use_tqdm=False,
    )
    elapsed = time.perf_counter() - started

    stats = llm.engine.model_runner.model_engine.get_pipeline_transport_stats()
    payload = {
        "pp": args.pp,
        "batch_size": args.batch_size,
        "elapsed_seconds": elapsed,
        "request_ids": [output.request_id for output in outputs],
        "prompt_token_ids": [output.prompt_token_ids for output in outputs],
        "output_token_ids": [output.outputs[0].token_ids for output in outputs],
        "transport_stats": stats,
    }
    print("PP_SMOKE_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
