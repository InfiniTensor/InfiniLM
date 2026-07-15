#!/usr/bin/env python3
"""Exercise dynamic request arrival and completion with PP enabled."""

import argparse
import asyncio
import json

from infinilm.llm import AsyncLLMEngine, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pp", type=int, default=2)
    return parser.parse_args()


async def collect(engine, request):
    token_ids = []
    finish_reason = None
    async for output in engine.stream_request(request, timeout=30.0):
        if output.token_id >= 0:
            token_ids.append(output.token_id)
        if output.finished:
            finish_reason = output.finish_reason.value
    return {
        "request_id": request.request_id,
        "token_ids": token_ids,
        "finish_reason": finish_reason,
    }


def add_request(engine, request_id, prompt, max_tokens):
    return engine.add_request(
        messages=None,
        prompt=prompt,
        request_id=request_id,
        sampling_params=SamplingParams(
            max_tokens=max_tokens,
            temperature=1.0,
            top_p=1.0,
            top_k=1,
            ignore_eos=True,
        ),
    )


async def run(args):
    engine = AsyncLLMEngine(
        model_path=args.model,
        device="cuda",
        tensor_parallel_size=1,
        pipeline_parallel_size=args.pp,
        cache_type="paged",
        max_batch_size=8,
        max_tokens=16,
        num_blocks=128,
        block_size=256,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        enable_graph=False,
        attn_backend="paged-attn",
    )

    engine.start()
    try:
        first = [
            add_request(
                engine,
                "arrival-a",
                "The first dynamically scheduled request says",
                12,
            ),
            add_request(
                engine,
                "arrival-b",
                "A short request returns",
                4,
            ),
        ]
        tasks = [asyncio.create_task(collect(engine, request)) for request in first]

        # Add more work after the background step thread has started executing.
        await asyncio.sleep(0.05)
        if all(request.is_finished() for request in first):
            raise AssertionError(
                "initial requests finished before dynamic arrivals were added"
            )
        later = [
            add_request(
                engine,
                "arrival-c",
                "A request that joins an active batch says",
                7,
            ),
            add_request(
                engine,
                "arrival-d",
                "The final short request returns",
                3,
            ),
        ]
        tasks.extend(
            asyncio.create_task(collect(engine, request)) for request in later
        )
        results = await asyncio.gather(*tasks)
    finally:
        engine.stop()

    expected_lengths = {
        "arrival-a": 12,
        "arrival-b": 4,
        "arrival-c": 7,
        "arrival-d": 3,
    }
    actual_lengths = {
        result["request_id"]: len(result["token_ids"])
        for result in results
    }
    if actual_lengths != expected_lengths:
        raise AssertionError(
            f"unexpected output routing/lengths: {actual_lengths}"
        )

    stats = engine.engine.model_runner.model_engine.get_pipeline_transport_stats()
    expected_transfers = sum(expected_lengths.values()) * (args.pp - 1)
    if stats["name"] != "gpu-peer-copy-async-event":
        raise AssertionError(f"unexpected PP transport: {stats}")
    if stats["transfers"] != expected_transfers:
        raise AssertionError(
            f"unexpected PP transfer count: {stats}, expected={expected_transfers}"
        )
    print(
        "PP_ASYNC_JSON="
        + json.dumps(
            {"results": results, "transport_stats": stats},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
