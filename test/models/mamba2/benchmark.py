"""Small reproducible raw-token benchmark for the NVIDIA Mamba2 adapter."""

import argparse
import json
import os
import subprocess
import time

import torch

import infinicore
from infinilm.cache import PagedKVCacheConfig
from infinilm.distributed import DistConfig
from infinilm.infer_engine import InferEngine
from infinilm.modeling_utils import load_model_state_dict_by_file


def gpu_memory_mib():
    row = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    used, total = (int(value.strip()) for value in row.split(",", 1))
    return {"used_mib": used, "total_mib": total}


def forward(engine, batch_size, input_len, step, state_rows):
    decode = step > 0
    length = 1 if decode else input_len
    tokens = [1] * (batch_size * length)
    offsets = [index * length for index in range(batch_size + 1)]
    past = [input_len + step - 1 if decode else 0] * batch_size
    total = [value + length for value in past]
    positions = [
        position
        for past_length in past
        for position in range(past_length, past_length + length)
    ]
    cu_seqlens = [0]
    for value in total:
        cu_seqlens.append(cu_seqlens[-1] + value)
    return engine.forward_raw(
        infinicore.from_list([tokens], dtype=infinicore.int64),
        position_ids=infinicore.from_list(positions, dtype=infinicore.int64),
        past_kv_lengths=infinicore.from_list(past, dtype=infinicore.int32),
        total_kv_lengths=infinicore.from_list(total, dtype=infinicore.int32),
        input_offsets=infinicore.from_list(offsets, dtype=infinicore.int32),
        cu_seqlens=infinicore.from_list(cu_seqlens, dtype=infinicore.int32),
        mamba_init_state_indices=infinicore.from_list(
            [0 if not decode else row for row in state_rows],
            dtype=infinicore.int32,
        ),
        mamba_final_state_indices=infinicore.from_list(
            state_rows, dtype=infinicore.int32
        ),
        sample_all_positions=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("MAMBA2_MODEL_PATH"), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    engine = InferEngine(
        args.model,
        device=infinicore.device("cuda", 0),
        distributed_config=DistConfig(1),
        cache_config=PagedKVCacheConfig(
            num_blocks=64, block_size=16, max_batch_size=args.batch_size
        ),
        attention_backend="paged-attn",
        weight_load_mode="sync",
    )
    load_model_state_dict_by_file(engine, args.model, dtype=infinicore.float16)
    rows = list(range(1, args.batch_size + 1))
    print(json.dumps({"memory_after_load": gpu_memory_mib()}), flush=True)

    for _ in range(2):
        forward(engine, args.batch_size, args.input_len, 0, rows)
        for step in range(1, args.output_len + 1):
            forward(engine, args.batch_size, args.input_len, step, rows)
    torch.cuda.synchronize()

    prefill_samples = []
    decode_samples = []
    for _ in range(args.runs):
        start = time.perf_counter()
        forward(engine, args.batch_size, args.input_len, 0, rows)
        torch.cuda.synchronize()
        prefill_samples.append(time.perf_counter() - start)

        start = time.perf_counter()
        for step in range(1, args.output_len + 1):
            forward(engine, args.batch_size, args.input_len, step, rows)
        torch.cuda.synchronize()
        decode_samples.append(time.perf_counter() - start)

    prefill = sum(prefill_samples) / len(prefill_samples)
    decode = sum(decode_samples) / len(decode_samples)
    print(
        json.dumps(
            {
                "batch_size": args.batch_size,
                "input_len": args.input_len,
                "output_len": args.output_len,
                "prefill_ms": prefill * 1000,
                "decode_ms": decode * 1000,
                "decode_tokens_per_second": (
                    args.batch_size * args.output_len / decode
                ),
                "memory_after_benchmark": gpu_memory_mib(),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
