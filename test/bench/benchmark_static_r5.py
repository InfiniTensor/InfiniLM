"""Measure fixed-workload static generation and retain tokens for A/B checks."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from infinilm import _core
from infinilm.cache import StaticKVCacheConfig
from infinilm.distributed import DistConfig
from infinilm.infer_engine import GenerationConfig, InferEngine
from infinilm.modeling_utils import load_model_state_dict_by_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-ids", type=Path, required=True)
    parser.add_argument("--device", default="iluvatar:0")
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--cache-capacity", type=int)
    parser.add_argument("--static-r5-profile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.input_ids.read_text())
    if (
        not isinstance(rows, list)
        or not rows
        or not isinstance(rows[0], list)
        or not rows[0]
        or any(
            not isinstance(row, list)
            or len(row) != len(rows[0])
            or any(type(token) is not int or token < 0 for token in row)
            for row in rows
        )
    ):
        parser.error("`--input-ids` must contain a nonempty rectangular token matrix.")
    if args.new_tokens < 1 or args.repeat < 1 or args.warmup < 0:
        parser.error("Token and repeat counts must be positive; warmup may be zero.")
    capacity = (
        len(rows[0]) + args.new_tokens
        if args.cache_capacity is None
        else args.cache_capacity
    )
    if capacity < len(rows[0]) + args.new_tokens:
        parser.error("`--cache-capacity` must cover the complete request.")

    def cache_config():
        return StaticKVCacheConfig(
            max_batch_size=len(rows),
            max_cache_len=capacity,
        )

    engine = InferEngine(
        args.model,
        device=_core.device(args.device),
        distributed_config=DistConfig(1),
        cache_config=cache_config(),
        enable_graph_compiling=False,
        static_r5_profile=args.static_r5_profile,
    )
    load_model_state_dict_by_file(engine, args.model, dtype=engine.dtype)
    records = []
    reference = None
    for iteration in range(args.warmup + args.repeat):
        engine.reset_cache(cache_config())
        input_ids = _core.from_list(rows, dtype=_core.int64)
        start = time.perf_counter()
        outputs = engine.generate(
            input_ids,
            GenerationConfig(
                max_new_tokens=args.new_tokens,
                top_k=1,
                top_p=1.0,
                temperature=1.0,
                stop_on_eos=False,
            ),
        )
        tokens = [output.to_numpy().reshape(-1).tolist() for output in outputs]
        elapsed_ms = (time.perf_counter() - start) * 1000
        if len(tokens) != args.new_tokens or any(
            len(row) != len(rows) for row in tokens
        ):
            raise RuntimeError("Generation returned an unexpected output shape.")
        if reference is not None and tokens != reference:
            raise RuntimeError("Repeated greedy generation produced different tokens.")
        reference = tokens
        records.append(
            {
                "warmup": iteration < args.warmup,
                "elapsed_ms": elapsed_ms,
                "tokens": tokens,
            }
        )

    result = {
        "batch_size": len(rows),
        "input_length": len(rows[0]),
        "output_length": args.new_tokens,
        "cache_capacity": capacity,
        "input_sha256": hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
        "static_r5_profile": args.static_r5_profile,
        "median_ms": statistics.median(
            row["elapsed_ms"] for row in records if not row["warmup"]
        ),
        "requests": records,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({key: value for key, value in result.items() if key != "requests"})
    )


if __name__ == "__main__":
    main()
