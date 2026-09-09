import argparse
import json
import statistics
import time

import infinicore


def measure_get_stream(iterations):
    get_stream = infinicore.get_stream
    start = time.perf_counter_ns()
    for _ in range(iterations):
        get_stream()
    return (time.perf_counter_ns() - start) / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=10_000)
    parser.add_argument("--iterations", type=int, default=1_000_000)
    parser.add_argument("--repeat", type=int, default=7)
    args = parser.parse_args()

    if args.warmup < 0 or args.iterations <= 0 or args.repeat <= 0:
        parser.error(
            "warmup must be non-negative; iterations and repeat must be positive"
        )

    infinicore.set_device(args.device)
    expected_stream = infinicore.get_stream()
    for _ in range(args.warmup):
        if infinicore.get_stream() != expected_stream:
            raise RuntimeError("runtime stream changed during warmup")

    samples = [measure_get_stream(args.iterations) for _ in range(args.repeat)]
    if infinicore.get_stream() != expected_stream:
        raise RuntimeError("runtime stream changed during benchmark")

    print(
        json.dumps(
            {
                "benchmark": "infinicore.get_stream",
                "device": args.device,
                "iterations": args.iterations,
                "repeat": args.repeat,
                "unit": "ns/call",
                "min": min(samples),
                "median": statistics.median(samples),
                "max": max(samples),
                "samples": samples,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
