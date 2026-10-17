#!/usr/bin/env python3
"""Run fixed-length serving benchmarks over a matrix of load shapes."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--benchmark", default="/root/apps/llm-benchmark/iluvatar_bench")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--input-lengths", type=int, nargs="+", default=[32, 256, 1024])
    parser.add_argument("--output-lengths", type=int, nargs="+", default=[256, 1024, 2048])
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--starting-seed", type=int, default=1000)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def matching_result(rows, concurrency, input_len, output_len):
    for row in reversed(rows):
        if (
            row.get("max_concurrency") == concurrency
            and row.get("random_input_len") == input_len
            and row.get("random_output_len") == output_len
            and row.get("failed") == 0
        ):
            return row
    return None


def append_jsonl(path: Path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    args = parse_args()
    output = Path(args.output)
    summary = Path(args.summary)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)

    existing = read_jsonl(output)
    seed = args.starting_seed
    failures = []

    for concurrency in args.concurrencies:
        for input_len in args.input_lengths:
            for output_len in args.output_lengths:
                seed += 1
                label = f"c{concurrency}-i{input_len}-o{output_len}-s{seed}"
                previous = (
                    matching_result(existing, concurrency, input_len, output_len)
                    if args.skip_existing
                    else None
                )
                if previous is not None:
                    print(f"SKIP {label}", flush=True)
                    continue

                command = [
                    args.benchmark,
                    "sgl-perf",
                    "--backend",
                    args.backend,
                    "--host",
                    args.host,
                    "--port",
                    str(args.port),
                    "--model",
                    args.model,
                    "--tokenizer",
                    args.tokenizer,
                    "--dataset-name",
                    "random-ids",
                    "--tokenize-prompt",
                    "--random-input-len",
                    str(input_len),
                    "--random-output-len",
                    str(output_len),
                    "--random-range-ratio",
                    "1.0",
                    "--num-prompts",
                    str(concurrency),
                    "--max-concurrency",
                    str(concurrency),
                    "--warmup-requests",
                    str(args.warmup_requests),
                    "--seed",
                    str(seed),
                    "--output-file",
                    str(output),
                ]
                case_log = log_dir / f"{args.backend}-{label}.log"
                started = time.monotonic()
                print(f"RUN {label}", flush=True)
                with case_log.open("w", encoding="utf-8") as log_handle:
                    completed = subprocess.run(
                        command,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                elapsed = time.monotonic() - started
                rows = read_jsonl(output)
                result = matching_result(rows, concurrency, input_len, output_len)
                record = {
                    "label": label,
                    "backend": args.backend,
                    "concurrency": concurrency,
                    "input_len": input_len,
                    "output_len": output_len,
                    "seed": seed,
                    "returncode": completed.returncode,
                    "wrapper_elapsed_s": round(elapsed, 3),
                    "log": str(case_log),
                    "result": result,
                }
                append_jsonl(summary, record)
                if completed.returncode != 0 or result is None:
                    failures.append(label)
                    tail = "\n".join(
                        case_log.read_text(encoding="utf-8").splitlines()[-40:]
                    )
                    print(f"FAIL {label}\n{tail}", flush=True)
                else:
                    print(
                        f"DONE {label} output_tok_s={result.get('output_throughput')} "
                        f"median_e2e_ms={result.get('median_e2e_latency_ms')}",
                        flush=True,
                    )

    if failures:
        print("FAILED_CASES " + ",".join(failures), file=sys.stderr)
        return 1
    print("ALL_CASES_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
