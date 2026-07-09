#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from statistics import mean


TOTAL_TIME_PATTERNS = [
    re.compile(r"total_time:\s*([0-9.]+)\s*ms", re.IGNORECASE),
    re.compile(r"Generation completed in\s*([0-9.]+)\s*ms", re.IGNORECASE),
]

THROUGHPUT_PATTERNS = [
    re.compile(r"decode/output throughput:\s*([0-9.]+)\s*tok/s", re.IGNORECASE),
    re.compile(r"Decode\s+Avg\s+ITL:\s*[0-9.]+\s*ms\s+Throughput:\s*([0-9.]+)\s*tok/s", re.IGNORECASE),
    re.compile(r"Throughput:\s*([0-9.]+)\s*tok/s", re.IGNORECASE),
]

CASE_PATTERNS = [
    re.compile(r"case=([^\s]+)"),
    re.compile(r"Batchsize=(\d+)\s+Per_Batch_Input_Len=(\d+)\s+Per_Batch_New_Tokens=(\d+)", re.IGNORECASE),
    re.compile(r"batch_size=(\d+)\s+input_len=(\d+)\s+output_len=(\d+)", re.IGNORECASE),
]

FINISH_STATUS_PATTERN = re.compile(r"\[finish\].*?status=([0-9]+)")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def first_pattern_values(patterns, text):
    for pattern in patterns:
        values = [float(match.group(1)) for match in pattern.finditer(text)]
        if values:
            return values
    return []


def infer_case(path: Path, text: str) -> str:
    for pattern in CASE_PATTERNS:
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        match = matches[-1]
        if len(match.groups()) == 1:
            return match.group(1)
        return f"bs{match.group(1)}_{match.group(2)}_{match.group(3)}"

    stem = path.stem
    for prefix in ("torch_bench_dsv2lc_", "infinilm_bench_dsv2lc_", "pytorch_bench_dsv2lc_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def parse_log(path: Path):
    text = read_text(path)
    total_ms_values = first_pattern_values(TOTAL_TIME_PATTERNS, text)
    throughput_values = first_pattern_values(THROUGHPUT_PATTERNS, text)
    finish_statuses = [int(match.group(1)) for match in FINISH_STATUS_PATTERN.finditer(text)]

    status = "ok" if total_ms_values else "missing_metrics"
    has_error_text = (
        "Traceback (most recent call last)" in text
        or "out of memory" in text.lower()
        or "cudamalloc" in text.lower()
    )
    if any(code != 0 for code in finish_statuses) or has_error_text:
        status = "failed" if not total_ms_values else "ok_with_errors"

    return {
        "path": path,
        "case": infer_case(path, text),
        "total_ms_values": total_ms_values,
        "throughput_values": throughput_values,
        "total_ms": mean(total_ms_values) if total_ms_values else None,
        "throughput": mean(throughput_values) if throughput_values else None,
        "runs": len(total_ms_values),
        "status": status,
    }


def fmt_time(ms):
    if ms is None:
        return "NA"
    seconds = ms / 1000.0
    if seconds < 60:
        return f"{seconds:.2f}s"
    return f"{seconds / 60:.2f}min"


def fmt_num(value):
    return "NA" if value is None else f"{value:.2f}"


def fmt_samples(values):
    if not values:
        return "NA"
    return ", ".join(fmt_time(value) for value in values)


def main():
    parser = argparse.ArgumentParser(
        description="Compare PyTorch and InfiniLM benchmark logs. Multiple runs in one log are averaged before speedup is computed."
    )
    parser.add_argument("--inputs", nargs="+", required=True, help="benchmark log files, usually torch log then infinilm log")
    args = parser.parse_args()

    results = [parse_log(Path(path)) for path in args.inputs]

    print("| file | case | status | runs | avg_total_time | avg_throughput tok/s | total_time_samples |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for item in results:
        print(
            f"| {item['path'].name} | {item['case']} | {item['status']} | {item['runs']} | "
            f"{fmt_time(item['total_ms'])} | {fmt_num(item['throughput'])} | {fmt_samples(item['total_ms_values'])} |"
        )

    if len(results) >= 2:
        base, candidate = results[0], results[1]
        print()
        print(f"baseline={base['path'].name}")
        print(f"candidate={candidate['path'].name}")

        if base["total_ms"] and candidate["total_ms"]:
            speedup = base["total_ms"] / candidate["total_ms"]
            reduction = (1.0 - candidate["total_ms"] / base["total_ms"]) * 100.0
            print(f"time_speedup={speedup:.3f}x")
            print(f"time_reduction={reduction:.2f}%")

        if base["throughput"] and candidate["throughput"]:
            speedup = candidate["throughput"] / base["throughput"]
            print(f"throughput_speedup={speedup:.3f}x")


if __name__ == "__main__":
    main()

