#!/usr/bin/env python3
"""Summarize `vllm bench serve` result files into a throughput table.

Parses the `bsX_inY_outZ.txt` files produced by run_bench.sh and prints one row
per (concurrency, input, output) config. Pass one result dir to just tabulate,
or two dirs (base first, then this) to get a base-vs-this comparison with the
output/total throughput deltas that the 赛题 scores on.

Usage:
    python3 summarize.py bench_results/this
    python3 summarize.py bench_results/base bench_results/this
"""

import glob
import os
import re
import sys

# Label -> canonical metric. vllm 各版本标签略有差异，用宽松正则匹配数值。
_METRICS = {
    "req_s": r"Request throughput \(req/s\):\s*([\d.]+)",
    "out_tok_s": r"Output token throughput \(tok/s\):\s*([\d.]+)",
    "total_tok_s": r"Total Token throughput \(tok/s\):\s*([\d.]+)",
    "ttft_ms": r"Mean TTFT \(ms\):\s*([\d.]+)",
    "tpot_ms": r"Mean TPOT \(ms\):\s*([\d.]+)",
}
_FNAME = re.compile(r"bs(\d+)_in(\d+)_out(\d+)\.txt$")


def parse_file(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    row = {}
    for key, pat in _METRICS.items():
        m = re.search(pat, text)
        row[key] = float(m.group(1)) if m else None
    return row


def load_dir(d):
    """Return {(bs, in, out): metrics} for one result dir."""
    out = {}
    for path in sorted(glob.glob(os.path.join(d, "bs*_in*_out*.txt"))):
        m = _FNAME.search(os.path.basename(path))
        if not m:
            continue
        cfg = tuple(int(x) for x in m.groups())
        out[cfg] = parse_file(path)
    return out


def fmt(v, width=10):
    return ("%.2f" % v).rjust(width) if isinstance(v, float) else "-".rjust(width)


def print_single(d):
    data = load_dir(d)
    if not data:
        print(f"(无结果文件: {d})")
        return
    print(f"\n=== {d} ===")
    hdr = ["bs", "in", "out", "out_tok/s", "total_tok/s", "req/s", "TTFT_ms", "TPOT_ms"]
    print("  ".join(h.rjust(10) for h in hdr))
    for cfg in sorted(data):
        r = data[cfg]
        cols = [
            str(cfg[0]),
            str(cfg[1]),
            str(cfg[2]),
            r["out_tok_s"],
            r["total_tok_s"],
            r["req_s"],
            r["ttft_ms"],
            r["tpot_ms"],
        ]
        print("  ".join(c.rjust(10) if isinstance(c, str) else fmt(c) for c in cols))


def print_compare(base_dir, this_dir):
    base, this = load_dir(base_dir), load_dir(this_dir)
    keys = sorted(set(base) | set(this))
    if not keys:
        print("(两目录都无结果)")
        return
    print(f"\n=== base={base_dir}  vs  this={this_dir} ===")
    print("列: out_tok/s (base -> this, Δ%) | total_tok/s (base -> this, Δ%)")
    hdr = [
        "bs",
        "in",
        "out",
        "out_base",
        "out_this",
        "out_Δ%",
        "tot_base",
        "tot_this",
        "tot_Δ%",
    ]
    print("  ".join(h.rjust(10) for h in hdr))
    for cfg in keys:
        b, t = base.get(cfg, {}), this.get(cfg, {})

        def delta(bv, tv):
            if isinstance(bv, float) and isinstance(tv, float) and bv > 0:
                return (tv - bv) / bv * 100.0
            return None

        cols = [
            str(cfg[0]),
            str(cfg[1]),
            str(cfg[2]),
            b.get("out_tok_s"),
            t.get("out_tok_s"),
            delta(b.get("out_tok_s"), t.get("out_tok_s")),
            b.get("total_tok_s"),
            t.get("total_tok_s"),
            delta(b.get("total_tok_s"), t.get("total_tok_s")),
        ]
        print("  ".join(c.rjust(10) if isinstance(c, str) else fmt(c) for c in cols))


def main():
    args = sys.argv[1:]
    if len(args) == 1:
        print_single(args[0])
    elif len(args) == 2:
        print_compare(args[0], args[1])
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
