import argparse
import os
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="examples/jiuge.py", help="Path to jiuge.py (default: examples/jiuge.py)")
    ap.add_argument("--out", default="/tmp/torchprof_jiuge.json", help="Chrome trace output path")
    ap.add_argument("--warmup", type=int, default=0, help="Warmup runs (not profiled)")
    ap.add_argument("--active", type=int, default=1, help="Profiled runs")
    ap.add_argument("jiuge_args", nargs=argparse.REMAINDER, help="Args forwarded to jiuge.py after '--'")
    args = ap.parse_args()

    # Ensure CUDA is ready before starting profiler.
    torch.cuda.synchronize()

    def run_once():
        import runpy

        # Run the target script as __main__ with its own argv.
        import sys

        old_argv = sys.argv
        try:
            forwarded = list(args.jiuge_args)
            if forwarded[:1] == ["--"]:
                forwarded = forwarded[1:]
            sys.argv = [args.script] + forwarded
            runpy.run_path(args.script, run_name="__main__")
        finally:
            sys.argv = old_argv

    for _ in range(args.warmup):
        run_once()
        torch.cuda.synchronize()

    # Profile "active" runs. Keep it small; jiuge loads weights so this is heavyweight.
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
        with_modules=False,
    ) as prof:
        t0 = time.time()
        for _ in range(args.active):
            run_once()
            torch.cuda.synchronize()
        t1 = time.time()
        try:
            prof.export_chrome_trace(args.out)
            print(f"wrote: {args.out}")
        except Exception as e:
            print(f"warn: failed to export chrome trace to {args.out}: {e}")

    print(f"elapsed_s: {t1 - t0:.3f}")
    # Print a short table of the hottest CUDA kernels/ops.
    print(
        prof.key_averages()
        .table(sort_by="self_cuda_time_total", row_limit=40)
    )


if __name__ == "__main__":
    main()

