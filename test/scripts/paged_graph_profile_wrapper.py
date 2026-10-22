"""Capture the third fixed-shape decode using CUDA Profiler API (no NVTX dependency)."""

import ctypes
import os
from pathlib import Path
import runpy

os.environ["INFINICORE_GRAPH_DEBUG"] = "1"
os.environ["INFINILM_REQUIRE_DEVICE_GRAPH"] = "1"
from infinilm.lib import _infinilm

cudart = ctypes.CDLL("libcudart.so")
forward = _infinilm.InferEngine.forward
count = 0


def profiled(self, inp):
    global count
    ids, table = inp.input_ids, inp.block_tables
    decode = ids is not None and table is not None and ids.shape[1] == table.shape[0]
    if decode:
        count += 1
    capture = decode and count == 3
    if capture:
        assert cudart.cudaProfilerStart() == 0
    try:
        return forward(self, inp)
    finally:
        if capture:
            assert cudart.cudaProfilerStop() == 0


_infinilm.InferEngine.forward = profiled
bench = Path(__file__).with_name("paged_graph_device_bench.py")
if not bench.exists():
    bench = Path(__file__).with_name("paged_graph_bench_ext.py")
runpy.run_path(str(bench), run_name="__main__")
