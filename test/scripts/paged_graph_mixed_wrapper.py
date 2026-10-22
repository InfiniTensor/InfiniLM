"""Correctness fixture with distinct request tokens, lengths and fenced logit readback."""

import ctypes
import json
from pathlib import Path
import runpy
import sys
import numpy as np
from infinilm import _core as core
from infinilm.lib import _infinilm

output = Path(sys.argv[sys.argv.index("--out") + 1])
cudart = ctypes.CDLL("libcudart.so")
cudart.cudaMemcpy.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
]


def tensor(values, dtype):
    return core.from_numpy(np.asarray(values, dtype=dtype))._underlying


def array(value):
    return core.Tensor(value).to_numpy()


old_forward = _infinilm.InferEngine.forward
saved = False


def checked(self, inp):
    global saved
    table = array(inp.block_tables)
    if (table[:, 0] < 0).any():
        return old_forward(self, inp)
    ids_shape = inp.input_ids.shape
    batch = table.shape[0]
    decode = ids_shape[1] == batch
    rows = table[:, 0] // 33
    if decode:
        positions = array(inp.position_ids).reshape(-1)
        q = np.maximum(1, positions - 17 * (rows % 3)).astype(np.int64)
        inp.input_ids = tensor((42 + rows * 31).reshape(1, batch), np.int64)
        inp.position_ids = tensor(q, np.int64)
        inp.past_sequence_lengths = tensor(q, np.int32)
        inp.total_sequence_lengths = tensor(q + 1, np.int32)
        inp.slot_mapping = tensor(table[:, 0].astype(np.int64) * 256 + q, np.int64)
    else:
        assert batch == 1
        row = int(rows[0])
        n = max(1, ids_shape[1] - 17 * (row % 3))
        inp.input_ids = tensor(
            (128 + (np.arange(n) * 13 + row * 17) % 1024).reshape(1, n), np.int64
        )
        inp.position_ids = tensor(np.arange(n), np.int64)
        inp.past_sequence_lengths = tensor([0], np.int32)
        inp.total_sequence_lengths = tensor([n], np.int32)
        inp.input_offsets = tensor([0, n], np.int32)
        inp.cu_seqlens = tensor([0, n], np.int32)
        inp.slot_mapping = tensor(int(table[0, 0]) * 256 + np.arange(n), np.int64)
    result = old_forward(self, inp)
    if decode and batch == 16 and not saved:
        scores = core.Tensor(result.logits)
        assert scores.is_contiguous()
        assert cudart.cudaSetDevice(scores.device.index) == 0
        # Producer completion is explicit; never execute the worker's Runtime
        # on this caller thread or change Tensor's ownership/copy contract.
        assert cudart.cudaDeviceSynchronize() == 0
        if scores.dtype == core.bfloat16:
            raw = np.empty(scores.numel(), dtype=np.uint16)
        elif scores.dtype == core.float16:
            raw = np.empty(scores.numel(), dtype=np.float16)
        else:
            assert scores.dtype == core.float32
            raw = np.empty(scores.numel(), dtype=np.float32)
        assert cudart.cudaMemcpy(raw.ctypes.data, scores.data_ptr(), raw.nbytes, 2) == 0
        if scores.dtype == core.bfloat16:
            values = (raw.astype(np.uint32) << 16).view(np.float32)
        else:
            values = raw.astype(np.float32)
        values = values.reshape(16, -1)
        assert np.isfinite(values).all() and not np.array_equal(values[0], values[1])
        np.save(output.with_suffix(".logits.npy"), values)
        output.with_suffix(".logits.json").write_text(
            json.dumps(
                dict(
                    shape=list(values.shape),
                    dtype=str(scores.dtype),
                    fixture="distinct row/position token pattern; length=max(1,L-17*(physical_row%3))",
                )
            )
        )
        saved = True
    return result


_infinilm.InferEngine.forward = checked
bench = Path(__file__).with_name("paged_graph_bench_final.py")
if not bench.exists():
    bench = Path(__file__).with_name("paged_graph_device_bench.py")
runpy.run_path(str(bench), run_name="__main__")
