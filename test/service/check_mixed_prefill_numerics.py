"""Run both split settings in separate processes, then compare saved arrays.

Uses native InfiniLM forward only. A four-layer checkpoint is sufficient.
The sequence covers mixed prefill, continuation, decode and state-row swaps.
"""

import argparse
import ctypes
from pathlib import Path
import numpy as np
import torch
import infinicore as ic
from infinilm.infer_engine import InferEngine
from infinilm.cache import PagedKVCacheConfig
import infinilm.modeling_utils as weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output")
    args = parser.parse_args()
    torch.set_num_threads(4)
    ic.set_device(ic.device("cuda", 0))
    # Fixture config truncates the model; check every required weight.
    check = weights.check_parameters
    weights.check_parameters = lambda keys, loaded: check(
        keys, list(set(keys) & set(loaded))
    )
    engine = InferEngine(
        args.model,
        device=ic.device("cuda", 0),
        cache_config=PagedKVCacheConfig(
            num_blocks=128, block_size=64, max_batch_size=8
        ),
        enable_graph_compiling=True,
        attention_backend="flash-attn",
        weight_load_mode="sync",
        dtype="bfloat16",
    )
    weights.load_model_state_dict_by_file(engine, args.model, engine.dtype)
    arrays = {}

    def read(tensor):
        tensor = ic.Tensor(tensor).to(ic.device("cpu"))
        raw = ctypes.string_at(tensor.data_ptr(), tensor.numel() * 2)
        return (
            torch.from_numpy(np.frombuffer(raw, dtype=np.uint16).copy())
            .view(torch.bfloat16)
            .float()
            .numpy()
            .reshape(tuple(tensor.shape))
        )

    def run(name, rows, lengths, past, initial=None, final=None):
        offsets = np.cumsum([0] + lengths).tolist()
        positions = [
            i
            for start, length in zip(past, lengths)
            for i in range(start, start + length)
        ]
        ids = [
            (row * 971 + i * 7) % 10000 + 100
            for row, start, length in zip(rows, past, lengths)
            for i in range(start, start + length)
        ]
        total = [a + b for a, b in zip(past, lengths)]

        def tensor(value, dtype):
            return ic.from_list(value, dtype=dtype)

        data = dict(
            input_ids=tensor([ids], ic.int64),
            position_ids=tensor(
                (
                    positions
                    if engine.position_id_axes == 1
                    else [positions] * engine.position_id_axes
                ),
                ic.int64,
            ),
            past_kv_lengths=tensor(past, ic.int32),
            total_kv_lengths=tensor(total, ic.int32),
            input_offsets=tensor(offsets, ic.int32),
            cu_seqlens=tensor(np.cumsum([0] + total).tolist(), ic.int32),
            block_tables=tensor(
                [list(range(r * 8, (r + 1) * 8)) for r in rows], ic.int32
            ),
            slot_mapping=tensor(
                [
                    r * 8 * 64 + i
                    for r, start, length in zip(rows, past, lengths)
                    for i in range(start, start + length)
                ],
                ic.int64,
            ),
            sample_all_positions=False,
            temperature=0,
            top_k=1,
            top_p=1,
        )
        if engine.has_mamba_cache:
            data.update(
                mamba_init_state_indices=tensor(initial, ic.int32),
                mamba_final_state_indices=tensor(final, ic.int32),
            )
        output = super(InferEngine, engine).forward(engine._build_input(**data))
        arrays[name + "_logits"] = read(output.logits)
        arrays[name + "_ids"] = ic.Tensor(output.output_ids).to_numpy().copy()
        assert np.isfinite(arrays[name + "_logits"]).all()
        print("PASS_FORWARD", name, rows, lengths, flush=True)

    for iteration in range(2):
        tag = str(iteration)
        run(tag + "_seed", [0, 1, 2], [31, 47, 63], [0, 0, 0], [0, 0, 0], [1, 3, 5])
        run(
            tag + "_mixed",
            [3, 0, 1, 2],
            [257, 1, 1, 1],
            [0, 31, 47, 63],
            [0, 1, 3, 5],
            [7, 1, 3, 5],
        )
        run(
            tag + "_decode",
            [3, 0, 1, 2],
            [1] * 4,
            [257, 32, 48, 64],
            [7, 1, 3, 5],
            [7, 1, 3, 5],
        )
        run(
            tag + "_continuation",
            [3, 0, 1, 2],
            [129, 1, 1, 1],
            [258, 33, 49, 65],
            [7, 1, 3, 5],
            [7, 1, 3, 5],
        )
        run(
            tag + "_swap",
            [3, 0, 1, 2],
            [17, 1, 1, 1],
            [387, 34, 50, 66],
            [7, 1, 3, 5],
            [1, 7, 3, 5],
        )
        run(
            tag + "_after_swap",
            [3, 0, 1, 2],
            [1] * 4,
            [404, 35, 51, 67],
            [1, 7, 3, 5],
            [1, 7, 3, 5],
        )
    np.savez_compressed(args.output, **arrays)
    print("NATIVE_MIXED_NUMERICS_SAVED", args.output, flush=True)


if __name__ == "__main__":
    main()
