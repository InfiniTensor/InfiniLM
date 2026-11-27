"""Replay reference trajectories with InfiniLM and save logits at the same points.

`--reference` names the prefix of the files written by `reference_logits.py`.
The first unmask step and the resolved block are read from them, so the engine
sees the same inputs as the reference.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from infinilm._core.utils import to_torch_dtype
from infinilm.diffusion.clover import CloverGenerator

WORKLOAD = Path(__file__).with_name("workload.json")


def parse_cases(value):
    """Return the workload ids given as a comma-separated list."""
    try:
        cases = sorted({int(item) for item in value.split(",")})
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Cases must be a comma-separated list of ids."
        ) from error
    if not cases or cases[0] < 0:
        raise argparse.ArgumentTypeError("Cases must be a comma-separated list of ids.")
    return cases


def parse_shifts(value):
    """Return the position shifts given as a comma-separated list."""
    try:
        return sorted({int(item) for item in value.split(",") if item})
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Shifts must be a comma-separated list of integers."
        ) from error


def forward(generator, ids, past, *, preview, layout, shift=0):
    """Call the engine like `CloverGenerator._forward`, keeping every position's logits."""
    length = len(ids)
    rows = list(range(past + shift, past + shift + length))
    positions = [rows] if layout == "batched" else rows

    def tensor(values, dtype):
        return generator.core.from_numpy(np.asarray(values, dtype=dtype))

    int64, int32 = np.int64, np.int32
    return generator.engine.forward_raw(
        tensor([ids], int64),
        position_ids=tensor(positions, int64),
        past_kv_lengths=tensor([past], int32),
        total_kv_lengths=tensor([past + length], int32),
        input_offsets=tensor([0, length], int32),
        cu_seqlens=tensor([0, past + length], int32),
        block_tables=tensor([[0]], int32),
        mamba_init_state_indices=tensor([0], int32),
        mamba_final_state_indices=tensor([0], int32),
        diffusion_preview=preview,
        diffusion_expert_budget=False,
        sample_all_positions=True,
        temperature=0,
        top_k=1,
    )


def logits_of(generator, result, tokens):
    """Copy the engine logits out before the next forward reuses the buffer."""
    value = result["logits"]
    # Copy between like-typed buffers: a BF16 model with an FP32 output head
    # yields FP32 logits, which a BF16 staging buffer would truncate.
    value_dtype = value.dtype
    out = torch.empty(
        value.shape,
        device=generator.torch_device,
        dtype=to_torch_dtype(value_dtype),
    )
    target = generator.core.from_blob(
        out.data_ptr(),
        list(value.shape),
        dtype=value_dtype,
        device=generator.device,
        owner=out,
    )
    target.copy_(value)
    generator.core.set_device(generator.device)
    generator.core.sync_device()
    return out.view(tokens, -1).float().cpu().numpy()


def replay_case(generator, reference, *, layout, shifts):
    """Replay one recorded trajectory and collect logits at the same points."""
    mask, width = generator.mask_id, generator.block_size
    prompt_ids = reference["prompt_ids"].tolist()
    past = len(prompt_ids)
    full_mask = [mask] * width
    transfer = reference["step1_transfer"].astype(bool)
    partial = list(full_mask)
    for position in np.nonzero(transfer)[0]:
        partial[int(position)] = int(reference["step1_prediction"][int(position)])
    resolved = reference["resolved_block"].tolist()

    def logits(ids, at, *, preview, shift=0):
        """Return the logits of one forward as a NumPy array."""
        result = forward(
            generator, ids, at, preview=preview, layout=layout, shift=shift
        )
        return logits_of(generator, result, len(ids))

    generator.engine.reset_cache(generator.cache_config)
    saved = {
        "prompt_ids": np.asarray(prompt_ids, dtype=np.int64),
        "step1_transfer": transfer,
        "resolved_block": np.asarray(resolved, dtype=np.int64),
        "prompt_logits": logits(prompt_ids, 0, preview=False),
        "preview_full_mask_logits": logits(full_mask, past, preview=True),
    }
    if "preview_partial_logits" in reference:
        saved["preview_partial_logits"] = logits(partial, past, preview=True)
    saved["replay_logits"] = logits(resolved, past, preview=False)
    saved["next_preview_logits"] = logits(full_mask, past + width, preview=True)
    # The engine cache is stateful, so each shifted prefill starts from a reset.
    for shift in shifts:
        generator.engine.reset_cache(generator.cache_config)
        saved[f"prompt_logits_shift_{shift}"] = logits(
            prompt_ids, 0, preview=False, shift=shift
        )
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Checkpoint directory.")
    parser.add_argument(
        "--reference", required=True, help="Prefix written by `reference_logits.py`."
    )
    parser.add_argument(
        "--out", required=True, help="Output prefix; one `.npz` per case."
    )
    parser.add_argument("--workload", type=Path, default=WORKLOAD)
    parser.add_argument("--cases", type=parse_cases, default=parse_cases("0,1,2,3"))
    parser.add_argument(
        "--device", required=True, help="InfiniLM device name or alias."
    )
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--max-context", type=int, default=2048)
    parser.add_argument(
        "--positions",
        choices=("packed", "batched"),
        default="packed",
        help="`batched` passes `[1, tokens]`, which `Mrope` reads as three axis rows.",
    )
    parser.add_argument(
        "--shifts",
        type=parse_shifts,
        default=[],
        help="Also prefill with every token position shifted by each constant.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Replay each case this many times and require identical logits.",
    )
    args = parser.parse_args()

    items = {item["id"]: item for item in json.loads(args.workload.read_text())}
    missing = [case for case in args.cases if case not in items]
    if missing:
        parser.error(f"The workload has no case {missing}.")
    missing = [
        case
        for case in args.cases
        if not Path(f"{args.reference}_{case}.npz").is_file()
    ]
    if missing:
        parser.error(f"No reference record for case {missing}.")
    generator = CloverGenerator(
        args.model, tp=args.tp, max_context=args.max_context, device=args.device
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for case in args.cases:
        reference = np.load(f"{args.reference}_{case}.npz")
        start = time.perf_counter()
        saved = replay_case(
            generator, reference, layout=args.positions, shifts=args.shifts
        )
        for _ in range(args.repeat - 1):
            again = replay_case(
                generator, reference, layout=args.positions, shifts=args.shifts
            )
            if not all(np.array_equal(saved[key], again[key]) for key in saved):
                raise SystemExit(
                    f"case {case}: repeated replays produced different logits."
                )
        np.savez(f"{args.out}_{case}.npz", **saved)
        # Compare the complete first block: with EOS honored, an engine
        # that emits EOS at the first position and a reference that starts
        # with a real token both truncate to an empty list and a broken
        # comparison would still report the first block as identical.
        result = generator.generate(
            items[case]["prompt"],
            max_new_tokens=generator.block_size,
            ignore_eos=True,
        )
        tokens = result.token_ids
        same = tokens == reference["resolved_block"][: len(tokens)].tolist() and (
            len(tokens) == generator.block_size
            if "resolved_block" in reference
            and reference["resolved_block"] is not None
            else True
        )
        print(
            f"case {case}: replay {time.perf_counter() - start:.1f}s | "
            f"first block identical to the reference: {same} | "
            f"iterations={result.iterations} text={result.text!r}",
            flush=True,
        )


if __name__ == "__main__":
    main()
