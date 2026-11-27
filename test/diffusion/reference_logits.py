"""Save the reference implementation's logits along its block diffusion trajectory.

Each selected workload case writes ``<out>_<id>.npz`` with the logits of the
causal prefill, the first preview of an all-mask block, the preview after the
first unmask step, the causal replay of the resolved block and the first
preview of the next block. ``engine_logits.py`` replays the same trajectory.

The block loop follows ``CloverMoeModelLM.generate`` with neutral penalties.
The reference code is loaded from the checkpoint directory with
``trust_remote_code``.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def run_case(model, tokenizer, prompt, forced, shifts):
    """Run one prompt through the block loop and collect logits.

    `forced` is the record of another run. When given, the first unmask step and
    the resolved block are taken from it instead of this run's own decisions.
    """
    config = model.config
    block_size, mask_id = config.block_size, config.mask_token_id
    threshold = config.denoising_config["threshold"]
    device = model.model.embed_tokens.weight.device
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(prompt_ids, "keys"):
        prompt_ids = prompt_ids["input_ids"]
    if forced is not None and not np.array_equal(prompt_ids, forced["prompt_ids"]):
        raise ValueError("Reference trajectory was recorded with different prompt IDs.")
    ids = torch.tensor([prompt_ids], device=device)
    saved = {"prompt_ids": np.asarray(prompt_ids, dtype=np.int64)}

    def preview(block, cache):
        """Run one preview forward. The clean cache is returned unchanged."""
        return model(
            block,
            attention_mask=torch.ones_like(block, dtype=torch.bool),
            past_key_values=cache,
            use_cache=True,
            preview=True,
        )

    with torch.inference_mode():
        prefill = model(input_ids=ids, use_cache=True)
        cache = prefill.past_key_values
        saved["prompt_logits"] = prefill.logits[0].float().cpu().numpy()
        # RoPE only depends on relative positions, so a constant shift is an
        # equivalent input that rounds differently in low precision.
        for shift in shifts:
            positions = (torch.arange(ids.shape[1], device=device) + shift).view(1, -1)
            saved[f"prompt_logits_shift_{shift}"] = (
                model(input_ids=ids, position_ids=positions, use_cache=False)
                .logits[0]
                .float()
                .cpu()
                .numpy()
            )
        block = torch.full((1, block_size), mask_id, device=device, dtype=torch.long)
        pending = torch.ones_like(block[0], dtype=torch.bool)
        iterations = 0
        while bool(pending.any()):
            iterations += 1
            logits = preview(block, cache).logits[0].float()
            if iterations == 1:
                saved["preview_full_mask_logits"] = logits.cpu().numpy()
            elif iterations == 2:
                saved["preview_partial_logits"] = logits.cpu().numpy()
            raw = logits.clone()
            raw[:, mask_id] = float("-inf")
            confidence, prediction = raw.softmax(-1).max(-1)
            confidence = confidence.masked_fill(~pending, float("-inf"))
            transfer = (confidence > threshold) & pending
            if not bool(transfer.any()):
                transfer[confidence.argmax()] = True
            if forced is not None:
                if iterations == 1:
                    prediction = torch.from_numpy(forced["step1_prediction"]).to(device)
                    transfer = torch.from_numpy(
                        forced["step1_transfer"].astype(bool)
                    ).to(device)
                else:
                    prediction = torch.from_numpy(forced["resolved_block"]).to(device)
                    transfer = pending.clone()
            if iterations == 1:
                saved.update(
                    step1_prediction=prediction.cpu().numpy(),
                    step1_transfer=transfer.cpu().numpy(),
                )
            block[0, transfer] = prediction[transfer]
            pending[transfer] = False
        saved["resolved_block"] = block[0].cpu().numpy()
        hidden = model.model(block, past_key_values=cache, use_cache=True)
        saved["replay_logits"] = (
            model.lm_head(hidden.last_hidden_state)[0].float().cpu().numpy()
        )
        next_block = torch.full(
            (1, block_size), mask_id, device=device, dtype=torch.long
        )
        saved["next_preview_logits"] = (
            preview(next_block, hidden.past_key_values).logits[0].float().cpu().numpy()
        )
        saved["iterations"] = np.asarray(iterations, dtype=np.int64)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Checkpoint directory.")
    parser.add_argument(
        "--out", required=True, help="Output prefix; one `.npz` per case."
    )
    parser.add_argument("--workload", type=Path, default=WORKLOAD)
    parser.add_argument("--cases", type=parse_cases, default=parse_cases("0,1,2,3"))
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--shifts",
        type=parse_shifts,
        default=[],
        help="Also prefill with every token position shifted by each constant.",
    )
    parser.add_argument(
        "--trajectory",
        help="Prefix of a previous run whose first unmask step and resolved block are replayed.",
    )
    args = parser.parse_args()

    items = {item["id"]: item for item in json.loads(args.workload.read_text())}
    missing = [case for case in args.cases if case not in items]
    if missing:
        parser.error(f"The workload has no case {missing}.")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=getattr(torch, args.dtype),
        device_map="auto",
    ).eval()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for case in args.cases:
        forced = np.load(f"{args.trajectory}_{case}.npz") if args.trajectory else None
        start = time.perf_counter()
        saved = run_case(model, tokenizer, items[case]["prompt"], forced, args.shifts)
        np.savez(f"{args.out}_{case}.npz", **saved)
        print(
            f"case {case}: prompt_tokens={len(saved['prompt_ids'])} "
            f"iterations={int(saved['iterations'])} ({time.perf_counter() - start:.1f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
