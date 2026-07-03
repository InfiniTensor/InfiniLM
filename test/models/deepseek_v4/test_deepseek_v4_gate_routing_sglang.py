#!/usr/bin/env python3
"""SGLang DeepSeek V4 gate-routing unit tests.

The production SGLang loader remaps HF gate names before loading parameters.
This test validates the same two routing modes as the InfiniLM test, then checks
that remapped gate tensors preserve the expected dtypes and values.
"""

from __future__ import annotations

import argparse
from typing import Dict

import torch

from _deepseek_v4_gate_routing_common import (
    EXPECTED_HF_DTYPES,
    assert_close,
    inspect_checkpoint_gate_dtypes,
    roundtrip_hf_gate_state_dict,
    run_reference_gate_tests,
)


EXPECTED_SGLANG_DTYPES = {
    "model.layers.0.mlp.topk.tid2eid": torch.int64,
    "model.layers.0.mlp.gate.weight": torch.bfloat16,
    "model.layers.3.mlp.gate.weight": torch.bfloat16,
    "model.layers.3.mlp.gate.e_score_correction_bias": torch.float32,
}


def sglang_remap_gate_name(name: str) -> str:
    if name.startswith("layers."):
        name = "model." + name
    name = name.replace(".ffn.", ".mlp.")
    name = name.replace(".gate.tid2eid", ".topk.tid2eid")
    name = name.replace(".gate.bias", ".gate.e_score_correction_bias")
    return name


def validate_sglang_remapped_gate_state() -> Dict[str, torch.Tensor]:
    hf_loaded = roundtrip_hf_gate_state_dict()
    remapped = {sglang_remap_gate_name(k): v for k, v in hf_loaded.items()}
    for hf_name, hf_dtype in EXPECTED_HF_DTYPES.items():
        mapped_name = sglang_remap_gate_name(hf_name)
        if mapped_name not in remapped:
            raise AssertionError(f"missing remapped tensor: {mapped_name}")
        got = remapped[mapped_name]
        expected_dtype = EXPECTED_SGLANG_DTYPES[mapped_name]
        if got.dtype != expected_dtype:
            raise AssertionError(
                f"{mapped_name} dtype mismatch: got={got.dtype}, expected={expected_dtype}"
            )
        if got.dtype != hf_dtype:
            raise AssertionError(
                f"{mapped_name} changed dtype during remap: got={got.dtype}, hf={hf_dtype}"
            )
        assert_close(mapped_name, got, hf_loaded[hf_name])
    return remapped


def test_sglang_deepseek_v4_gate_routing_reference() -> None:
    run_reference_gate_tests("sglang")
    validate_sglang_remapped_gate_state()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        help="Optional DeepSeek V4 checkpoint directory to inspect real gate dtypes.",
    )
    args = parser.parse_args()

    run_reference_gate_tests("sglang")
    remapped = validate_sglang_remapped_gate_state()
    print(f"sglang: remapped gate keys={sorted(remapped)}")
    if args.checkpoint_dir:
        inspect_checkpoint_gate_dtypes(args.checkpoint_dir)


if __name__ == "__main__":
    main()
