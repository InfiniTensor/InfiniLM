#!/usr/bin/env python3
"""InfiniLM DeepSeek V4 gate-routing unit tests.

This script mirrors the current C++ DeepseekV4Gate routing behavior with tiny
fixed tensors, and verifies that gate weights preserve their checkpoint dtypes
after safetensors load/readback:

* ffn.gate.tid2eid: I64
* ffn.gate.weight: BF16
* ffn.gate.bias: F32
"""

from __future__ import annotations

import argparse

from _deepseek_v4_gate_routing_common import (
    inspect_checkpoint_gate_dtypes,
    run_reference_gate_tests,
)


def test_infinilm_deepseek_v4_gate_routing_reference() -> None:
    run_reference_gate_tests("infinilm")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        help="Optional DeepSeek V4 checkpoint directory to inspect real gate dtypes.",
    )
    args = parser.parse_args()

    run_reference_gate_tests("infinilm")
    if args.checkpoint_dir:
        inspect_checkpoint_gate_dtypes(args.checkpoint_dir)


if __name__ == "__main__":
    main()
