#!/usr/bin/env python3
"""Exercise native tiny LFM2 full and cached forward paths.

This test loads the already-built InfiniCore and InfiniLM extension modules
directly.  It intentionally avoids the high-level Python package so it can
isolate C++ model/runtime behavior from optional Transformers dependencies.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--infinicore-extension", type=Path, required=True)
    parser.add_argument("--infinilm-extension", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "npu"), default="cpu")
    parser.add_argument("--attn-backend", default="default")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser.parse_args()


def load_extension(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot create import specification for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_to_numpy(core: Any, tensor: Any, np_dtype: Any) -> np.ndarray:
    # A GPU data_ptr cannot be dereferenced by ctypes.  Copy to host and finish
    # the device-to-host transfer before accessing the NumPy view.
    tensor = tensor.to(core.Device(core.Device.Type.CPU, 0))
    core.sync_device()
    tensor = tensor.contiguous()
    shape = tuple(tensor.shape)
    size = int(tensor.numel())
    c_type = np.ctypeslib.as_ctypes_type(np.dtype(np_dtype))
    buffer = (c_type * size).from_address(tensor.data_ptr())
    return np.ctypeslib.as_array(buffer).copy().reshape(shape)


def make_input(
    core: Any,
    extension: Any,
    token_ids: list[int],
    past_length: int,
    total_length: int,
) -> Any:
    sequence_length = len(token_ids)
    return extension.InferEngine.Input(
        core.from_list([token_ids], core.DataType.I64),
        position_ids=core.from_list(
            [list(range(past_length, past_length + sequence_length))],
            core.DataType.I64,
        ),
        past_sequence_lengths=core.from_list([past_length], core.DataType.I32),
        total_sequence_lengths=core.from_list([total_length], core.DataType.I32),
        input_offsets=core.from_list([0, sequence_length], core.DataType.I32),
        cu_seqlens=core.from_list([0, total_length], core.DataType.I32),
        sample_all_positions=True,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
    )


def main() -> None:
    args = parse_args()
    if args.device == "npu":
        # The CANN-generated launch stubs linked into InfiniCore expect the
        # Ascend runtime to be initialized before the extension is dlopen'ed.
        # Importing torch_npu performs that process-wide initialization.
        import torch
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            raise RuntimeError("Ascend NPU is not available")
    core = load_extension("_infinicore", args.infinicore_extension)
    extension = load_extension("_infinilm", args.infinilm_extension)
    device_type = {
        "cpu": core.Device.Type.CPU,
        "cuda": core.Device.Type.NVIDIA,
        "npu": core.Device.Type.ASCEND,
    }[args.device]
    with args.config.open("r", encoding="utf-8") as config_file:
        config_text = json.dumps(json.load(config_file))

    cache_config = extension.StaticKVCacheConfig(1, 32)
    engine = extension.InferEngine(
        config_text,
        extension.DistConfig(1),
        device_type,
        cache_config,
        False,
        args.attn_backend,
        None,
        False,
        "sync",
        False,
    )

    rng = np.random.default_rng(20260914)
    # from_blob does not own the NumPy storage.  Keep every source array alive
    # until load_params has copied/consumed the corresponding tensor.
    source_arrays: list[np.ndarray] = []
    parameters = {}
    for name, parameter in engine.state_dict()[0].items():
        shape = tuple(parameter.shape)
        if name.endswith("norm.weight"):
            array = np.ones(shape, dtype=np.float32)
        else:
            array = rng.normal(0.0, 0.02, size=shape).astype(np.float32)
        source_arrays.append(array)
        parameters[name] = core.from_blob(
            array.ctypes.data,
            list(shape),
            core.DataType.F32,
            core.Device(core.Device.Type.CPU, 0),
        )

    engine.load_params(parameters, True)
    engine.process_weights_after_loading()

    tokens = [1, 17, 23, 5, 91, 7]
    engine.reset_cache(cache_config)
    full = engine.forward(make_input(core, extension, tokens, 0, len(tokens)))
    full_logits = tensor_to_numpy(core, full.logits, np.float32)

    engine.reset_cache(cache_config)
    engine.forward(make_input(core, extension, tokens[:-1], 0, len(tokens) - 1))
    cached = engine.forward(
        make_input(core, extension, tokens[-1:], len(tokens) - 1, len(tokens))
    )
    cached_logits = tensor_to_numpy(core, cached.logits, np.float32)

    full_last = full_logits.reshape(-1, full_logits.shape[-1])[-1]
    cached_last = cached_logits.reshape(-1, cached_logits.shape[-1])[-1]
    max_abs_error = float(np.max(np.abs(full_last - cached_last)))
    result = {
        "device": args.device,
        "attention_backend": args.attn_backend,
        "nvidia_tf32_override": os.getenv("NVIDIA_TF32_OVERRIDE"),
        "parameter_count": len(parameters),
        "tokens": tokens,
        "full_logits_shape": list(full_logits.shape),
        "cached_logits_shape": list(cached_logits.shape),
        "full_last_argmax": int(np.argmax(full_last)),
        "cached_last_argmax": int(np.argmax(cached_last)),
        "full_vs_cached_max_abs_error": max_abs_error,
        "argmax_match": int(np.argmax(full_last)) == int(np.argmax(cached_last)),
        "atol": args.atol,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    if not np.isfinite(max_abs_error):
        raise AssertionError("native tiny LFM2 produced non-finite logits")
    if max_abs_error > args.atol:
        raise AssertionError(
            f"native full and cached logits diverged: max abs error {max_abs_error}"
        )


if __name__ == "__main__":
    main()
