import contextlib
import dataclasses
from collections.abc import Callable

import torch


@dataclasses.dataclass
class Payload:
    func: Callable

    ref: Callable

    args: tuple

    kwargs: dict

    rtol: float = 1e-5

    atol: float = 1e-8


def get_available_devices():
    devices = ["cpu"]

    if torch.cuda.is_available():
        devices.append("cuda")

    if hasattr(torch, "mlu") and torch.mlu.is_available():
        devices.append("mlu")

    if hasattr(torch, "musa") and torch.musa.is_available():
        devices.append("musa")

    if hasattr(torch, "npu") and torch.npu.is_available():
        devices.append("npu")

    return tuple(devices)


with contextlib.suppress(ImportError, ModuleNotFoundError):
    import torch_mlu  # noqa: F401

with contextlib.suppress(ImportError, ModuleNotFoundError):
    import torch_npu  # noqa: F401


def empty_strided(shape, strides, *, dtype=None, device=None):
    if strides is None:
        return torch.empty(shape, dtype=dtype, device=device)

    return torch.empty_strided(shape, strides, dtype=dtype, device=device)


def randn_strided(shape, strides, *, dtype=None, device=None):
    output = empty_strided(shape, strides, dtype=dtype, device=device)

    output.as_strided(
        (output.untyped_storage().size() // output.element_size(),), (1,)
    ).normal_()

    return output


def rand_strided(shape, strides, *, dtype=None, device=None):
    output = empty_strided(shape, strides, dtype=dtype, device=device)

    output.as_strided(
        (output.untyped_storage().size() // output.element_size(),), (1,)
    ).uniform_(0, 1)

    return output


def randint_strided(low, high, shape, strides, *, dtype=None, device=None):
    output = empty_strided(shape, strides, dtype=dtype, device=device)

    flat = output.as_strided(
        (output.untyped_storage().size() // output.element_size(),), (1,)
    )

    try:
        flat.random_(low, high)
    except RuntimeError as exc:
        if "random_" not in str(exc) or "not implemented" not in str(exc):
            raise

        values = torch.randint(low, high, flat.shape, dtype=torch.int64).to(
            dtype=output.dtype,
            device=output.device,
        )
        flat.copy_(values)

    return output


_STREAM_ACCESSORS = {
    "npu": ("npu", "npu_stream"),
    "cuda": ("cuda", "cuda_stream"),
    "mlu": ("mlu", "mlu_stream"),
    "musa": ("musa", "musa_stream"),
}


def get_stream(device):
    """Return the raw stream handle for `device`, or 0 for CPU.

    Uses the device-specific `torch.<dev>.current_stream()` API rather than
    `torch.accelerator.current_stream()` — the latter returns a different
    stream object on torch 2.9 + vllm-ascend, producing cross-stream data
    hazards on cached-executor ops.
    """
    if isinstance(device, torch.device):
        device = device.type

    if isinstance(device, str) and ":" in device:
        device = device.split(":")[0]

    if device == "cpu":
        return 0

    mod_name, attr = _STREAM_ACCESSORS.get(device, (None, None))

    if mod_name is None:
        return 0

    mod = getattr(torch, mod_name, None)

    if mod is None:
        return 0

    stream = mod.current_stream()

    return getattr(stream, attr, 0)


def clone_strided(input):
    output = empty_strided(
        input.size(), input.stride(), dtype=input.dtype, device=input.device
    )

    as_strided_args = (output.untyped_storage().size() // output.element_size(),), (1,)

    output.as_strided(*as_strided_args).copy_(input.as_strided(*as_strided_args))

    return output
