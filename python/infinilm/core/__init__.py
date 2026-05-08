import contextlib


def _arm_hygon_serialize_workaround():
    import os
    import sys

    here = os.path.dirname(__file__)
    marker = os.path.join(os.path.dirname(here), "lib", "libflash_attn_hygon_dlsym.so")
    if not os.path.exists(marker):
        return
    if (
        "--enable-graph" in sys.argv
        or os.environ.get("INFINILM_HYGON_NO_SERIALIZE") == "1"
    ):
        return
    os.environ.setdefault("AMD_SERIALIZE_KERNEL", "3")
    os.environ.setdefault("AMD_SERIALIZE_COPY", "3")


_arm_hygon_serialize_workaround()

with contextlib.suppress(ImportError):
    from ._preload import preload

    preload()

from .device import device
from .dtype import (
    bfloat16,
    bool,
    cdouble,
    cfloat,
    chalf,
    complex32,
    complex64,
    complex128,
    double,
    dtype,
    float,
    float16,
    float32,
    float64,
    half,
    int,
    int8,
    int16,
    int32,
    int64,
    long,
    short,
    uint8,
)
from .tensor import (
    Tensor,
    empty,
    empty_like,
    from_blob,
    from_list,
    from_numpy,
    from_torch,
    ones,
    strided_empty,
    strided_from_blob,
    zeros,
)
from . import context, nn, utils
from .context import (
    get_device,
    get_device_count,
    get_stream,
    is_graph_recording,
    set_device,
    start_graph_recording,
    stop_graph_recording,
    sync_device,
    sync_stream,
)
from .device_event import DeviceEvent

__all__ = [
    "context",
    "nn",
    "utils",
    "device",
    "DeviceEvent",
    "dtype",
    "Tensor",
    "get_device",
    "get_device_count",
    "get_stream",
    "set_device",
    "sync_device",
    "sync_stream",
    "is_graph_recording",
    "start_graph_recording",
    "stop_graph_recording",
    "bfloat16",
    "bool",
    "cdouble",
    "cfloat",
    "chalf",
    "complex32",
    "complex64",
    "complex128",
    "double",
    "float",
    "float16",
    "float32",
    "float64",
    "half",
    "int",
    "int8",
    "int16",
    "int32",
    "int64",
    "long",
    "short",
    "uint8",
    "empty",
    "empty_like",
    "from_blob",
    "from_list",
    "from_numpy",
    "from_torch",
    "ones",
    "strided_empty",
    "strided_from_blob",
    "zeros",
]
