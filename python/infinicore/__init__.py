import contextlib


def _arm_hygon_serialize_workaround():
    """Force AMD HIP runtime to serialize kernel launches and DMA copies on
    Hygon DCU (DTK 2604+) for tp>1 eager mode only.

    Background: when N ranks share one HIP context (InfiniLM's multi-thread tp
    model), each rank's first forward call triggers concurrent
    hsa_executable_freeze invocations. DTK's HSA loader uses BlitKernel-DMA
    inside Freeze, and the BLIT engine queue races across ranks — one rank's
    SubmitLinearCopyCommand wedges on hsa_signal_wait_scacquire and the rest
    queue forever. AMD_SERIALIZE_KERNEL=3 + AMD_SERIALIZE_COPY=3 serializes
    the BLIT engine work at the runtime layer.

    Trade-off: SERIALIZE_KERNEL=3 is INCOMPATIBLE with HIP graph capture —
    cudaDeviceSynchronize returns err 900 inside StreamBeginCapture. So when
    graph mode is in use we skip the auto-arm and rely on the per-op fence
    threaded through PagedCompiler → Graph::instantiate to lock-step JIT
    loads across ranks (same-kernel concurrent freezes don't trip the loader
    bug, only divergent ones do).

    Detection:
    - The dlsym shim presence gates Hygon-only behavior.
    - argv `--enable-graph` and `INFINILM_HYGON_NO_SERIALIZE=1` opt out.

    Must be set BEFORE any HIP/HSA library initializes — i.e. before
    `import torch` or `from infinicore.lib import _infinicore`.
    """
    import os
    import sys

    here = os.path.dirname(__file__)
    marker = os.path.join(os.path.dirname(here), "infinilm", "lib", "libflash_attn_hygon_dlsym.so")
    if not os.path.exists(marker):
        return
    if "--enable-graph" in sys.argv or os.environ.get("INFINILM_HYGON_NO_SERIALIZE") == "1":
        return
    os.environ.setdefault("AMD_SERIALIZE_KERNEL", "3")
    os.environ.setdefault("AMD_SERIALIZE_COPY", "3")


_arm_hygon_serialize_workaround()

with contextlib.suppress(ImportError):
    from ._preload import preload

    preload()

import infinicore.context as context
import infinicore.nn as nn

# Import context functions
from infinicore.context import (
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
from infinicore.device import device
from infinicore.device_event import DeviceEvent
from infinicore.dtype import (
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
from infinicore.ops.acos import acos
from infinicore.ops.add import add
from infinicore.ops.add_rms_norm import add_rms_norm
from infinicore.ops.addbmm import addbmm
from infinicore.ops.addcmul import addcmul
from infinicore.ops.addr import addr
from infinicore.ops.all import all
from infinicore.ops.argwhere import argwhere
from infinicore.ops.asin import asin
from infinicore.ops.asinh import asinh
from infinicore.ops.atanh import atanh
from infinicore.ops.attention import attention
from infinicore.ops.baddbmm import baddbmm
from infinicore.ops.bilinear import bilinear
from infinicore.ops.binary_cross_entropy_with_logits import (
    binary_cross_entropy_with_logits,
)
from infinicore.ops.bitwise_right_shift import bitwise_right_shift
from infinicore.ops.block_diag import block_diag
from infinicore.ops.broadcast_to import broadcast_to
from infinicore.ops.cat import cat
from infinicore.ops.cdist import cdist
from infinicore.ops.cross_entropy import cross_entropy
from infinicore.ops.diff import diff
from infinicore.ops.digamma import digamma
from infinicore.ops.dist import dist
from infinicore.ops.equal import equal
from infinicore.ops.flipud import flipud
from infinicore.ops.float_power import float_power
from infinicore.ops.floor import floor
from infinicore.ops.floor_divide import floor_divide
from infinicore.ops.fmin import fmin
from infinicore.ops.fmod import fmod
from infinicore.ops.hypot import hypot
from infinicore.ops.index_add import index_add
from infinicore.ops.index_copy import index_copy
from infinicore.ops.inner import inner
from infinicore.ops.kron import kron
from infinicore.ops.kthvalue import kthvalue
from infinicore.ops.kv_caching import kv_caching
from infinicore.ops.ldexp import ldexp
from infinicore.ops.lerp import lerp
from infinicore.ops.logaddexp import logaddexp
from infinicore.ops.logaddexp2 import logaddexp2
from infinicore.ops.logcumsumexp import logcumsumexp
from infinicore.ops.logdet import logdet
from infinicore.ops.logical_and import logical_and
from infinicore.ops.logical_not import logical_not
from infinicore.ops.masked_select import masked_select
from infinicore.ops.matmul import matmul
from infinicore.ops.mha_kvcache import mha_kvcache
from infinicore.ops.mha_varlen import mha_varlen
from infinicore.ops.mul import mul
from infinicore.ops.narrow import narrow
from infinicore.ops.paged_attention import paged_attention
from infinicore.ops.paged_attention_prefill import paged_attention_prefill
from infinicore.ops.paged_caching import paged_caching
from infinicore.ops.rearrange import rearrange
from infinicore.ops.reciprocal import reciprocal
from infinicore.ops.scatter import scatter
from infinicore.ops.sinh import sinh
from infinicore.ops.squeeze import squeeze
from infinicore.ops.sum import sum
from infinicore.ops.take import take
from infinicore.ops.tan import tan
from infinicore.ops.topk import topk
from infinicore.ops.unsqueeze import unsqueeze
from infinicore.ops.vander import vander
from infinicore.ops.var import var
from infinicore.ops.var_mean import var_mean
from infinicore.tensor import (
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

__all__ = [
    # Modules.
    "context",
    "nn",
    # Classes.
    "device",
    "DeviceEvent",
    "dtype",
    "Tensor",
    # Context functions.
    "get_device",
    "get_device_count",
    "get_stream",
    "set_device",
    "sync_device",
    "sync_stream",
    "is_graph_recording",
    "start_graph_recording",
    "stop_graph_recording",
    # Data Types.
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
    # Operations.
    "addcmul",
    "atanh",
    "binary_cross_entropy_with_logits",
    "cdist",
    "reciprocal",
    "add",
    "addr",
    "add_rms_norm",
    "argwhere",
    "asin",
    "acos",
    "addbmm",
    "floor",
    "attention",
    "block_diag",
    "kron",
    "bitwise_right_shift",
    "kv_caching",
    "asinh",
    "baddbmm",
    "bilinear",
    "fmod",
    "cat",
    "inner",
    "masked_select",
    "logaddexp",
    "logaddexp2",
    "matmul",
    "equal",
    "mul",
    "diff",
    "digamma",
    "dist",
    "logdet",
    "narrow",
    "ldexp",
    "lerp",
    "kthvalue",
    "squeeze",
    "unsqueeze",
    "rearrange",
    "cross_entropy",
    "tan",
    "empty",
    "empty_like",
    "from_blob",
    "from_list",
    "from_numpy",
    "from_torch",
    "mha_kvcache",
    "mha_varlen",
    "fmin",
    "floor_divide",
    "float_power",
    "flipud",
    "scatter",
    "logcumsumexp",
    "logical_not",
    "logical_and",
    "vander",
    "paged_caching",
    "paged_attention",
    "paged_attention_prefill",
    "hypot",
    "index_copy",
    "index_add",
    "take",
    "sinh",
    "ones",
    "broadcast_to",
    "strided_empty",
    "strided_from_blob",
    "zeros",
    "sum",
    "var_mean",
    "var",
    "topk",
    "all",
]

use_ntops = False

with contextlib.suppress(ImportError, ModuleNotFoundError):
    import sys

    import ntops

    for op_name in ntops.torch.__all__:
        getattr(ntops.torch, op_name).__globals__["torch"] = sys.modules[__name__]

    use_ntops = True
