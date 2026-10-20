"""Unified test for every operator emitted by `generate_torch_ops.py`.

The generator writes `generated/torch_ops_metadata.json` listing every op
with full per-parameter info (`name`, `type`, `is_tensor`, `is_out`).
A single parametrized test reads that metadata, builds inputs from the
parameter list, calls the InfiniOps wrapper and the torch reference, and
compares each output tensor.  Adding an op to `scripts/torch_ops.yaml`
extends coverage with no test changes.
"""

import json
import pathlib
import re

import infini.ops
import pytest
import torch

from tests.utils import clone_strided, randn_strided

# PyTorch backends are emitted at this slot — see `_PYTORCH_SLOT` in
# `scripts/generate_torch_ops.py`.
_PYTORCH_SLOT = 8

_INSTALLED_METADATA_PATH = (
    pathlib.Path(infini.ops.__file__).resolve().with_name("torch_ops_metadata.json")
)
_SOURCE_METADATA_PATH = (
    pathlib.Path(__file__).resolve().parent.parent
    / "generated"
    / "torch_ops_metadata.json"
)

_METADATA_PATH = next(
    (
        path
        for path in (_INSTALLED_METADATA_PATH, _SOURCE_METADATA_PATH)
        if path.exists()
    ),
    _SOURCE_METADATA_PATH,
)
_METADATA = (
    json.loads(_METADATA_PATH.read_text()) if _METADATA_PATH.exists() else {"ops": []}
)

_SHAPES = (
    (13, 4),
    (13, 4, 4),
    (4, 4, 5632),
)

_DTYPES = (
    (torch.float32, 1e-5, 1e-5),
    (torch.float16, 1e-2, 1e-2),
    (torch.bfloat16, 1e-2, 1e-2),
)

# Op-specific input shapes for matrix ops (`mm` etc.) which cannot use
# `randn_strided(shape)` for both inputs.  The tuple is one shape per
# tensor input, in YAML order.
_TENSOR_SHAPES = {
    "mm": ((8, 16), (16, 12)),
    "bmm": ((4, 8, 16), (4, 16, 12)),
    "matmul": ((8, 16), (16, 12)),
    "dot": ((16,), (16,)),
    "vdot": ((16,), (16,)),
    "mv": ((8, 16), (16,)),
    "inner": ((8, 16), (8, 16)),
    "outer": ((8,), (12,)),
    "ger": ((8,), (12,)),
    "kron": ((3, 4), (2, 3)),
    "internal_compute_linear_combination": ((3, 2, 2), (5, 3)),
}

# Per-(op, param-name) values for non-tensor inputs.  Lookup falls back
# to a type-based default if no entry exists.
_SCALAR_VALUES = {
    ("clamp_min", "min"): -0.5,
    ("clamp_max", "max"): 0.5,
    ("leaky_relu", "negative_slope"): 0.01,
    ("hardshrink", "lambd"): 0.5,
    ("softshrink", "lambd"): 0.5,
    ("mvlgamma", "p"): 2,
    ("prod", "dim"): 0,
    ("cumsum", "dim"): 0,
    ("cumprod", "dim"): 0,
    ("logcumsumexp", "dim"): 0,
    ("cummax", "dim"): 0,
    ("cummin", "dim"): 0,
    ("softmax", "dim"): -1,
    ("log_softmax", "dim"): -1,
    ("threshold", "threshold"): 0.0,
    ("threshold", "value"): 0.0,
    ("hardtanh", "min_val"): -1.0,
    ("hardtanh", "max_val"): 1.0,
    ("softplus", "beta"): 1.0,
    ("softplus", "threshold"): 20.0,
    ("elu", "alpha"): 1.0,
    ("elu", "scale"): 1.25,
    ("elu", "input_scale"): 0.75,
    ("sub", "alpha"): 1.0,
    ("addcmul", "value"): 1.0,
    ("addcdiv", "value"): 1.0,
    # `str reduce` modes accepted by the corresponding ATen kernels.
    ("index_reduce", "reduce"): "amax",
    ("scatter_reduce", "reduce"): "amax",
    ("scatter_reduce_two", "reduce"): "amax",
    # `int dim` for ops where 0 is a safe choice for our test shapes.
    ("kthvalue_values", "k"): 1,
    ("kthvalue_values", "dim"): 0,
    ("mode_values", "dim"): 0,
}

_TYPE_DEFAULTS = {"int": 0, "SymInt": 0, "bool": False, "str": "none"}

# Mirrors `kStringToDataType` in `src/data_type.h`.  Any tensor passed to
# an InfiniOps op must have one of these dtypes; others (complex,
# quantised types) abort the process inside `DataTypeFromString`.  Some
# vendor torch forks lag behind upstream and lack `uint16` / `uint32` /
# `uint64` (added in PyTorch 2.3); resolve them lazily and keep the
# attributes that actually exist.
_SUPPORTED_DTYPE_NAMES = (
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "bool",
    "float16",
    "bfloat16",
    "float32",
    "float64",
)
_SUPPORTED_DTYPES = frozenset(
    getattr(torch, name) for name in _SUPPORTED_DTYPE_NAMES if hasattr(torch, name)
)


_LIST_SIZE_RE = re.compile(r"\[(\d+)\]")


def _optional_inner_type(aten_type):
    if aten_type.endswith("?"):
        return aten_type[:-1]

    return None


def _is_inplace_aten_name(name):
    """Return whether `name` is an ATen in-place operator name."""

    return name.endswith("_") and not name.endswith("__")


def _list_default(aten_type):
    """Default value for a required `int[N]` / `SymInt[N]` param.  Most
    such params name a `dim` or `kernel_size`; `[0]` works for `dim` and
    causes `kernel_size`-style ops to fail their reference call cleanly,
    which the test then skips."""
    size_match = _LIST_SIZE_RE.search(aten_type)
    n = int(size_match.group(1)) if size_match else 1

    return [0] * n


# Errors emitted by upstream PyTorch and vendor-forked variants for
# unsupported (op, dtype, device) combinations.  We skip rather than fail
# on these — the gap is in PyTorch, not InfiniOps.
_VENDOR_SKIP_PATTERNS = (
    "not implemented for",  # upstream PyTorch
    "CNNL_STATUS_BAD_PARAM",  # `torch_mlu` (Cambricon)
    "MUDNN failed",  # `torch_musa` (Moore)
    "Could not run",  # missing dispatcher entry on this backend
    "don't support tensor dtype",  # `torch_mlu` dtype check
    "unknown format type",  # `torch_npu` format descriptor gap
    "result requires dtype",  # output dtype mismatch (e.g. `float_power`)
    # ATen kernels for some loss ops (`mse_loss`, `huber_loss`, …) use
    # the `out` buffer as intermediate scratch and resize it before the
    # final reduction.  Our `from_blob` outputs are non-resizable, so
    # the kernel aborts the call with this message.  Skip these — the
    # zero-copy wrapper can't drive that codepath.
    "Trying to resize storage that is not resizable",
)

# Random-sampling ops never match a fresh torch reference call —
# they consume RNG state and return different draws.  Skip rather
# than try to align the two PRNG streams.
_RANDOM_OPS = frozenset(
    {
        "bernoulli",
        "bernoulli_",
        "multinomial",
        "normal",
        "rand",
        "randn",
        "randint",
        "randperm",
        "random_",
        "rrelu_with_noise",
    }
)

# Ops whose vendor kernel hangs indefinitely on at least one platform
# (`mode` on `torch_musa` for MUSA tensors).  Skip until the vendor
# fixes the underlying kernel — letting the CI block on a hanging
# kernel costs ~30 min per platform run.
_VENDOR_HANG_OPS = frozenset(
    {
        "mode",
    }
)

# Ops whose vendor `_out` path does not match the vendor functional reference
# for the generic random inputs generated by this harness.
_VENDOR_DIVERGENT_OPS = frozenset(
    {
        ("musa", "native_batch_norm"),
    }
)

# Ops whose vendor kernel crashes the Python process, so they must be skipped
# before calling into the InfiniOps/PyTorch slot.
_VENDOR_CRASH_OPS = frozenset(
    {
        ("npu", "mish"),
        ("npu", "mse_loss"),
        ("npu", "nonzero"),
        ("npu", "norm"),
        ("npu", "nuclear_norm"),
        ("npu", "smooth_l1_loss"),
        ("npu", "soft_margin_loss"),
        ("npu", "_linalg_svd"),
        ("npu", "svd"),
    }
)

# Ops where the ATen `_out` schema and the Python reference (`torch.<op>`,
# `torch.nn.functional.<op>`) diverge in positional-argument ordering or
# parameter representation, so the harness's purely-positional reference call
# lands an InfiniOps argument on the wrong reference parameter. E.g. ATen
# `binary_cross_entropy_out(self, target, weight=None, reduction=Mean, out)`
# uses integer `reduction`, while `torch.nn.functional.binary_cross_entropy`
# has legacy `size_average` / `reduce` parameters before string `reduction`.
# The InfiniOps wrapper itself is fine; only the harness's reference call is
# wrong.
_REFERENCE_SIGNATURE_MISMATCH_OPS = frozenset(
    {
        "binary_cross_entropy",
        "binary_cross_entropy_backward",
    }
)

# ATen `_compute_linear_combination.out` accumulates into `out`, so matching
# the functional reference requires a zero-initialized destination.
_ZERO_INITIALIZED_OUT_OPS = frozenset({"_compute_linear_combination"})

# Full reductions with low-precision inputs diverge between the functional
# (`torch.<op>(x)`) and `_out` paths because of intermediate-precision
# choices we cannot align from outside ATen.
_LARGE_REDUCTION_OPS = frozenset(
    {"sum", "mean", "nansum", "nanmean", "prod", "std", "var"}
)

# Ops with input-domain `TORCH_CHECK` macros that fire as device-side
# `assert` on CUDA when our generic random fp32 inputs fall outside the
# expected range.  The Python-side `RuntimeError` is catchable, but the
# CUDA context is left poisoned and every subsequent test errors at
# setup.  Skip these on cuda; the CPU path raises a clean exception
# that the existing harness already handles.
_DEVICE_ASSERTING_OPS = frozenset(
    {
        "binary_cross_entropy",  # requires inputs in [0, 1]
        "multi_margin_loss",
        "multilabel_margin_loss",
        "nll_loss",
        "nll_loss2d",
        # cuDNN paths divide by `kernel_size`/`stride` and SIGFPE on the
        # `[0, 0]` defaults our harness substitutes for required `int[N]`
        # parameters.
        "cudnn_convolution",
        "slow_conv3d",
        "slow_conv_transpose2d",
        "slow_conv_transpose3d",
        "thnn_conv2d",
        "im2col",
        "col2im",
        "reflection_pad1d",
        "reflection_pad2d",
        "reflection_pad3d",
        "replication_pad1d",
        "replication_pad2d",
        "replication_pad3d",
        "upsample_bicubic2d",
        "upsample_bilinear2d",
        "upsample_linear1d",
        "upsample_nearest1d",
        "upsample_nearest2d",
        "upsample_nearest3d",
        "upsample_trilinear3d",
        "avg_pool2d",
        "avg_pool3d",
        "max_pool2d_with_indices",
        "max_pool3d_with_indices",
        "adaptive_max_pool2d",
        "adaptive_max_pool3d",
        "adaptive_avg_pool2d",
        "adaptive_avg_pool3d",
    }
)


def _torch_func(op_name):
    """Resolve the reference function across `torch`, `torch.special`,
    and `torch.nn.functional`.  `special_<x>` falls through to
    `torch.special.<x>` with the prefix stripped."""

    if _is_inplace_aten_name(op_name):
        method_name = op_name

        def _call_inplace(input, *args):
            return getattr(input, method_name)(*args)

        return _call_inplace

    if op_name == "index":
        return torch.ops.aten.index.Tensor

    candidates = [
        (torch, op_name),
        (torch.special, op_name),
        (torch.nn.functional, op_name),
    ]

    if op_name.startswith("special_"):
        candidates.append((torch.special, op_name.removeprefix("special_")))

    for namespace, attr in candidates:
        func = getattr(namespace, attr, None)

        if func is not None:
            return func

    pytest.skip(f"no reference function for `{op_name}` in PyTorch")


def _pascal(snake_name):
    return "".join(part.capitalize() for part in snake_name.split("_"))


def _skip_if_not_active(op_name, device):
    op_class = getattr(infini.ops, _pascal(op_name), None)

    if op_class is None:
        pytest.skip(f"`{op_name}` class not exposed on this build")

    if _PYTORCH_SLOT not in op_class.active_implementation_indices(device):
        pytest.skip(f"`{op_name}` slot {_PYTORCH_SLOT} not active on `{device}`")


def _skip_low_precision_reduction(op_name, dtype, device):
    if op_name in _LARGE_REDUCTION_OPS:
        if dtype in (torch.float16, torch.bfloat16):
            pytest.skip(f"`{op_name}` precision diverges on fp16/bf16")

        if device == "musa":
            pytest.skip(f"`{op_name}` on `torch_musa` diverges from CPU reference")


def _build_input_value(op_name, param, shape, dtype, device, tensor_idx):
    """Build the value passed to a non-out parameter."""

    if param["is_tensor"]:
        per_op = _TENSOR_SHAPES.get(op_name)
        tshape = per_op[tensor_idx] if per_op is not None else shape

        return randn_strided(tshape, None, dtype=dtype, device=device)

    key = (op_name, param["name"])

    if key in _SCALAR_VALUES:
        return _SCALAR_VALUES[key]

    t = param["type"]
    optional_inner = _optional_inner_type(t)

    if optional_inner == "Tensor":
        per_op = _TENSOR_SHAPES.get(op_name)
        tshape = per_op[tensor_idx] if per_op is not None else shape

        return randn_strided(tshape, None, dtype=dtype, device=device)

    if optional_inner == "ScalarType":
        return None

    if optional_inner == "str":
        return None

    if optional_inner in {"Scalar", "float"}:
        return 0.5

    if optional_inner in {"int", "SymInt", "bool"}:
        return _TYPE_DEFAULTS[optional_inner]

    if t == "float[]?":
        return None

    if t == "Tensor?[]":
        index = torch.arange(min(2, shape[0]), dtype=torch.int64, device=device)
        return [index, None]

    if t.startswith(("int[", "SymInt[")) or t in {"int[]?", "SymInt[]?"}:
        return _list_default(t)

    if t.startswith(("int[", "SymInt[")) or t in {"int[]", "SymInt[]"}:
        return _list_default(t)

    return _TYPE_DEFAULTS.get(t, 0.5)


def _call_infini(op_name, *args, **kwargs):
    try:
        getattr(infini.ops, op_name)(
            *args, **kwargs, implementation_index=_PYTORCH_SLOT
        )
    except RuntimeError as exc:
        if any(p in str(exc) for p in _VENDOR_SKIP_PATTERNS):
            pytest.skip(f"`{op_name}` unsupported by torch on this device/dtype")

        raise


def _assert_close(actual, expected, rtol, atol):
    if actual.dtype.is_floating_point:
        assert torch.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
    else:
        assert torch.equal(actual, expected)


def _run_max_unpool_cases(op_name, dtype, device, rtol, atol):
    if op_name == "max_unpool2d":
        pool = torch.nn.functional.max_pool2d
        unpool = torch.nn.functional.max_unpool2d
        explicit_spatial_shape = (6, 10)
        explicit_kernel_size = [2, 3]
        explicit_stride = [3, 4]
        explicit_padding = [0, 1]
        spatial_dimensions = 2
    else:
        pool = torch.nn.functional.max_pool3d
        unpool = torch.nn.functional.max_unpool3d
        explicit_spatial_shape = (6, 10, 6)
        explicit_kernel_size = [2, 3, 2]
        explicit_stride = [3, 4, 3]
        explicit_padding = [0, 1, 0]
        spatial_dimensions = 3

    cases = (
        (
            True,
            explicit_spatial_shape,
            explicit_kernel_size,
            explicit_stride,
            explicit_padding,
            "full",
        ),
        (
            False,
            explicit_spatial_shape,
            explicit_kernel_size,
            explicit_stride,
            explicit_padding,
            "spatial",
        ),
    )

    for (
        batched,
        spatial_shape,
        kernel_size,
        stride,
        padding,
        output_size_kind,
    ) in cases:
        input_shape = (1, 2, *spatial_shape) if batched else (2, *spatial_shape)
        source = randn_strided(input_shape, None, dtype=dtype, device=device)
        pooled, indices = pool(
            source,
            kernel_size,
            stride=stride,
            padding=padding,
            return_indices=True,
        )

        if output_size_kind == "full":
            output_size = list(source.shape)
        elif output_size_kind == "spatial":
            output_size = list(source.shape[-spatial_dimensions:])
        else:
            output_size = None

        expected = unpool(pooled, indices, kernel_size, stride, padding, output_size)
        actual = torch.empty_like(expected)

        _call_infini(
            op_name,
            pooled,
            indices,
            kernel_size,
            stride,
            padding,
            output_size,
            actual,
        )

        _assert_close(actual, expected, rtol, atol)

        legacy = torch.empty_like(expected)
        legacy_kwargs = {
            "input": pooled,
            "indices": indices,
            "output_size": list(expected.shape[-spatial_dimensions:]),
            "out": legacy,
        }

        if op_name == "max_unpool3d":
            legacy_kwargs.update(
                stride=kernel_size if stride is None else stride,
                padding=padding,
            )

        _call_infini(op_name, **legacy_kwargs)
        _assert_close(legacy, expected, rtol, atol)


def _testable_ops():
    """Filter the metadata down to ops the harness can drive.

    When multiple ATen overloads share the same `aten_name` they all
    end up under one generated InfiniOps class (e.g., `std.dim` and
    `std.correction` share the same wrapper), but each has a distinct ATen
    `_out` signature.  The reference call we synthesize from
    `op_meta['params']` only exercises one signature; the secondary
    overloads either rely on hidden defaults whose ATen interpretation
    differs from the Python wrapper's (`std.correction(self, dim=None,
    correction=None, ...)` defaults to a different correction than
    `torch.std(self)`), or expose a positional shape that the Python
    reference does not accept (e.g., `binary_cross_entropy_out`'s
    `reduction:int` lands on the reference's `weight:Tensor?`).  Keep
    only the first overload of each `aten_name`."""
    seen = set()
    keep = []

    for op in _METADATA.get("ops", []):
        if op["aten_name"] in seen:
            continue

        seen.add(op["aten_name"])
        keep.append(op)

    return keep


def _op_meta_id(op_meta):
    if not isinstance(op_meta, dict):
        return "empty"

    # Multiple ATen overloads now share a single class name (`scatter` covers
    # `scatter.src`, `scatter.value`, `scatter.reduce`, ...) — disambiguate
    # parametrize ids by appending the visible parameter type signature so
    # pytest does not collapse them into duplicate ids.

    return op_meta["overload_name"]


@pytest.mark.parametrize("op_meta", _testable_ops(), ids=_op_meta_id)
@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize(("dtype", "rtol", "atol"), _DTYPES)
def test_op(op_meta, shape, dtype, device, rtol, atol):
    op_name = op_meta["name"]
    aten_name = op_meta.get("aten_name", op_name)
    is_inplace = _is_inplace_aten_name(aten_name)
    is_implicit_inplace = is_inplace and not any(p["is_out"] for p in op_meta["params"])
    _skip_if_not_active(op_name, device)
    _skip_low_precision_reduction(aten_name, dtype, device)

    if aten_name in {"max_unpool2d", "max_unpool3d"}:
        _run_max_unpool_cases(aten_name, dtype, device, rtol, atol)

        return

    if aten_name in _RANDOM_OPS:
        pytest.skip(f"`{aten_name}` is non-deterministic (independent draws diverge)")

    if aten_name in _REFERENCE_SIGNATURE_MISMATCH_OPS:
        pytest.skip(
            f"`{aten_name}`'s ATen `_out` and Python reference signatures "
            "have different positional ordering"
        )

    if aten_name in _VENDOR_HANG_OPS:
        pytest.skip(f"`{aten_name}` hangs on at least one vendor kernel")

    if (device, aten_name) in _VENDOR_DIVERGENT_OPS:
        pytest.skip(f"`{aten_name}` diverges on `{device}` vendor kernel")

    if (device, aten_name) in _VENDOR_CRASH_OPS:
        pytest.skip(f"`{aten_name}` crashes on `{device}` vendor kernel")

    if device == "cuda" and aten_name in _DEVICE_ASSERTING_OPS:
        pytest.skip(
            f"`{aten_name}` triggers a CUDA device-side assert on random inputs"
        )

    in_params = (
        op_meta["params"]
        if is_implicit_inplace
        else [p for p in op_meta["params"] if not p["is_out"]]
    )
    out_params = [p for p in op_meta["params"] if p["is_out"]]

    # Build inputs in InfiniOps API order.
    inputs = []
    tensor_idx = 0

    for p in in_params:
        inputs.append(
            _build_input_value(aten_name, p, shape, dtype, device, tensor_idx)
        )

        if p["is_tensor"] or p["type"] == "Tensor?":
            tensor_idx += 1

    # Run the reference to discover output shape(s)/dtype(s).
    # An op may reject our generic `randn(shape)` input with any of these
    # exception types — the gap is in our test harness's input synthesis,
    # not in the InfiniOps wrapper.
    ref_inputs = [
        clone_strided(inputs[index])
        if isinstance(inputs[index], torch.Tensor)
        else inputs[index]
        for index in sorted(
            range(len(in_params)),
            key=lambda index: in_params[index]["schema_index"],
        )
    ]

    if aten_name == "elu":
        ref_func = torch.ops.aten.elu.default
    else:
        ref_func = _torch_func(aten_name)

    try:
        ref = ref_func(*ref_inputs)
    except (
        RuntimeError,
        TypeError,
        ValueError,
        IndexError,
        NotImplementedError,
    ) as exc:
        pytest.skip(f"`torch.{aten_name}` rejects these inputs: {exc}")

    ref_outs = ref if isinstance(ref, tuple) else (ref,)

    if is_inplace:
        ref_outs = (ref_inputs[0],)

    if len(ref_outs) != len(out_params):
        # The Python-facing function (e.g. `F.adaptive_max_pool2d`) often
        # exposes a subset of the ATen `_out` schema's outputs (returning
        # only `out`, hiding `indices` behind a `return_indices=True`
        # kwarg).  Without a per-op map of how to coax the full tuple
        # out, skip — the InfiniOps wrapper itself is fine.
        pytest.skip(
            f"`{aten_name}` reference produced {len(ref_outs)} output(s); "
            f"schema declares {len(out_params)}"
        )

    # InfiniOps `DataType` supports only `int{8,16,32,64}`,
    # `uint{8,16,32,64}`, `bool`, `float{16,32,64}`, and `bfloat16`. Tensors with
    # any other torch dtype (`complex64`, `complex128`, etc.) abort
    # on `DataTypeFromString`, so skip the test rather than crash the process.
    tensors = [*ref_outs, *(x for x in inputs if isinstance(x, torch.Tensor))]
    unsupported = next(
        (t.dtype for t in tensors if t.dtype not in _SUPPORTED_DTYPES), None
    )

    if unsupported is not None:
        pytest.skip(
            f"`{op_name}` uses dtype {unsupported} — not in InfiniOps `DataType`"
        )

    # On CUDA, `torch.empty_like` of a 0-element tensor gives a tensor
    # whose `data_ptr()` is unregistered with the device; passing it
    # through to the wrapper trips "pointer resides on host memory".
    if any(t.numel() == 0 for t in ref_outs):
        pytest.skip(
            f"`{op_name}` produced 0-element output (unregistered data_ptr on cuda)"
        )

    if is_implicit_inplace:
        _call_infini(op_name, *inputs)
        _assert_close(inputs[0], ref_outs[0], rtol, atol)

        return

    outs = [torch.empty_like(t) for t in ref_outs]

    if aten_name in _ZERO_INITIALIZED_OUT_OPS:
        for out in outs:
            out.zero_()

    _call_infini(op_name, *inputs, *outs)

    for actual, expected in zip(outs, ref_outs):
        _assert_close(actual, expected, rtol, atol)

    if aten_name == "elu":
        input, alpha = inputs[:2]
        alpha_out = torch.empty_like(ref_outs[0])
        _call_infini(op_name, input, alpha, alpha_out)
        _assert_close(
            alpha_out, torch.nn.functional.elu(input, alpha=alpha), rtol, atol
        )
