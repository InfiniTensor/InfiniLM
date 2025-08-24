#!/usr/bin/env python3

import torch
import ctypes
from ctypes import c_uint64
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)
from enum import Enum, auto

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # shape, x_stride, y_stride, min_val, max_val
    # 基本形状测试
    ((10,), None, None, -1.0, 1.0),
    ((5, 10), None, None, -1.0, 1.0),
    ((2, 3, 4), None, None, -1.0, 1.0),
    # 不同的min_val和max_val
    ((10,), None, None, 0.0, 2.0),
    ((5, 10), None, None, 0.0, 2.0),
    ((2, 3, 4), None, None, 0.0, 2.0),
    ((10,), None, None, -2.0, 0.0),
    ((5, 10), None, None, -2.0, 0.0),
    ((2, 3, 4), None, None, -2.0, 0.0),
    # 奇怪形状测试
    ((7, 13), None, None, -1.0, 1.0),  # 质数维度
    ((3, 5, 7), None, None, -1.0, 1.0),  # 三维质数
    # 非标准形状测试
    ((1, 1), None, None, -1.0, 1.0),  # 最小形状
    ((100, 100), None, None, -1.0, 1.0),  # 大形状
    ((16, 16, 16), None, None, -1.0, 1.0),  # 大三维
    # 极端值测试
    ((10,), None, None, -1000.0, 1000.0),  # 大范围
    ((10,), None, None, -0.001, 0.001),  # 小范围
    ((10,), None, None, 0.0, 0.0),  # min=max
]


_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]


_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-6},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-3},
}


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_X = auto()


_INPLACE = [
    Inplace.INPLACE_X,
    Inplace.OUT_OF_PLACE,
]

_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def clip(y, x, min_val, max_val):
    torch.clamp(x, min_val, max_val, out=y)


def test(
    handle,
    device,
    shape,
    x_stride=None,
    y_stride=None,
    min_val=-1.0,
    max_val=1.0,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=InfiniDtype.F32,
    sync=None,
):
    x = TestTensor(shape, x_stride, dtype, device)
    min_ = TestTensor(
        shape, [0 for _ in shape], dtype, device, mode="zeros", bias=min_val
    )
    max_ = TestTensor(
        shape, [0 for _ in shape], dtype, device, mode="zeros", bias=max_val
    )

    if inplace == Inplace.INPLACE_X:
        if x_stride != y_stride:
            return
        y = x
    else:
        y = TestTensor(shape, y_stride, dtype, device)

    if y.is_broadcast():
        return

    print(
        f"Testing Clip on {InfiniDeviceNames[device]} with shape:{shape} x_stride:{x_stride} y_stride:{y_stride} "
        f"min_val:{min_val} max_val:{max_val} dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}"
    )

    clip(y.torch_tensor(), x.torch_tensor(), min_val, max_val)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()

    check_error(
        LIBINFINIOP.infiniopCreateClipDescriptor(
            handle,
            ctypes.byref(descriptor),
            y.descriptor,
            x.descriptor,
            min_.descriptor,
            max_.descriptor,
        )
    )

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetClipWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    def lib_clip():
        check_error(
            LIBINFINIOP.infiniopClip(
                descriptor,
                workspace.data() if workspace is not None else None,
                workspace_size.value,
                y.data(),
                x.data(),
                min_.data(),
                max_.data(),
                None,
            )
        )

    lib_clip()

    # Destroy the tensor descriptors
    for tensor in [x, y, min_, max_]:
        tensor.destroy_desc()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: clip(y.torch_tensor(), x.torch_tensor(), min_val, max_val), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_clip(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyClipDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
