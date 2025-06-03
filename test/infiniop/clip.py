#!/usr/bin/env python3

import torch
import ctypes
from ctypes import POINTER, Structure, c_int32, c_size_t, c_uint64, c_void_p, c_float
from libinfiniop import (
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    open_lib,
    to_tensor,
    get_test_devices,
    check_error,
    rearrange_if_needed,
    create_workspace,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
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
    ((7, 13), None, None, -1.0, 1.0),     # 质数维度
    ((3, 5, 7), None, None, -1.0, 1.0),   # 三维质数
    # 非标准形状测试
    ((1, 1), None, None, -1.0, 1.0),       # 最小形状
    ((100, 100), None, None, -1.0, 1.0),   # 大形状
    ((16, 16, 16), None, None, -1.0, 1.0), # 大三维
    # 极端值测试
    ((10,), None, None, -1000.0, 1000.0),  # 大范围
    ((10,), None, None, -0.001, 0.001),    # 小范围
    ((10,), None, None, 0.0, 0.0),         # min=max
    # 特殊形状测试
    ((0,), None, None, -1.0, 1.0),         # 空张量
    ((1, 0), None, None, -1.0, 1.0),       # 空维度

]


_TENSOR_DTYPES = [torch.float16, torch.float32]


_TOLERANCE_MAP = {
    torch.float16: {"atol": 1e-3, "rtol": 1e-3},
    torch.float32: {"atol": 1e-7, "rtol": 1e-6},
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


class ClipDescriptor(Structure):
    _fields_ = [("device_type", c_int32), ("device_id", c_int32)]
infiniopClipDescriptor_t = POINTER(ClipDescriptor)


def clip(x, min_val, max_val):
    return torch.clamp(x, min_val, max_val)


def create_tensor_with_stride(shape, stride, dtype, device):
    """Create a tensor with specific stride without using view() that might cause errors."""
    x = torch.rand(shape, dtype=dtype, device=device) * 4.0 - 2.0  # Range: [-2, 2]
    if stride is None:
        return x
    if len(shape) == 2 and len(stride) == 2:
        if stride == (shape[1], 1):
            return x.contiguous()
        elif stride == (1, shape[0]):
            return x.transpose(0, 1).contiguous().transpose(0, 1)
        else:
            y = torch.zeros(shape, dtype=dtype, device=device)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    y[i, j] = x[i, j]
            return y.contiguous()
    return x


def test(
    lib,
    handle,
    torch_device,
    shape,
    x_stride=None,
    y_stride=None,
    min_val=-1.0,
    max_val=1.0,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=torch.float32,
):
    print(
        f"Testing Clip on {torch_device} with shape:{shape} x_stride:{x_stride} y_stride:{y_stride} "
        f"min_val:{min_val} max_val:{max_val} dtype:{dtype} inplace:{inplace}"
    )
    x = create_tensor_with_stride(shape, x_stride, dtype, torch_device)
    ans = clip(x, min_val, max_val)
    x = rearrange_if_needed(x, x_stride)
    x_tensor = to_tensor(x, lib)
    if inplace == Inplace.INPLACE_X:
        y = x
        y_tensor = x_tensor
    else:
        y = torch.zeros(shape, dtype=dtype).to(torch_device)
        y = rearrange_if_needed(y, y_stride)
        y_tensor = to_tensor(y, lib)
    descriptor = infiniopClipDescriptor_t()
    check_error(
        lib.infiniopCreateClipDescriptor(
            handle, ctypes.byref(descriptor), y_tensor.descriptor, x_tensor.descriptor
        )
    )

    workspace_size = c_uint64(0)
    check_error(
        lib.infiniopGetClipWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = create_workspace(workspace_size.value, x.device)

    def lib_clip():
        check_error(
            lib.infiniopClip(
                descriptor,
                workspace.data_ptr() if workspace is not None else None,
                workspace_size.value,
                y_tensor.data,
                x_tensor.data,
                c_float(min_val),
                c_float(max_val),
                None,
            )
        )

    lib_clip()

    # Now we can destroy the tensor descriptors
    x_tensor.destroyDesc(lib)
    if inplace != Inplace.INPLACE_X:
        y_tensor.destroyDesc(lib)

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG or not torch.allclose(y, ans, atol=atol, rtol=rtol):
        print("\nExpected:")
        print(ans)
        print("\nActual:")
        print(y)
        print("\nDifference:")
        print(torch.abs(y - ans))
        print("\nMax difference:", torch.max(torch.abs(y - ans)).item())
        debug(y, ans, atol=atol, rtol=rtol)
    assert torch.allclose(y, ans, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: clip(x, min_val, max_val), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_clip(), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(lib.infiniopDestroyClipDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    lib = open_lib()

    lib.infiniopCreateClipDescriptor.restype = c_int32
    lib.infiniopCreateClipDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopClipDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetClipWorkspaceSize.restype = c_int32
    lib.infiniopGetClipWorkspaceSize.argtypes = [
        infiniopClipDescriptor_t,
        POINTER(c_uint64),
    ]

    lib.infiniopClip.restype = c_int32
    lib.infiniopClip.argtypes = [
        infiniopClipDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_float,
        c_float,
        c_void_p,
    ]

    lib.infiniopDestroyClipDescriptor.restype = c_int32
    lib.infiniopDestroyClipDescriptor.argtypes = [
        infiniopClipDescriptor_t,
    ]

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(lib, device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
