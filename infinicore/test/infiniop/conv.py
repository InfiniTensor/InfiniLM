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
from typing import List, Tuple
import math
from torch.nn import functional as F

# constant for control whether profile the pytorch and lib functions
# NOTE: need to manually add synchronization function to the lib function,
#       e.g., cudaDeviceSynchronize() for CUDA
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000
_TEST_CASES = [
    # x_shape, x_stride, w_shape, w_stride, pads, strides, dilations, x_strides
    (
        (32, 3, 4),
        (12, 4, 1),
        (32, 3, 5),
        (15, 5, 1),
        (1,),
        (1,),
        (1,),
    ),
    (
        (1, 3, 4, 4),
        (48, 16, 4, 1),
        (2, 3, 3, 3),
        (27, 9, 3, 1),
        (1, 1),
        (1, 2),
        (2, 1),
    ),
    (
        (32, 3, 32, 32),
        (32 * 32 * 3, 32 * 32, 32, 1),
        (64, 3, 5, 5),
        (75, 25, 5, 1),
        (2, 2),
        (2, 2),
        (1, 1),
    ),
    (
        (1, 1, 4, 4, 4),
        (64, 64, 16, 4, 1),
        (1, 1, 5, 5, 5),
        (125, 125, 25, 5, 1),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
    ),
    (
        (32, 3, 32, 32, 32),
        (32 * 32 * 32 * 3, 32 * 32 * 32, 32 * 32, 32, 1),
        (64, 3, 5, 5, 5),
        (375, 125, 25, 5, 1),
        (3, 2, 2),
        (4, 3, 3),
        (2, 2, 1),
    ),
]


# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def conv(x, w, stride, padding, dilation, y_tensor, bias=None):
    match len(x.shape) - 2:
        case 1:
            y_tensor.copy_(
                F.conv1d(
                    x, w, bias=bias, stride=stride, padding=padding, dilation=dilation
                )
            )
        case 2:
            y_tensor.copy_(
                F.conv2d(
                    x, w, bias=bias, stride=stride, padding=padding, dilation=dilation
                )
            )
        case 3:
            y_tensor.copy_(
                F.conv3d(
                    x, w, bias=bias, stride=stride, padding=padding, dilation=dilation
                )
            )
        case _:
            print("Error: Pytorch -> Unsupported tensor dimension")


# infer the shape of the output given the inputs for a N-ary convolution
def inferShapeStride(
    x_shape: List[int],
    w_shape: List[int],
    pads: List[int],
    strides: List[int],
    dilations: List[int],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    assert (
        len(x_shape)
        == len(w_shape)
        == len(pads) + 2
        == len(dilations) + 2
        == len(strides) + 2
    ), "x and w should have the same length; pads, strides, and dilatinos should have the same length; the length of pads should be that of x - 2"
    output_dims = [
        math.floor(
            (x_shape[i + 2] + 2 * pads[i] - dilations[i] * (w_shape[i + 2] - 1) - 1)
            / strides[i]
            + 1
        )
        for i in range(len(pads))
    ]
    output_shape = (x_shape[0], w_shape[0]) + tuple(output_dims)
    output_strides = [1]
    for s in reversed(output_shape[1:]):
        output_strides.insert(0, output_strides[0] * s)
    output_strides = tuple(output_strides)
    return output_shape, output_strides


# convert a python tuple to a ctype void pointer
def tuple_to_void_p(py_tuple: Tuple):
    array = ctypes.c_int64 * len(py_tuple)
    data_array = array(*py_tuple)
    return ctypes.cast(data_array, ctypes.c_void_p)


def test(
    handle,
    device,
    x_shape,
    x_stride,
    w_shape,
    w_stride,
    pads,
    strides,
    dilations,
    tensor_dtype=InfiniDtype.F16,
    sync=None,
):
    assert len(pads) == len(strides) == len(dilations)
    x = TestTensor(x_shape, x_stride, dt=tensor_dtype, device=device, scale=0.01)
    w = TestTensor(w_shape, w_stride, dt=tensor_dtype, device=device, scale=0.01)
    y_shape, y_stride = inferShapeStride(x_shape, w_shape, pads, strides, dilations)
    y = TestTensor(y_shape, y_stride, dt=tensor_dtype, device=device)

    b = (
        TestTensor((w.shape[0],), (1,), dt=tensor_dtype, device=device, scale=0.01)
        if w.shape[0] > 1
        else None
    )
    print(
        f"Testing Conv on {InfiniDeviceNames[device]} with x_shape: {x_shape}, w_shape: {w_shape}, b_shape: {w_shape[0]}, pads: {pads}, strides: {strides}, dilations: {dilations}, x_stride: {x_stride} dtype:{InfiniDtypeNames[tensor_dtype]}"
    )
    conv(
        x.torch_tensor(),
        w.torch_tensor(),
        strides,
        pads,
        dilations,
        y.torch_tensor(),
        b.torch_tensor() if b is not None else None,
    )

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateConvDescriptor(
            handle,
            ctypes.byref(descriptor),
            y.descriptor,
            x.descriptor,
            w.descriptor,
            b.descriptor if b is not None else None,
            tuple_to_void_p(pads),
            tuple_to_void_p(strides),
            tuple_to_void_p(dilations),
            len(pads),
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [x, y, w, b]:
        if tensor is not None:
            tensor.destroy_desc()

    workspace_size = ctypes.c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetConvWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, y.device)

    def lib_conv():
        check_error(
            LIBINFINIOP.infiniopConv(
                descriptor,
                workspace.data(),
                workspace_size.value,
                y.data(),
                x.data(),
                w.data(),
                b.data() if b is not None else None,
                None,
            )
        )

    lib_conv()
    atol, rtol = get_tolerance(_TOLERANCE_MAP, tensor_dtype)
    if DEBUG:
        debug(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: conv(x.torch_tensor(), w.torch_tensor(), strides, pads, dilations, b.torch_tensor() if b is not None else None), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_conv(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyConvDescriptor(descriptor))


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
