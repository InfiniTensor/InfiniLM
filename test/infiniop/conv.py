from ctypes import POINTER, Structure, c_int32, c_uint64, c_void_p
import ctypes
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from operatorspy import (
    open_lib,
    to_tensor,
    DeviceEnum,
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    create_handle,
    destroy_handle,
    check_error,
)

from operatorspy.tests.test_utils import get_args
import torch
import math
import ctypes
from torch.nn import functional as F
from typing import List, Tuple

# constant for control whether profile the pytorch and lib functions
# NOTE: need to manually add synchronization function to the lib function,
#       e.g., cudaDeviceSynchronize() for CUDA
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


class ConvDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopConvDescriptor_t = POINTER(ConvDescriptor)


def conv(x, w, stride, padding, dilation):
    match len(x.shape) - 2:
        case 1:
            return F.conv1d(x, w, stride=stride, padding=padding, dilation=dilation)
        case 2:
            return F.conv2d(x, w, stride=stride, padding=padding, dilation=dilation)
        case 3:
            return F.conv3d(x, w, stride=stride, padding=padding, dilation=dilation)
        case _:
            print("Error: Pytorch -> Unsupported tensor dimension")
            return None


# infer the shape of the output given the inputs for a N-ary convolution
def inferShape(
    x_shape: List[int],
    w_shape: List[int],
    pads: List[int],
    strides: List[int],
    dilations: List[int],
) -> Tuple[int, ...]:
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
    return (x_shape[0], w_shape[0]) + tuple(output_dims)


# convert a python tuple to a ctype void pointer
def tuple_to_void_p(py_tuple: Tuple):
    array = ctypes.c_int64 * len(py_tuple)
    data_array = array(*py_tuple)
    return ctypes.cast(data_array, ctypes.c_void_p)


def test(
    lib,
    handle,
    torch_device,
    x_shape,
    w_shape,
    pads,
    strides,
    dilations,
    tensor_stride=None,
    tensor_dtype=torch.float16,
    sync=None
):
    assert len(pads) == len(strides) == len(dilations)
    print(
        f"Testing Conv on {torch_device} with x_shape: {x_shape}, w_shape: {w_shape}, b_shape: {w_shape[0]}, pads: {pads}, strides: {strides}, dilations: {dilations}, x_stride: {tensor_stride} dtype:{tensor_dtype}"
    )
    x = torch.rand(x_shape, dtype=tensor_dtype).to(torch_device)
    w = torch.rand(w_shape, dtype=tensor_dtype).to(torch_device)
    y = torch.zeros(
        inferShape(x.shape, w.shape, pads, strides, dilations), dtype=tensor_dtype
    ).to(torch_device)

    for i in range(NUM_PRERUN if PROFILE else 1):
        ans = conv(x, w, strides, pads, dilations)
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            _ = conv(x, w, strides, pads, dilations)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"pytorch time: {elapsed :6f}")

    x_tensor = to_tensor(x, lib)
    w_tensor = to_tensor(w, lib)
    y_tensor = to_tensor(y, lib)
    
    if sync is not None:
        sync()

    descriptor = infiniopConvDescriptor_t()
    check_error(
        lib.infiniopCreateConvDescriptor(
            handle,
            ctypes.byref(descriptor),
            y_tensor.descriptor,
            x_tensor.descriptor,
            w_tensor.descriptor,
            tuple_to_void_p(pads),
            tuple_to_void_p(strides),
            tuple_to_void_p(dilations),
            len(pads),
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x_tensor.descriptor.contents.invalidate()
    w_tensor.descriptor.contents.invalidate()
    y_tensor.descriptor.contents.invalidate()

    workspaceSize = ctypes.c_uint64(0)
    check_error(
        lib.infiniopGetConvWorkspaceSize(descriptor, ctypes.byref(workspaceSize))
    )
    workspace = torch.zeros(int(workspaceSize.value), dtype=torch.uint8).to(
        torch_device
    )
    workspace_ptr = ctypes.cast(workspace.data_ptr(), ctypes.POINTER(ctypes.c_uint8))

    for i in range(NUM_PRERUN if PROFILE else 1):
        check_error(
            lib.infiniopConv(
                descriptor,
                workspace_ptr,
                workspaceSize,
                y_tensor.data,
                x_tensor.data,
                w_tensor.data,
                None,
            )
        )
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            check_error(
                lib.infiniopConv(
                    descriptor,
                    workspace_ptr,
                    workspaceSize,
                    y_tensor.data,
                    x_tensor.data,
                    w_tensor.data,
                    None,
                )
            )
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"    lib time: {elapsed :6f}")

    if tensor_dtype == torch.float16:
        assert torch.allclose(y, ans, atol=0, rtol=1e-2)
    else:
        assert torch.allclose(y, ans, atol=0, rtol=1e-3)
    check_error(lib.infiniopDestroyConvDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for x_shape, w_shape, pads, strides, dilations, x_strides in test_cases:
        # fmt: off
        test(lib, handle, "cpu", x_shape, w_shape, pads, strides, dilations, x_strides, tensor_dtype=torch.float16)
        test(lib, handle, "cpu", x_shape, w_shape, pads, strides, dilations, x_strides, tensor_dtype=torch.float32)
        # fmt: on
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for x_shape, w_shape, pads, strides, dilations, x_strides in test_cases:
        # fmt: off
        test(lib, handle, "cuda", x_shape, w_shape, pads, strides, dilations, x_strides, tensor_dtype=torch.float16)
        test(lib, handle, "cuda", x_shape, w_shape, pads, strides, dilations, x_strides, tensor_dtype=torch.float32)
        # fmt: on
    destroy_handle(lib, handle)


def test_bang(lib, test_cases):
    import torch_mlu

    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)
    for x_shape, w_shape, pads, strides, dilations, x_strides in test_cases:
        # fmt: off
        test(lib, handle, "mlu", x_shape, w_shape, pads, strides, dilations, x_strides, tensor_dtype=torch.float16)
        test(lib, handle, "mlu", x_shape, w_shape, pads, strides, dilations, x_strides, tensor_dtype=torch.float32)
        # fmt: on
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # x_shape, w_shape, pads, strides, dilations, x_strides
        (
            (32, 3, 4),
            (32, 3, 5),
            (1,),
            (1,),
            (1,),
            None,
        ),
        (
            (1, 3, 4, 4),
            (2, 3, 3, 3),
            (1, 1),
            (1, 2),
            (2, 1),
            None,
        ),
        (
            (32, 3, 128, 128),
            (64, 3, 5, 5),
            (2, 2),
            (2, 2),
            (1, 1),
            None,
        ),
        (
            (1, 1, 4, 4, 4),
            (1, 1, 5, 5, 5),
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            None,
        ),
        (
            (32, 3, 32, 32, 32),
            (64, 3, 5, 5, 5),
            (3, 2, 2),
            (4, 3, 3),
            (2, 2, 1),
            None,
        ),
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateConvDescriptor.restype = c_int32
    lib.infiniopCreateConvDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopConvDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_uint64,
    ]
    lib.infiniopConv.restype = c_int32
    lib.infiniopConv.argtypes = [
        infiniopConvDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyConvDescriptor.restype = c_int32
    lib.infiniopDestroyConvDescriptor.argtypes = [
        infiniopConvDescriptor_t,
    ]

    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    if args.bang:
        test_bang(lib, test_cases)
    if not (args.cpu or args.cuda or args.bang):
        test_cpu(lib, test_cases)
    print("\033[92mTest passed!\033[0m")
