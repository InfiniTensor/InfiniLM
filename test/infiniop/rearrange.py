import ctypes
from ctypes import POINTER, Structure, c_int32, c_uint64, c_void_p
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from operatorspy import (
    open_lib,
    to_tensor,
    CTensor,
    DeviceEnum,
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    create_handle,
    destroy_handle,
    check_error,
    rearrange_tensor,
)

from operatorspy.tests.test_utils import get_args
import torch


class RerrangeDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopRearrangeDescriptor_t = POINTER(RerrangeDescriptor)


def test(
    lib,
    handle,
    torch_device,
    x_shape,
    x_stride,
    y_shape,
    y_stride,
    x_dtype=torch.float16,
):
    print(
        f"Testing Rerrange on {torch_device} with x_shape:{x_shape} x_stride:{x_stride} y_shape:{y_shape} y_stride:{y_stride} x_dtype:{x_dtype}"
    )
    x = torch.rand(x_shape, dtype=x_dtype).to(torch_device)
    y = torch.zeros(y_shape, dtype=x_dtype).to(torch_device)
    if x_stride is not None:
        x = rearrange_tensor(x, x_stride)
    if y_stride is not None:
        y = rearrange_tensor(y, y_stride)
    x_tensor = to_tensor(x, lib)
    y_tensor = to_tensor(y, lib)

    descriptor = infiniopRearrangeDescriptor_t()
    check_error(
        lib.infiniopCreateRearrangeDescriptor(
            handle, ctypes.byref(descriptor), y_tensor.descriptor, x_tensor.descriptor
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x_tensor.descriptor.contents.invalidate()
    y_tensor.descriptor.contents.invalidate()

    check_error(
        lib.infiniopRearrange(descriptor, y_tensor.data, x_tensor.data, None)
    )
    assert torch.allclose(x, y, atol=0, rtol=1e-3)
    check_error(lib.infiniopDestroyRearrangeDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for test_case in test_cases:
        x_shape, x_stride = test_case[0]
        y_shape, y_stride = test_case[1]
        test(lib, handle, "cpu", x_shape, x_stride, y_shape, y_stride)
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for test_case in test_cases:
        x_shape, x_stride = test_case[0]
        y_shape, y_stride = test_case[1]
        test(lib, handle, "cuda", x_shape, x_stride, y_shape, y_stride)
    destroy_handle(lib, handle)

def test_bang(lib, test_cases):
    import torch_mlu
    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)
    for test_case in test_cases:
        x_shape, x_stride = test_case[0]
        y_shape, y_stride = test_case[1]
        test(lib, handle, "mlu", x_shape, x_stride, y_shape, y_stride)
    destroy_handle(lib, handle)

def test_ascend(lib, test_cases):
    import torch_npu

    device = DeviceEnum.DEVICE_ASCEND
    handle = create_handle(lib, device)
    for test_case in test_cases:
        x_shape, x_stride = test_case[0]
        y_shape, y_stride = test_case[1]
        test(lib, handle, "npu", x_shape, x_stride, y_shape, y_stride)
    destroy_handle(lib, handle) 

if __name__ == "__main__":
    args = get_args()
    test_cases = [
        # ((src_shape, src_stride), (dst_shape, dst_stride))
        (((2, 4, 32), None), ((2, 4, 32), (256, 64, 1))),
        (((32, 6, 64), (64, 2560, 1)), ((32, 6, 64), None)),
        (((4, 6, 64), (64, 2560, 1)), ((4, 6, 64), (131072, 64, 1))),
        (((1, 32, 64), (2048, 64, 1)), ((1, 32, 64), (2048, 64, 1))),
        (((32, 1, 64), (64, 2560, 1)), ((32, 1, 64), (64, 64, 1))),
        (((4, 1, 64), (64, 2560, 1)), ((4, 1, 64), (64, 11264, 1))),
        (((64,), (1,)), ((64,), (1,))),
        ]
    lib = open_lib()
    lib.infiniopCreateRearrangeDescriptor.restype = c_int32
    lib.infiniopCreateRearrangeDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopRearrangeDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopRearrange.restype = c_int32
    lib.infiniopRearrange.argtypes = [
        infiniopRearrangeDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyRearrangeDescriptor.restype = c_int32
    lib.infiniopDestroyRearrangeDescriptor.argtypes = [infiniopRearrangeDescriptor_t]
    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    if args.bang:
        test_bang(lib, test_cases)
    if args.ascend:
        test_ascend(lib, test_cases)
    print("\033[92mTest passed!\033[0m")
