from ctypes import POINTER, Structure, c_int32, c_void_p
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
    rearrange_tensor,
)

from operatorspy.tests.test_utils import get_args
import torch

# constant for control whether profile the pytorch and lib functions
# NOTE: need to manually add synchronization function to the lib function,
#       e.g., cudaDeviceSynchronize() for CUDA
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


class ExpandDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopExpandDescriptor_t = POINTER(ExpandDescriptor)


def expand(x, y):
    if PROFILE:
        ans = x.expand_as(y).clone()
        torch.cuda.synchronize()
        return ans
    return x.expand_as(y)


def test(
    lib,
    handle,
    torch_device,
    y_shape,
    x_shape,
    y_stride=None,
    x_stride=None,
    tensor_dtype=torch.float16,
    sync=None
):
    print(
        f"Testing Expand on {torch_device} with x_shape:{x_shape} y_shape:{y_shape} x_stride:{x_stride} y_stride:{y_stride} dtype:{tensor_dtype}"
    )

    x = torch.rand(x_shape, dtype=tensor_dtype).to(torch_device)
    y = torch.rand(y_shape, dtype=tensor_dtype).to(torch_device)

    if x_stride is not None:
        x = rearrange_tensor(x, x_stride)
    if y_stride is not None:
        y = rearrange_tensor(y, y_stride)

    for i in range(NUM_PRERUN if PROFILE else 1):
        ans = expand(x, y)
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            _ = expand(x, y)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"pytorch time: {elapsed :6f}")

    x_tensor = to_tensor(x, lib)
    y_tensor = to_tensor(y, lib)
    
    if sync is not None:
        sync()

    descriptor = infiniopExpandDescriptor_t()
    check_error(
        lib.infiniopCreateExpandDescriptor(
            handle,
            ctypes.byref(descriptor),
            y_tensor.descriptor,
            x_tensor.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x_tensor.descriptor.contents.invalidate()
    y_tensor.descriptor.contents.invalidate()

    for i in range(NUM_PRERUN if PROFILE else 1):
        check_error(lib.infiniopExpand(descriptor, y_tensor.data, x_tensor.data, None))
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            check_error(
                lib.infiniopExpand(descriptor, y_tensor.data, x_tensor.data, None)
            )
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"    lib time: {elapsed :6f}")
    assert torch.allclose(y, ans, atol=0, rtol=1e-3)
    check_error(lib.infiniopDestroyExpandDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for y_shape, x_shape, y_stride, x_stride in test_cases:
        # fmt: off
        test(lib, handle, "cpu", y_shape, x_shape, y_stride, x_stride, tensor_dtype=torch.float16)
        test(lib, handle, "cpu", y_shape, x_shape, y_stride, x_stride, tensor_dtype=torch.float32)
        # fmt: on
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for y_shape, x_shape, y_stride, x_stride in test_cases:
        # fmt: off
        test(lib, handle, "cuda", y_shape, x_shape, y_stride, x_stride, tensor_dtype=torch.float16)
        test(lib, handle, "cuda", y_shape, x_shape, y_stride, x_stride, tensor_dtype=torch.float32)
        # fmt: on
    destroy_handle(lib, handle)


def test_bang(lib, test_cases):
    import torch_mlu

    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)
    for y_shape, x_shape, y_stride, x_stride in test_cases:
        # fmt: off
        test(lib, handle, "mlu", y_shape, x_shape, y_stride, x_stride, tensor_dtype=torch.float16)
        test(lib, handle, "mlu", y_shape, x_shape, y_stride, x_stride, tensor_dtype=torch.float32)
        # fmt: on
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # fmt: off
        # y_shape, x_shape, y_stride, x_stride
        ((), (), None, None),
        ((3, 3), (1,), None, None),
        ((5, 4, 3), (4, 3,), None, (6, 1)),
        ((99, 111), (111,), None, None),
        ((2, 4, 3), (1, 3), None, None),
        ((2, 20, 3), (2, 1, 3), None, None),
        ((2, 3, 4, 5), (5,), None, None),
        ((3, 2, 4, 5), (3, 2, 1, 1), None, None),
        ((32, 256, 112, 112), (32, 256, 112, 1), None, None),
        # fmt: on
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateExpandDescriptor.restype = c_int32
    lib.infiniopCreateExpandDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopExpandDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopExpand.restype = c_int32
    lib.infiniopExpand.argtypes = [
        infiniopExpandDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyExpandDescriptor.restype = c_int32
    lib.infiniopDestroyExpandDescriptor.argtypes = [
        infiniopExpandDescriptor_t,
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
