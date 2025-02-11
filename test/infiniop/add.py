from ctypes import POINTER, Structure, c_int32, c_void_p
import ctypes
import sys
import os

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
from enum import Enum, auto
import torch


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_A = auto()
    INPLACE_B = auto()


class AddDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopAddDescriptor_t = POINTER(AddDescriptor)


def add(x, y):
    return torch.add(x, y)


def test(
    lib,
    handle,
    torch_device,
    c_shape, 
    a_shape, 
    b_shape,
    tensor_dtype=torch.float16,
    inplace=Inplace.OUT_OF_PLACE,
):
    print(
        f"Testing Add on {torch_device} with c_shape:{c_shape} a_shape:{a_shape} b_shape:{b_shape} dtype:{tensor_dtype} inplace: {inplace.name}"
    )
    if a_shape != b_shape and inplace != Inplace.OUT_OF_PLACE:
        print("Unsupported test: broadcasting does not support in-place")
        return

    a = torch.rand(a_shape, dtype=tensor_dtype).to(torch_device)
    b = torch.rand(b_shape, dtype=tensor_dtype).to(torch_device)
    c = torch.rand(c_shape, dtype=tensor_dtype).to(torch_device) if inplace == Inplace.OUT_OF_PLACE else (a if inplace == Inplace.INPLACE_A else b)

    ans = add(a, b)

    a_tensor = to_tensor(a, lib)
    b_tensor = to_tensor(b, lib)
    c_tensor = to_tensor(c, lib) if inplace == Inplace.OUT_OF_PLACE else (a_tensor if inplace == Inplace.INPLACE_A else b_tensor)
    descriptor = infiniopAddDescriptor_t()

    check_error(
        lib.infiniopCreateAddDescriptor(
            handle,
            ctypes.byref(descriptor),
            c_tensor.descriptor,
            a_tensor.descriptor,
            b_tensor.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    c_tensor.descriptor.contents.invalidate()
    a_tensor.descriptor.contents.invalidate()
    b_tensor.descriptor.contents.invalidate()

    check_error(
        lib.infiniopAdd(descriptor, c_tensor.data, a_tensor.data, b_tensor.data, None)
    )
    assert torch.allclose(c, ans, atol=0, rtol=1e-3)
    check_error(lib.infiniopDestroyAddDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for c_shape, a_shape, b_shape, inplace in test_cases:
        test(lib, handle, "cpu", c_shape, a_shape, b_shape, tensor_dtype=torch.float16, inplace=inplace)
        test(lib, handle, "cpu", c_shape, a_shape, b_shape, tensor_dtype=torch.float32, inplace=inplace)
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for c_shape, a_shape, b_shape, inplace in test_cases:
        test(lib, handle, "cuda", c_shape, a_shape, b_shape, tensor_dtype=torch.float16, inplace=inplace)
        test(lib, handle, "cuda", c_shape, a_shape, b_shape, tensor_dtype=torch.float32, inplace=inplace)
    destroy_handle(lib, handle)


def test_bang(lib, test_cases):
    import torch_mlu

    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)
    for c_shape, a_shape, b_shape, inplace in test_cases:
        test(lib, handle, "mlu", c_shape, a_shape, b_shape, tensor_dtype=torch.float16, inplace=inplace)
        test(lib, handle, "mlu", c_shape, a_shape, b_shape, tensor_dtype=torch.float32, inplace=inplace)
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # c_shape, a_shape, b_shape, inplace
        # ((32, 150, 512000), (32, 150, 512000), (32, 150, 512000), Inplace.OUT_OF_PLACE),
        # ((32, 150, 51200), (32, 150, 51200), (32, 150, 1), Inplace.OUT_OF_PLACE),
        # ((32, 150, 51200), (32, 150, 51200), (32, 150, 51200), Inplace.OUT_OF_PLACE),
        ((1, 3), (1, 3), (1, 3), Inplace.OUT_OF_PLACE),
        ((), (), (), Inplace.OUT_OF_PLACE),
        ((3, 3), (3, 3), (3, 3), Inplace.OUT_OF_PLACE),
        ((2, 20, 3), (2, 1, 3), (2, 20, 3), Inplace.OUT_OF_PLACE),
        ((32, 20, 512), (32, 20, 512), (32, 20, 512), Inplace.INPLACE_A),
        ((32, 20, 512), (32, 20, 512), (32, 20, 512), Inplace.INPLACE_B),
        ((32, 256, 112, 112), (32, 256, 112, 1), (32, 256, 112, 112), Inplace.OUT_OF_PLACE),
        ((32, 256, 112, 112), (32, 256, 112, 112), (32, 256, 112, 112), Inplace.OUT_OF_PLACE),
        ((2, 4, 3), (2, 1, 3), (4, 3), Inplace.OUT_OF_PLACE),
        ((2, 3, 4, 5), (2, 3, 4, 5), (5,), Inplace.OUT_OF_PLACE),
        ((3, 2, 4, 5), (4, 5), (3, 2, 1, 1), Inplace.OUT_OF_PLACE),
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateAddDescriptor.restype = c_int32
    lib.infiniopCreateAddDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopAddDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopAdd.restype = c_int32
    lib.infiniopAdd.argtypes = [
        infiniopAddDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyAddDescriptor.restype = c_int32
    lib.infiniopDestroyAddDescriptor.argtypes = [
        infiniopAddDescriptor_t,
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
