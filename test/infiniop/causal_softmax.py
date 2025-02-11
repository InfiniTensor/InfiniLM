from ctypes import POINTER, Structure, c_int32, c_uint64, c_void_p
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
    rearrange_tensor,
    create_workspace,
)

from operatorspy.tests.test_utils import get_args
import torch


class CausalSoftmaxDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopCausalSoftmaxDescriptor_t = POINTER(CausalSoftmaxDescriptor)


def causal_softmax(x):
    type = x.dtype
    mask = torch.tril(torch.ones_like(x), diagonal=-1).flip(dims=[-2, -1])
    y = x.clone()
    masked = torch.where(mask == 1, -torch.inf, y.to(torch.float32))
    return torch.nn.functional.softmax(masked, dim=-1).to(type)


def test(lib, handle, torch_device, x_shape, x_stride=None, x_dtype=torch.float16):
    print(
        f"Testing CausalSoftmax on {torch_device} with x_shape:{x_shape} x_stride:{x_stride} dtype:{x_dtype}"
    )
    x = torch.rand(x_shape, dtype=x_dtype).to(torch_device)
    if x_stride is not None:
        x = rearrange_tensor(x, x_stride)
    ans = causal_softmax(x)
    x_tensor = to_tensor(x, lib)
    descriptor = infiniopCausalSoftmaxDescriptor_t()
    check_error(
        lib.infiniopCreateCausalSoftmaxDescriptor(
            handle, ctypes.byref(descriptor), x_tensor.descriptor
        )
    )
    workspace_size = c_uint64(0)
    check_error(
        lib.infiniopGetCausalSoftmaxWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x_tensor.descriptor.contents.invalidate()

    workspace = create_workspace(workspace_size.value, x.device)
    check_error(
        lib.infiniopCausalSoftmax(
            descriptor,
            workspace.data_ptr() if workspace is not None else None,
            workspace_size.value,
            x_tensor.data,
            None,
        )
    )
    assert torch.allclose(x, ans, atol=0, rtol=1e-2)
    check_error(lib.infiniopDestroyCausalSoftmaxDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for x_shape, x_stride in test_cases:
        test(lib, handle, "cpu", x_shape, x_stride)
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for x_shape, x_stride in test_cases:
        test(lib, handle, "cuda", x_shape, x_stride)
    destroy_handle(lib, handle)


def test_bang(lib, test_cases):
    import torch_mlu

    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)
    for x_shape, x_stride in test_cases:
        test(lib, handle, "mlu", x_shape, x_stride)
    destroy_handle(lib, handle)

def test_ascend(lib, test_cases):
    import torch_npu

    device = DeviceEnum.DEVICE_ASCEND
    handle = create_handle(lib, device)
    for x_shape, x_stride in test_cases:
        test(lib, handle, "npu", x_shape, x_stride)

    destroy_handle(lib, handle)

if __name__ == "__main__":
    test_cases = [
        # x_shape, x_stride
        ((32, 20, 512), None),
        ((32, 20, 512), (20480, 512, 1)), # Ascend 暂不支持非连续
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateCausalSoftmaxDescriptor.restype = c_int32
    lib.infiniopCreateCausalSoftmaxDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopCausalSoftmaxDescriptor_t),
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopGetCausalSoftmaxWorkspaceSize.restype = c_int32
    lib.infiniopGetCausalSoftmaxWorkspaceSize.argtypes = [
        infiniopCausalSoftmaxDescriptor_t,
        POINTER(c_uint64),
    ]
    lib.infiniopCausalSoftmax.restype = c_int32
    lib.infiniopCausalSoftmax.argtypes = [
        infiniopCausalSoftmaxDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyCausalSoftmaxDescriptor.restype = c_int32
    lib.infiniopDestroyCausalSoftmaxDescriptor.argtypes = [
        infiniopCausalSoftmaxDescriptor_t,
    ]

    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    if args.bang:
        test_bang(lib, test_cases)
    if args.ascend:
        test_ascend(lib, test_cases)
    if not (args.cpu or args.cuda or args.bang or args.ascend):
        test_cpu(lib, test_cases)
    print("\033[92mTest passed!\033[0m")
