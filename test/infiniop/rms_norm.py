from ctypes import POINTER, Structure, c_int32, c_uint64, c_void_p, c_float
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

class RMSNormDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopRMSNormDescriptor_t = POINTER(RMSNormDescriptor)

def rms_norm(x, w, eps):
    input_dtype = x.dtype
    hidden_states = x.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return w * hidden_states.to(input_dtype)


def test(lib, handle, torch_device, y_shape, x_shape, w_shape, dtype=torch.float16, w_dtype=torch.float16):
    print(f"Testing RMS_Norm on {torch_device} with y_shape:{y_shape} x_shape:{x_shape} w_shape:{w_shape}"
        f" dtype:{dtype} w_dtype:{w_dtype}")

    y = torch.zeros(y_shape, dtype=dtype).to(torch_device)
    x = torch.rand(x_shape, dtype=dtype).to(torch_device)
    w = torch.ones(w_shape, dtype=w_dtype).to(torch_device)

    eps = 1e-5
    ans = rms_norm(x, w, eps)

    y_tensor = to_tensor(y, lib)
    x_tensor = to_tensor(x, lib)
    w_tensor = to_tensor(w, lib)

    descriptor = infiniopRMSNormDescriptor_t()
    w_dataType = 0 if w_dtype==torch.float16 else 1

    check_error(
        lib.infiniopCreateRMSNormDescriptor(
            handle, ctypes.byref(descriptor), y_tensor.descriptor, x_tensor.descriptor,
            w_tensor.descriptor, eps
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x_tensor.descriptor.contents.invalidate()
    y_tensor.descriptor.contents.invalidate()
    w_tensor.descriptor.contents.invalidate()

    workspace_size = c_uint64(0)
    check_error(
        lib.infiniopGetRMSNormWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = create_workspace(workspace_size.value, y.device)
    check_error(
        lib.infiniopRMSNorm(
            descriptor,
            workspace.data_ptr() if workspace is not None else None,
            workspace_size.value,
            y_tensor.data,
            x_tensor.data,
            w_tensor.data,
            None,
        )
    )

    assert torch.allclose(y.to(dtype), ans.to(dtype), atol=1e-3, rtol=1e-3)
    check_error(lib.infiniopDestroyRMSNormDescriptor(descriptor))

def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for (y_shape, x_shape, w_shape, dtype, w_dtype) in test_cases:
        test(lib, handle, "cpu", y_shape, x_shape, w_shape, dtype, w_dtype)
    destroy_handle(lib, handle)

def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for (y_shape, x_shape, w_shape, dtype, w_dtype) in test_cases:
        test(lib, handle, "cuda", y_shape, x_shape, w_shape, dtype, w_dtype)
    destroy_handle(lib, handle)

def test_bang(lib, test_cases):
    import torch_mlu
    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)
    for (y_shape, x_shape, w_shape, dtype, w_dtype) in test_cases:
        test(lib, handle, "mlu", y_shape, x_shape, w_shape, dtype, w_dtype)
    destroy_handle(lib, handle)

def test_ascend(lib, test_cases):
    import torch_npu
    device = DeviceEnum.DEVICE_ASCEND
    handle = create_handle(lib, device)
    for (y_shape, x_shape, w_shape, dtype, w_dtype) in test_cases:
        test(lib, handle, "npu", y_shape, x_shape, w_shape, dtype, w_dtype)

    destroy_handle(lib, handle)

if __name__ == "__main__":
    test_cases = [
        # y_shape, x_shape, w_shape, dtype, w_dtype
        ((16, 2048), (16, 2048), (2048,), torch.float16, torch.float16),
        ((16, 2048), (16, 2048), (2048,), torch.float16, torch.float32),
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateRMSNormDescriptor.restype = c_int32
    lib.infiniopCreateRMSNormDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopRMSNormDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_float,
    ]

    lib.infiniopGetRMSNormWorkspaceSize.restype = c_int32
    lib.infiniopGetRMSNormWorkspaceSize.argtypes = [
        infiniopRMSNormDescriptor_t,
        POINTER(c_uint64),
    ]

    lib.infiniopRMSNorm.restypes = c_int32
    lib.infiniopRMSNorm.argtypes = [
        infiniopRMSNormDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyRMSNormDescriptor.restype = c_int32
    lib.infiniopDestroyRMSNormDescriptor.argtypes = [
        infiniopRMSNormDescriptor_t,
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
