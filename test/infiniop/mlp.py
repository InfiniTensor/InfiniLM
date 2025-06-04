from ctypes import POINTER, Structure, c_int32, c_uint64, c_void_p, c_float, c_bool
import ctypes
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
    create_workspace,
)

from operatorspy.tests.test_utils import get_args
import torch
import torch.nn as nn


class MLPDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopMLPDescriptor_t = POINTER(MLPDescriptor)


def swiglu(a, b):
    return a * b / (1 + torch.exp(-b.float()).to(b.dtype))


def mlp(y, x, w12, w3, alpha, residual):
    input_dtype = x.dtype

    intermediate_size = w3.shape[0]

    a = torch.matmul(
        x.to(torch.float32), w12[:, intermediate_size:].to(torch.float32)
    ).to(input_dtype)
    b = torch.matmul(
        x.to(torch.float32), w12[:, 0:intermediate_size].to(torch.float32)
    ).to(input_dtype)
    c = swiglu(a, b)
    d = torch.matmul(c.to(torch.float32), alpha * w3.to(torch.float32)).to(input_dtype)
    out = d + y if residual else d
    return out


def test(
    lib,
    handle,
    torch_device,
    num_tokens,
    hidden_size,
    intermediate_size,
    alpha,
    residual,
    dtype=torch.float16,
    x_stride=None,
    y_stride=None,
    w12_stride=None,
    w3_stride=None,
    sync=None
):
    print(
        f"Testing MLP on {torch_device} with num_tokens:{num_tokens} hidden_size:{hidden_size} intermediate_size:{intermediate_size}"
        f" alpha:{alpha} residual:{residual} dtype:{dtype} x_stride:{x_stride} y_stride:{y_stride} w12_stride:{w12_stride} w3_stride:{w3_stride}"
    )

    y = torch.rand([num_tokens, hidden_size], dtype=dtype).to(torch_device) * 0.01
    x = torch.rand([num_tokens, hidden_size], dtype=dtype).to(torch_device) * 0.01
    w12 = (
        torch.rand([hidden_size, 2 * intermediate_size], dtype=dtype).to(torch_device)
        * 0.01
    )
    w3 = (
        torch.rand([intermediate_size, hidden_size], dtype=dtype).to(torch_device)
        * 0.01
    )

    ans = mlp(y, x, w12, w3, alpha, residual)

    if x_stride is not None:
        x = rearrange_tensor(x, x_stride)
    if y_stride is not None:
        y = rearrange_tensor(y, y_stride)
    if w12_stride is not None:
        w12 = rearrange_tensor(w12, w12_stride)
    if w3_stride is not None:
        w3 = rearrange_tensor(w3, w3_stride)

    y_tensor = to_tensor(y, lib)
    x_tensor = to_tensor(x, lib)
    w12_tensor = to_tensor(w12, lib)
    w3_tensor = to_tensor(w3, lib)
    
    if sync is not None:
        sync()

    descriptor = infiniopMLPDescriptor_t()
    check_error(
        lib.infiniopCreateMLPDescriptor(
            handle,
            ctypes.byref(descriptor),
            y_tensor.descriptor,
            x_tensor.descriptor,
            w12_tensor.descriptor,
            w3_tensor.descriptor,
            alpha,
            residual,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    y_tensor.descriptor.contents.invalidate()
    x_tensor.descriptor.contents.invalidate()
    w12_tensor.descriptor.contents.invalidate()
    w3_tensor.descriptor.contents.invalidate()

    workspace_size = c_uint64(0)
    check_error(
        lib.infiniopGetMLPWorkspaceSize(descriptor, ctypes.byref(workspace_size))
    )
    workspace = create_workspace(workspace_size.value, x.device)

    check_error(
        lib.infiniopMLP(
            descriptor,
            workspace.data_ptr() if workspace is not None else None,
            workspace_size.value,
            y_tensor.data,
            x_tensor.data,
            w12_tensor.data,
            w3_tensor.data,
            None,
        )
    )
    assert torch.allclose(y, ans, atol=0, rtol=2e-2)

    check_error(lib.infiniopDestroyMLPDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)

    for (
        num_tokens,
        hidden_size,
        intermediate_size,
        alpha,
        residual,
        dtype,
        x_stride,
        y_stride,
        w12_stride,
        w3_stride,
    ) in test_cases:
        test(
            lib,
            handle,
            "cpu",
            num_tokens,
            hidden_size,
            intermediate_size,
            alpha,
            residual,
            dtype,
            x_stride,
            y_stride,
            w12_stride,
            w3_stride,
        )

    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)

    for (
        num_tokens,
        hidden_size,
        intermediate_size,
        alpha,
        residual,
        dtype,
        x_stride,
        y_stride,
        w12_stride,
        w3_stride,
    ) in test_cases:
        test(
            lib,
            handle,
            "cuda",
            num_tokens,
            hidden_size,
            intermediate_size,
            alpha,
            residual,
            dtype,
            x_stride,
            y_stride,
            w12_stride,
            w3_stride,
        )

    destroy_handle(lib, handle)


def test_bang(lib, test_cases):
    import torch_mlu

    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)

    for (
        num_tokens,
        hidden_size,
        intermediate_size,
        alpha,
        residual,
        dtype,
        x_stride,
        y_stride,
        w12_stride,
        w3_stride,
    ) in test_cases:
        test(
            lib,
            handle,
            "mlu",
            num_tokens,
            hidden_size,
            intermediate_size,
            alpha,
            residual,
            dtype,
            x_stride,
            y_stride,
            w12_stride,
            w3_stride,
        )

    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # num_tokens, hidden_size, intermediate_size, alpha, residual, dtype, x_stride, y_stride, w12_stride, w3_stride
        (4, 4096, 11008, 1.0, True, torch.float16, None, None, None, None),
        (4, 4096, 11008, 1.0, True, torch.float16, [8192, 1], [8192, 1], None, None),
        (
            4,
            4096,
            11008,
            1.0,
            True,
            torch.float16,
            None,
            None,
            [1, 4096],
            [1, 11008],
        ),
        (4, 4096, 11008, 1.0, False, torch.float16, None, None, None, None),
        (4, 4096, 11008, 1.0, False, torch.float16, [8192, 1], [8192, 1], None, None),
    ]
    args = get_args()
    lib = open_lib()

    lib.infiniopCreateMLPDescriptor.restype = c_int32
    lib.infiniopCreateMLPDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopMLPDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_float,
        c_bool,
    ]

    lib.infiniopGetMLPWorkspaceSize.restype = c_int32
    lib.infiniopGetMLPWorkspaceSize.argtypes = [
        infiniopMLPDescriptor_t,
        POINTER(c_uint64),
    ]

    lib.infiniopMLP.restype = c_int32
    lib.infiniopMLP.argtypes = [
        infiniopMLPDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyMLPDescriptor.restype = c_int32
    lib.infiniopDestroyMLPDescriptor.argtypes = [
        infiniopMLPDescriptor_t,
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
