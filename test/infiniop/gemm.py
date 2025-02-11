from ctypes import POINTER, Structure, c_int32, c_uint64, c_void_p, c_float, c_bool
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

class GEMMDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopGEMMDescriptor_t = POINTER(GEMMDescriptor)


def gemm(A, B, C=None, transA=False, transB=False, alpha=1.0, beta=0.0, dtype=torch.float32):
    A = A.T if transA else A
    B = B.T if transB else B
    result = alpha * torch.matmul(A if dtype != torch.float16 else A.to(torch.float32), B if dtype != torch.float16 else B.to(torch.float32)).to(dtype)
    if C is not None:
        result += beta * C if dtype != torch.float16 else C.to(torch.float32)
    if PROFILE:
        torch.cuda.synchronize()
    return result


def test(
    lib,
    handle,
    torch_device,
    alpha,
    beta,
    transA,
    transB,
    a_shape,
    b_shape,
    c_shape,
    y_shape,
    a_stride=None,
    b_stride=None,
    c_stride=None,
    y_stride=None,
    dtype=torch.float16,
):
    print(
        f"Testing GEMM on {torch_device} with transA: {transA} transB: {transB} " 
        f"a_shape:{a_shape} b_shape:{b_shape} c_shape:{c_shape} y_shape:{y_shape} "
        f"a_stride:{a_stride} b_stride:{b_stride} c_stride:{c_stride} y_stride:{y_stride} dtype:{dtype}"
    )

    a = torch.rand(a_shape, dtype=dtype).to(torch_device)
    b = torch.rand(b_shape, dtype=dtype).to(torch_device)
    c = torch.rand(c_shape, dtype=dtype).to(torch_device) if c_shape else None
    y = torch.rand(y_shape, dtype=dtype).to(torch_device)

    if a_stride is not None:
        a = rearrange_tensor(a, a_stride)
    if b_stride is not None:
        b = rearrange_tensor(b, b_stride)
    if c_stride is not None and c is not None:
        c = rearrange_tensor(c, c_stride)
    if y_stride is not None:
        y = rearrange_tensor(y, y_stride)

    for i in range(NUM_PRERUN if PROFILE else 1):
        ans = gemm(a, b, c, transA, transB, alpha, beta, dtype)
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            _ = gemm(a, b, c, transA, transB, alpha, beta, dtype)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"pytorch time: {elapsed :6f}")

    a_tensor = to_tensor(a, lib)
    b_tensor = to_tensor(b, lib)
    c_tensor = to_tensor(c, lib) if c is not None else None
    y_tensor = to_tensor(y, lib)
    descriptor = infiniopGEMMDescriptor_t()
    check_error(
        lib.infiniopCreateGEMMDescriptor(
            handle,
            ctypes.byref(descriptor),
            y_tensor.descriptor,
            a_tensor.descriptor,
            b_tensor.descriptor,
            c_tensor.descriptor if c_tensor else None,
            alpha,
            beta,
            transA,
            transB,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    a_tensor.descriptor.contents.invalidate()
    b_tensor.descriptor.contents.invalidate()
    if c_tensor is not None:
        c_tensor.descriptor.contents.invalidate()
    y_tensor.descriptor.contents.invalidate()

    workspace_size = ctypes.c_uint64(0)
    check_error(
        lib.infiniopGetGEMMWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = torch.zeros(int(workspace_size.value), dtype=torch.uint8).to(
        torch_device
    )
    workspace_ptr = ctypes.cast(workspace.data_ptr(), ctypes.POINTER(ctypes.c_uint8))

    for i in range(NUM_PRERUN if PROFILE else 1):
        check_error(
            lib.infiniopGEMM(
                descriptor,
                workspace_ptr,
                workspace_size,
                y_tensor.data,
                a_tensor.data,
                b_tensor.data,
                c_tensor.data if c_tensor else None,
                None,
            )
        )
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            check_error(
                lib.infiniopGEMM(
                    descriptor,
                    workspace_ptr,
                    workspace_size,
                    y_tensor.data,
                    a_tensor.data,
                    b_tensor.data,
                    c_tensor.data if c_tensor else None,
                    None,
                )
            )
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"    lib time: {elapsed :6f}")

    assert torch.allclose(y, ans, atol=0, rtol=1e-2)
    check_error(lib.infiniopDestroyGEMMDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for (
        alpha,
        beta,
        transA,
        transB,
        a_shape,
        b_shape,
        c_shape,
        y_shape,
        a_stride,
        b_stride,
        c_stride,
        y_stride,
    ) in test_cases:
        test(lib, handle, "cpu", alpha, beta, transA, transB, a_shape, b_shape, c_shape, y_shape, a_stride, b_stride, c_stride, y_stride, dtype=torch.float16)
        test(lib, handle, "cpu", alpha, beta, transA, transB, a_shape, b_shape, c_shape, y_shape, a_stride, b_stride, c_stride, y_stride, dtype=torch.float32)
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for (
        alpha,
        beta,
        transA,
        transB,
        a_shape,
        b_shape,
        c_shape,
        y_shape,
        a_stride,
        b_stride,
        c_stride,
        y_stride,
    ) in test_cases:
        test(lib, handle, "cuda", alpha, beta, transA, transB, a_shape, b_shape, c_shape, y_shape, a_stride, b_stride, c_stride, y_stride, dtype=torch.float16)
        test(lib, handle, "cuda", alpha, beta, transA, transB, a_shape, b_shape, c_shape, y_shape, a_stride, b_stride, c_stride, y_stride, dtype=torch.float32)
    destroy_handle(lib, handle)


def test_bang(lib, test_cases):
    import torch_mlu

    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)

    for (
        alpha,
        beta,
        transA,
        transB,
        a_shape,
        b_shape,
        c_shape,
        y_shape,
        a_stride,
        b_stride,
        c_stride,
        y_stride,
    ) in test_cases:
        test(lib, handle, "mlu", alpha, beta, transA, transB, a_shape, b_shape, c_shape, y_shape, a_stride, b_stride, c_stride, y_stride, dtype=torch.float16)
        test(lib, handle, "mlu", alpha, beta, transA, transB, a_shape, b_shape, c_shape, y_shape, a_stride, b_stride, c_stride, y_stride, dtype=torch.float32)

    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # alpha, beta, transA, transB, a_shape, b_shape, c_shape, y_shape, a_stride, b_stride, c_stride, y_stride
        (
            1.0,
            1.0,
            False,
            False,
            (1, 2048),
            (2048, 2048),
            (1, 2048),
            (1, 2048),
            None,
            None,
            None,
            None,
        ),
        (
            1.0,
            1.0,
            True,
            True,
            (2048, 4),
            (2048, 2048),
            (4, 2048),
            (4, 2048),
            None,
            None,
            None,
            None,
        ),
        (
            1.0,
            1.0,
            False,
            True,
            (1, 2048),
            (1000, 2048),
            (1000),
            (1, 1000),
            None,
            None,
            None,
            None,
        ),
        (
            1.0,
            1.0,
            True,
            False,
            (2048, 4),
            (2048, 2048),
            (2048),
            (4, 2048),
            (4096, 1),
            (4096, 1),
            (2,),
            (4096, 1),
        ),
        (
            1.0,
            1.0,
            False,
            False,
            (3, 1, 2048),
            (3, 2048, 2048),
            (1,),
            (3, 1, 2048),
            None,
            None,
            None,
            None,
        ),
        (
            1.0,
            1.0,
            True,
            False,
            (2048, 4),
            (2048, 2048),
            None,
            (4, 2048),
            (4096, 1),
            (4096, 1),
            (2,),
            (4096, 1),
        ),
    ]
    args = get_args()
    lib = open_lib()

    lib.infiniopCreateGEMMDescriptor.restype = c_int32
    lib.infiniopCreateGEMMDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopGEMMDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_float,
        c_float,
        c_bool,
        c_bool,
    ]

    lib.infiniopGetGEMMWorkspaceSize.restype = c_int32
    lib.infiniopGetGEMMWorkspaceSize.argtypes = [
        infiniopGEMMDescriptor_t,
        POINTER(c_uint64),
    ]

    lib.infiniopGEMM.restype = c_int32
    lib.infiniopGEMM.argtypes = [
        infiniopGEMMDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyGEMMDescriptor.restype = c_int32
    lib.infiniopDestroyGEMMDescriptor.argtypes = [
        infiniopGEMMDescriptor_t,
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
