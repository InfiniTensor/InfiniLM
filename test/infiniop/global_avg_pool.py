from ctypes import POINTER, Structure, c_int32, c_void_p, c_uint64
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
import torch, time

# constant for control whether profile the pytorch and lib functions
# NOTE: need to manually add synchronization function to the lib function,
#       e.g., cudaDeviceSynchronize() for CUDA
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


class GlobalAvgPoolDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopGlobalAvgPoolDescriptor_t = POINTER(GlobalAvgPoolDescriptor)


def inferShape(x):
    return x.shape[:2] + (1,) * (x.dim() - 2)


def globalAvgPool(x):
    y = torch.mean(x, dim=tuple(range(2, x.dim())), keepdim=True)
    if PROFILE:
        torch.cuda.synchronize()
    return y.view(*inferShape(x))


def test(
    lib,
    handle,
    torch_device,
    x_shape,
    tensor_dtype=torch.float16,
    sync=None
):
    print(
        f"Testing GlobalAvgPool on {torch_device} with input tensor_shape: {x_shape} dtype: {tensor_dtype}"
    )

    x = torch.rand(x_shape, dtype=tensor_dtype).to(torch_device)
    y = torch.zeros(inferShape(x), dtype=tensor_dtype).to(torch_device)

    for i in range(NUM_PRERUN if PROFILE else 1):
        ans = globalAvgPool(x)
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            _ = globalAvgPool(x)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"pytorch time: {elapsed :6f}")

    x_tensor = to_tensor(x, lib)
    y_tensor = to_tensor(y, lib)
    
    if sync is not None:
        sync()

    descriptor = infiniopGlobalAvgPoolDescriptor_t()
    check_error(
        lib.infiniopCreateGlobalAvgPoolDescriptor(
            handle,
            ctypes.byref(descriptor),
            y_tensor.descriptor,
            x_tensor.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x_tensor.descriptor.contents.invalidate()
    y_tensor.descriptor.contents.invalidate()

    workspaceSize = ctypes.c_uint64(0)
    check_error(
        lib.infiniopGetGlobalAvgPoolWorkspaceSize(
            descriptor, ctypes.byref(workspaceSize)
        )
    )
    workspace = torch.zeros(int(workspaceSize.value), dtype=torch.uint8).to(
        torch_device
    )
    workspace_ptr = ctypes.cast(workspace.data_ptr(), ctypes.POINTER(ctypes.c_uint8))

    for i in range(NUM_PRERUN if PROFILE else 1):
        check_error(
            lib.infiniopGlobalAvgPool(
                descriptor,
                workspace_ptr,
                workspaceSize,
                y_tensor.data,
                x_tensor.data,
                None,
            )
        )
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            check_error(
                lib.infiniopGlobalAvgPool(
                    descriptor,
                    workspace_ptr,
                    workspaceSize,
                    y_tensor.data,
                    x_tensor.data,
                    None,
                )
            )
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"    lib time: {elapsed :6f}")

    assert torch.allclose(y, ans, atol=0, rtol=1e-3)
    check_error(lib.infiniopDestroyGlobalAvgPoolDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for x_shape in test_cases:
        test(lib, handle, "cpu", x_shape, tensor_dtype=torch.float16)
        test(lib, handle, "cpu", x_shape, tensor_dtype=torch.float32)
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for x_shape in test_cases:
        test(lib, handle, "cuda", x_shape, tensor_dtype=torch.float16)
        test(lib, handle, "cuda", x_shape, tensor_dtype=torch.float32)
    destroy_handle(lib, handle)


def test_bang(lib, test_cases):
    import torch_mlu

    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)
    for x_shape in test_cases:
        test(lib, handle, "mlu", x_shape, tensor_dtype=torch.float16)
        test(lib, handle, "mlu", x_shape, tensor_dtype=torch.float32)
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
        # x_shape
        ((1, 3, 3)),
        ((1, 3, 1, 1, 3)),
        ((1, 3, 1, 1, 257)),
        ((1, 2, 1, 1, 514)),
        ((1, 3, 1, 1, 1025)),
        ((32, 256, 1, 112, 112)),
        ((2, 3, 2048000)),
        ((2, 1, 10243)),
        ((2, 20, 100)),
        ((3, 33, 333)),
        ((32, 20, 512)),
        ((3, 3, 11, 11, 11, 3, 2)),
        ((32, 256, 1, 112, 112)),
        ((32, 256, 112, 112)),
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateGlobalAvgPoolDescriptor.restype = c_int32
    lib.infiniopCreateGlobalAvgPoolDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopGlobalAvgPoolDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopGetGlobalAvgPoolWorkspaceSize.restype = c_int32
    lib.infiniopGetGlobalAvgPoolWorkspaceSize.argtypes = [
        infiniopGlobalAvgPoolDescriptor_t,
        POINTER(c_uint64),
    ]
    lib.infiniopGlobalAvgPool.restype = c_int32
    lib.infiniopGlobalAvgPool.argtypes = [
        infiniopGlobalAvgPoolDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyGlobalAvgPoolDescriptor.restype = c_int32
    lib.infiniopDestroyGlobalAvgPoolDescriptor.argtypes = [
        infiniopGlobalAvgPoolDescriptor_t,
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
