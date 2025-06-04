import torch
import ctypes
from ctypes import POINTER, Structure, c_int32, c_void_p
from libinfiniop import (
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    open_lib,
    to_tensor,
    get_test_devices,
    check_error,
    rearrange_if_needed,
    rearrange_tensor,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
)

def row_major_strides(shape):
    """生成张量的行优先(C风格)stride
    
    Args:
        shape: 张量形状
    
    Returns:
        行优先strides列表
    """
    # 行优先 (C风格，从最后一维到第一维)
    stride = 1
    strides = [1]
    for dim in reversed(shape[1:]):
        stride *= dim
        strides.insert(0, stride)
    return strides

def column_major_strides(shape):
    """生成张量的列优先(Fortran风格)stride
    
    Args:
        shape: 张量形状
    
    Returns:
        列优先strides列表
    """
    # 列优先 (Fortran风格，从第一维到最后一维)
    stride = 1
    strides = [stride]
    for dim in shape[:-1]:
        stride *= dim
        strides.append(stride)
    return strides



# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES = [
    # (shape, x_stride, y_stride)
    (
        (2, 4, 64),  # shape
        (2, 4, 8),   # x_stride
        (512, 128, 2) # y_stride
    ),
    (
        (100, 100),  # shape
        (1, 100),    # x_stride
        (100, 1)     # y_stride
    ),
    (
        (4, 4),      # shape
        (1, 4),      # x_stride
        (4, 1)       # y_stride
    ),
    (
        (4, 6, 64),  # shape
        (64, 4*64, 1), # x_stride
        (6*64, 64, 1)  # y_stride
    ),
    (
        (2000, 2000), # shape
        (1, 2000),    # x_stride
        (2000, 1)     # y_stride
    ),
    (
        (2001, 2001), # shape
        (1, 2001),    # x_stride
        (2001, 1)     # y_stride
    ),
    (
        (3, 4, 7, 53, 9), # shape
        row_major_strides((3, 4, 7, 53, 9)), # x_stride
        column_major_strides((3, 4, 7, 53, 9)) # y_stride
    ),
    (
        (3, 4, 50, 50, 5, 7), # shape
        row_major_strides((3, 4, 50, 50, 5, 7)),  # x_stride
        column_major_strides((3, 4, 50, 50, 5, 7)) # y_stride
    ),
]

# Data types used for testing
_TENSOR_DTYPES = [torch.float16, torch.float32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    torch.float16: {"atol": 0, "rtol": 0},
    torch.float32: {"atol": 0, "rtol": 0},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


class RerrangeDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopRearrangeDescriptor_t = POINTER(RerrangeDescriptor)


def test(
    lib,
    handle,
    torch_device,
    shape,
    x_stride,
    y_stride,
    dtype=torch.float16,
    sync=None
):
    print(
        f"Testing Rerrange on {torch_device} with shape:{shape} x_stride:{x_stride} y_stride:{y_stride} dtype:{dtype}"
    )

    x = torch.rand(shape, dtype=dtype).to(torch_device)
    y = torch.zeros(shape, dtype=dtype).to(torch_device)

    x, y = [
        rearrange_if_needed(tensor, stride)
        for tensor, stride in zip([x, y], [x_stride, y_stride])
    ]

    x_tensor, y_tensor = [to_tensor(tensor, lib) for tensor in [x, y]]
    
    if sync is not None:
        sync()

    descriptor = infiniopRearrangeDescriptor_t()
    check_error(
        lib.infiniopCreateRearrangeDescriptor(
            handle, ctypes.byref(descriptor), y_tensor.descriptor, x_tensor.descriptor
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [x_tensor, y_tensor]:
        tensor.destroyDesc(lib)

    def lib_rearrange():
        check_error(
            lib.infiniopRearrange(descriptor, y_tensor.data, x_tensor.data, None)
        )

    lib_rearrange()

    # Validate results
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(x, y, atol=atol, rtol=rtol)
    assert torch.allclose(x, y, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: rearrange_tensor(y, y_stride), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_rearrange(), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(lib.infiniopDestroyRearrangeDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
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

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(lib, device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
