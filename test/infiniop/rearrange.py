import torch
import ctypes
from ctypes import POINTER, Structure, c_int32, c_size_t, c_uint64, c_void_p, c_float
from libinfiniop import (
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    open_lib,
    to_tensor,
    get_test_devices,
    check_error,
    rearrange_if_needed,
    create_workspace,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
)

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES = [
    # ((src_shape, src_stride), (dst_shape, dst_stride))
    (((2, 4, 32), None), ((2, 4, 32), (256, 64, 1))),
    (((32, 6, 64), (64, 2560, 1)), ((32, 6, 64), None)),
    (((4, 6, 64), (64, 2560, 1)), ((4, 6, 64), (131072, 64, 1))),
    (((1, 32, 64), (2048, 64, 1)), ((1, 32, 64), (2048, 64, 1))),
    (((32, 1, 64), (64, 2560, 1)), ((32, 1, 64), (64, 64, 1))),
    (((4, 1, 64), (64, 2560, 1)), ((4, 1, 64), (64, 11264, 1))),
    (((64,), (1,)), ((64,), (1,))),
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

    x, y = [
        rearrange_if_needed(tensor, stride)
        for tensor, stride in zip([x, y], [x_stride, y_stride])
    ]
    x_tensor, y_tensor = [to_tensor(tensor, lib) for tensor in [x, y]]

    descriptor = infiniopRearrangeDescriptor_t()
    check_error(
        lib.infiniopCreateRearrangeDescriptor(
            handle, ctypes.byref(descriptor), y_tensor.descriptor, x_tensor.descriptor
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [x_tensor, y_tensor]:
        tensor.descriptor.contents.invalidate()

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
