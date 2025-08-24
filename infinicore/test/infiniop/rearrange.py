import torch
import ctypes
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
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
    ((100, 100), (1, 100), (100, 1)),  # shape  # x_stride  # y_stride
    ((4, 4), (1, 4), (4, 1)),  # shape  # x_stride  # y_stride
    ((4, 6, 64), (64, 4 * 64, 1), (6 * 64, 64, 1)),  # shape  # x_stride  # y_stride
    ((2000, 2000), (1, 2000), (2000, 1)),  # shape  # x_stride  # y_stride
    ((2001, 2001), (1, 2001), (2001, 1)),  # shape  # x_stride  # y_stride
    ((2, 2, 2, 4), (16, 8, 4, 1), (16, 8, 1, 2)),  # shape  # x_stride  # y_stride
    (
        (3, 4, 7, 53, 9),  # shape
        row_major_strides((3, 4, 7, 53, 9)),  # x_stride
        column_major_strides((3, 4, 7, 53, 9)),  # y_stride
    ),
    (
        (3, 4, 50, 50, 5, 7),  # shape
        row_major_strides((3, 4, 50, 50, 5, 7)),  # x_stride
        column_major_strides((3, 4, 50, 50, 5, 7)),  # y_stride
    ),
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 0, "rtol": 0},
    InfiniDtype.F32: {"atol": 0, "rtol": 0},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def rearrange_torch(y, x, x_shape, y_stride):
    y.set_(y.untyped_storage(), 0, x_shape, y_stride)
    y[:] = x.view_as(y)


def test(
    handle, torch_device, shape, x_stride, y_stride, dtype=InfiniDtype.F16, sync=None
):
    print(
        f"Testing Rerrange on {InfiniDeviceNames[torch_device]} with shape:{shape} x_stride:{x_stride} y_stride:{y_stride} dtype:{InfiniDtypeNames[dtype]}"
    )

    x = TestTensor(shape, x_stride, dtype, device)
    y = TestTensor(shape, y_stride, dtype, device, mode="ones")

    rearrange_torch(y.torch_tensor(), x.torch_tensor(), shape, y_stride)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateRearrangeDescriptor(
            handle, ctypes.byref(descriptor), y.descriptor, x.descriptor
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [x, y]:
        tensor.destroy_desc()

    def lib_rearrange():
        check_error(LIBINFINIOP.infiniopRearrange(descriptor, y.data(), x.data(), None))

    lib_rearrange()

    # Validate results
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: rearrange_torch(y.torch_tensor(), x.torch_tensor(), shape, y_stride), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_rearrange(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyRearrangeDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
