import torch
import ctypes
from ctypes import c_uint64
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
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)
from enum import Enum, auto

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # shape, x_stride, y_stride
    ((3, 3), None, None),
    ((32, 512), None, None),
    ((32, 512), (1024, 1), (1024, 1)),
    ((32, 5, 5), None, None),
    ((32, 20, 512), None, None),
    ((32, 20, 512), (20480, 512, 1), None),
    ((28, 15, 15), None, None),
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-2},
    InfiniDtype.BF16: {"atol": 5e-3, "rtol": 5e-2},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
}


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_X = auto()


_INPLACE = [
    Inplace.INPLACE_X,
    Inplace.OUT_OF_PLACE,
]

_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def causal_softmax(x):
    type = x.dtype
    mask = torch.tril(torch.ones_like(x), diagonal=-1).flip(dims=[-2, -1])
    masked = torch.where(mask == 1, -torch.inf, x.to(torch.float32))
    return torch.nn.functional.softmax(masked, dim=-1, dtype=type)


def test(
    handle,
    device,
    shape,
    x_stride=None,
    y_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing CausalSoftmax on {InfiniDeviceNames[device]} with shape:{shape} x_stride:{x_stride} y_stride:{y_stride} dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}"
    )

    x = TestTensor(shape, x_stride, dtype, device)
    ans = causal_softmax(x.torch_tensor())

    if inplace == Inplace.INPLACE_X:
        y = x
    else:
        y = TestTensor(shape, x_stride, dtype, device)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateCausalSoftmaxDescriptor(
            handle, ctypes.byref(descriptor), y.descriptor, x.descriptor
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x.destroy_desc()
    y.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetCausalSoftmaxWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    def lib_causal_softmax():
        check_error(
            LIBINFINIOP.infiniopCausalSoftmax(
                descriptor,
                workspace.data(),
                workspace_size.value,
                y.data(),
                x.data(),
                None,
            )
        )

    lib_causal_softmax()

    if sync is not None:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), ans, atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), ans, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: causal_softmax(x.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_causal_softmax(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyCausalSoftmaxDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
