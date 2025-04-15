import torch
import ctypes
from ctypes import POINTER, Structure, c_int32, c_void_p, c_uint64
from libinfiniop import (
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    open_lib,
    to_tensor,
    get_test_devices,
    check_error,
    rearrange_if_needed,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    create_workspace
)
from enum import Enum, auto

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # shape, a_stride, b_stride, c_stride
    ((13, 4), None, None, None),
    ((13, 4), (10, 1), (10, 1), (10, 1)),
    ((13, 4), (0, 1), None, None),
    ((13, 4, 4), None, None, None),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1), (20, 4, 1)),
    ((13, 4, 4), (4, 0, 1), (0, 4, 1), None),
    ((16, 5632), None, None, None),
    ((16, 5632), (13312, 1), (13312, 1), (13312, 1)),
    ((4, 4, 5632), None, None, None),
    ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), (45056, 5632, 1)),
]

class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_A = auto()
    INPLACE_B = auto()


# Inplace options applied for each test case in _TEST_CASES_
_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE_A,
    Inplace.INPLACE_B,
]

# Form the test cases by appending each element of _INPLACE to each tuple in _TEST_CASES_
_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

# Data types used for testing
_TENSOR_DTYPES = [torch.float16, torch.float32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    torch.float16: {"atol": 1e-4, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


class SwiGLUDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopSwiGLUDescriptor_t = POINTER(SwiGLUDescriptor)


def swiglu(a, b):
    return a * b / (1 + torch.exp(-b.float()).to(b.dtype))
    


def process_tensors(c, c_strides, a, a_stride, b, b_stride, inplace):
    """
    rearrange the tensors if needed and apply the inplace config.
    if inplace is true and the output (i.e., c) is placed to the broadcasted input,
    the inplace config is ignored and out-of-place is used
    """
    original_c_strides = c_strides if c_strides else c.stride()

    def _rearrange(tensor, strides):
        if strides and 0 in strides:
            tensor.set_(tensor.untyped_storage(), 0, tensor.shape, strides)
            return tensor
        else:
            return rearrange_if_needed(tensor, strides)

    a, b, c = [
        _rearrange(tensor, stride)
        for tensor, stride in zip([a, b, c], [a_stride, b_stride, c_strides])
    ]
    c = (
        c
        if inplace == Inplace.OUT_OF_PLACE
        else (a if inplace == Inplace.INPLACE_A else b)
    )
    # if inplace is true and c has broadcasted config, reset it to the original unbroadcasted strides
    if 0 in c.stride():
        c.set_(c.untyped_storage(), 0, c.shape, original_c_strides)

    return a, b, c


def test(
    lib,
    handle,
    torch_device,
    shape,
    a_stride=None,
    b_stride=None,
    c_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=torch.float16,
    sync=None,
):
    print(
        f"Testing SwiGLU on {torch_device} with shape:{shape} a_stride:{a_stride} b_stride:{b_stride} c_stride:{c_stride} "
        f"dtype:{dtype} inplace:{inplace}"
    )

    a = torch.rand(shape, dtype=dtype).to(torch_device)
    b = torch.rand(shape, dtype=dtype).to(torch_device)
    c = torch.rand(shape, dtype=dtype).to(torch_device)
    a, b, c = process_tensors(c, c_stride, a, a_stride, b, b_stride, inplace)

    ans = swiglu(a, b)

    a_tensor, b_tensor = [to_tensor(tensor, lib) for tensor in [a, b]]
    c_tensor = (
        to_tensor(c, lib)
        if inplace == Inplace.OUT_OF_PLACE
        else (a_tensor if inplace == Inplace.INPLACE_A else b_tensor)
    )
    if sync is not None:
        sync()

    descriptor = infiniopSwiGLUDescriptor_t()
    check_error(
        lib.infiniopCreateSwiGLUDescriptor(
            handle,
            ctypes.byref(descriptor),
            c_tensor.descriptor,
            a_tensor.descriptor,
            b_tensor.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [a_tensor, b_tensor, c_tensor]:
        tensor.destroyDesc(lib)

    workspace_size = c_uint64(0)
    check_error(
        lib.infiniopGetSwiGLUWorkspaceSize(descriptor, ctypes.byref(workspace_size))
    )
    workspace = create_workspace(workspace_size.value, c.device)

    def lib_swiglu():
        check_error(
            lib.infiniopSwiGLU(
                descriptor, 
                workspace.data_ptr() if workspace is not None else None,
                workspace_size.value,
                c_tensor.data, a_tensor.data, b_tensor.data, None
            )
        )

    lib_swiglu()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(c, ans, atol=atol, rtol=rtol)
    assert torch.allclose(c, ans, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: swiglu(a, b), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_swiglu(), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(lib.infiniopDestroySwiGLUDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    lib = open_lib()

    lib.infiniopCreateSwiGLUDescriptor.restype = c_int32
    lib.infiniopCreateSwiGLUDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopSwiGLUDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetSwiGLUWorkspaceSize.restype = c_int32
    lib.infiniopGetSwiGLUWorkspaceSize.argtypes = [
        infiniopSwiGLUDescriptor_t,
        POINTER(c_uint64),
    ]

    lib.infiniopSwiGLU.restype = c_int32
    lib.infiniopSwiGLU.argtypes = [
        infiniopSwiGLUDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroySwiGLUDescriptor.restype = c_int32
    lib.infiniopDestroySwiGLUDescriptor.argtypes = [
        infiniopSwiGLUDescriptor_t,
    ]

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(lib, device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
