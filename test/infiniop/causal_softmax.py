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
_TENSOR_DTYPES = [torch.float16, torch.float32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    torch.float16: {"atol": 1e-3, "rtol": 1e-2},
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


class CausalSoftmaxDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopCausalSoftmaxDescriptor_t = POINTER(CausalSoftmaxDescriptor)


def causal_softmax(x):
    type = x.dtype
    mask = torch.tril(torch.ones_like(x), diagonal=-1).flip(dims=[-2, -1])
    masked = torch.where(mask == 1, -torch.inf, x.to(torch.float32))
    return torch.nn.functional.softmax(masked, dim=-1, dtype=type)


def test(
    lib,
    handle,
    torch_device,
    shape,
    x_stride=None,
    y_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=torch.float16,
    sync=None
):
    print(
        f"Testing CausalSoftmax on {torch_device} with shape:{shape} x_stride:{x_stride} y_stride:{y_stride} dtype:{dtype} inplace:{inplace}"
    )

    x = torch.rand(shape, dtype=dtype).to(torch_device)
    mask = torch.tril(torch.ones_like(x), diagonal=-1).flip(dims=[-2, -1])
    x = torch.where(mask == 1, torch.full_like(x, torch.finfo(x.dtype).max), x)
    ans = causal_softmax(x)

    x = rearrange_if_needed(x, x_stride)

    x_tensor = to_tensor(x, lib)

    if inplace == Inplace.INPLACE_X:
        y = x
        y_tensor = x_tensor
    else:
        y = torch.zeros(shape, dtype=dtype).to(torch_device)
        y = rearrange_if_needed(y, y_stride)
        y_tensor = to_tensor(y, lib)
        
    if sync is not None:
        sync()

    descriptor = infiniopCausalSoftmaxDescriptor_t()
    check_error(
        lib.infiniopCreateCausalSoftmaxDescriptor(
            handle, ctypes.byref(descriptor), y_tensor.descriptor, x_tensor.descriptor
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x_tensor.destroyDesc(lib)

    workspace_size = c_uint64(0)
    check_error(
        lib.infiniopGetCausalSoftmaxWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = create_workspace(workspace_size.value, x.device)

    def lib_causal_softmax():
        check_error(
            lib.infiniopCausalSoftmax(
                descriptor,
                workspace.data_ptr() if workspace is not None else None,
                workspace_size.value,
                y_tensor.data,
                x_tensor.data,
                None,
            )
        )

    lib_causal_softmax()
    
    if sync is not None:
        sync() 

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y, ans, atol=atol, rtol=rtol)
    assert torch.allclose(y, ans, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: causal_softmax(x), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_causal_softmax(), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(lib.infiniopDestroyCausalSoftmaxDescriptor(descriptor))


if __name__ == "__main__":
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

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(lib, device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
