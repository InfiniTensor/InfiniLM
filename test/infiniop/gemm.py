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
    # alpha, beta, a_shape, b_shape, c_shape, a_stride, b_stride, c_stride
    (1.0, 0.0, (1, 2048), (2048, 2048), (1, 2048), None, None, None),
    (1.0, 0.0, (1, 2048), (2048, 2048), (1, 2048), None, None, None),
    (1.0, 0.0, (2, 4, 2048), (2, 2048, 2048), (2, 4, 2048), None, None, None),
    (1.0, 0.0, (2, 4, 2048), (2, 2048, 2048), (2, 4, 2048), None, None, None),
    (1.0, 0.0, (1, 2048), (2048, 2048), (1, 2048), (4096, 1), (4096, 1), (4096, 1)),
    (1.0, 0.0, (1, 2048), (2048, 2048), (1, 2048), (4096, 1), (4096, 1), (4096, 1)),
    (1.0, 1.0, (6, 2048), (2048, 2560), (6, 2560), (2048, 1), (1, 2048), (2560, 1)),
    (1.0, 1.0, (6, 2048), (2048, 2560), (6, 2560), (2048, 1), (1, 2048), (2560, 1)),
    (1.0 / 8.0, 0.0, (4, 8 * 6, 64), (4, 64, 6), (4, 8 * 6, 6), None, None, None),
    (1.0 / 8.0, 0.0, (4, 8 * 6, 64), (4, 64, 6), (4, 8 * 6, 6), None, None, None),
]

# Data types used for testing
_TENSOR_DTYPES = [torch.float16, torch.float32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    torch.float16: {"atol": 0, "rtol": 1e-2},
    torch.float32: {"atol": 0, "rtol": 1e-3},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


# ==============================================================================
#  Definitions
# ==============================================================================
class GemmDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopGemmDescriptor_t = POINTER(GemmDescriptor)


# PyTorch implementation for matrix multiplication
def gemm(_c, beta, _a, _b, alpha):
    a, b, c = _a.clone(), _b.clone(), _c.clone()
    result_dtype = c.dtype
    fp32_result = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    return alpha * fp32_result.to(result_dtype) + beta * c


# The argument list should be (lib, handle, torch_device, <param list>, dtype)
# The <param list> should keep the same order as the one specified in _TEST_CASES
def test(
    lib,
    handle,
    torch_device,
    alpha,
    beta,
    a_shape,
    b_shape,
    c_shape,
    a_stride=None,
    b_stride=None,
    c_stride=None,
    dtype=torch.float16,
    sync=None
):
    print(
        f"Testing Gemm on {torch_device} with alpha:{alpha}, beta:{beta},"
        f" a_shape:{a_shape}, b_shape:{b_shape}, c_shape:{c_shape},"
        f" a_stride:{a_stride}, b_stride:{b_stride}, c_stride:{c_stride}, dtype:{dtype}"
    )

    # Initialize tensors
    a = torch.rand(a_shape, dtype=dtype).to(torch_device)
    b = torch.rand(b_shape, dtype=dtype).to(torch_device)
    c = torch.ones(c_shape, dtype=dtype).to(torch_device)

    # Compute the PyTorch reference result
    ans = gemm(c, beta, a, b, alpha)

    a, b, c = [
        rearrange_if_needed(tensor, stride)
        for tensor, stride in zip([a, b, c], [a_stride, b_stride, c_stride])
    ]
    a_tensor, b_tensor, c_tensor = [to_tensor(tensor, lib) for tensor in [a, b, c]]

    if sync is not None:
        sync()

    descriptor = infiniopGemmDescriptor_t()
    check_error(
        lib.infiniopCreateGemmDescriptor(
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

    # Get workspace size and create workspace
    workspace_size = c_uint64(0)
    check_error(
        lib.infiniopGetGemmWorkspaceSize(descriptor, ctypes.byref(workspace_size))
    )
    workspace = create_workspace(workspace_size.value, a.device)

    # Execute infiniop gemm operator
    def lib_gemm():
        check_error(
            lib.infiniopGemm(
                descriptor,
                workspace.data_ptr() if workspace is not None else None,
                workspace_size.value,
                c_tensor.data,
                a_tensor.data,
                b_tensor.data,
                alpha,
                beta,
                None,
            )
        )

    lib_gemm()

    # Validate results
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(c, ans, atol=atol, rtol=rtol)
    assert torch.allclose(c, ans, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: gemm(c, beta, a, b, alpha), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_gemm(), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(lib.infiniopDestroyGemmDescriptor(descriptor))


# ==============================================================================
#  Main Execution
# ==============================================================================
if __name__ == "__main__":
    args = get_args()
    lib = open_lib()

    lib.infiniopCreateGemmDescriptor.restype = c_int32
    lib.infiniopCreateGemmDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopGemmDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetGemmWorkspaceSize.restype = c_int32
    lib.infiniopGetGemmWorkspaceSize.argtypes = [
        infiniopGemmDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopGemm.restype = c_int32
    lib.infiniopGemm.argtypes = [
        infiniopGemmDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
        c_float,
        c_float,
        c_void_p,
    ]

    lib.infiniopDestroyGemmDescriptor.restype = c_int32
    lib.infiniopDestroyGemmDescriptor.argtypes = [
        infiniopGemmDescriptor_t,
    ]

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(lib, device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
