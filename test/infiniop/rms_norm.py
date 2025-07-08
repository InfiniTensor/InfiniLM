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

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # y_shape, x_shape, w_shape, y_stride, x_stride
    ((1, 4), (1, 4), (4,), None, None),
    ((1, 4), (1, 4), (4,), None, None),
    ((16, 2048), (16, 2048), (2048,), None, None),
    ((16, 2048), (16, 2048), (2048,), None, None),
    ((16, 2048), (16, 2048), (2048,), (4096, 1), (4096, 1)),
    ((16, 2048), (16, 2048), (2048,), (4096, 1), (4096, 1)),
]

# w (weight) types
# Note: 'None' means the same as input dtype
_WEIGHT_DTYPES = [None, InfiniDtype.F32]
# x types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16]

# Form the test cases by appending each element of _WEIGHT_DTYPES to each tuple in _TEST_CASES_
_TEST_CASES = [
    test_case + (w_dtype,) for test_case in _TEST_CASES_ for w_dtype in _WEIGHT_DTYPES
]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 2e-3, "rtol": 2e-3},
    InfiniDtype.BF16: {"atol": 8e-3, "rtol": 8e-3},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def rms_norm(ans, x, w, eps):
    torch.pow(x, 2, out=ans)
    mean = torch.mean(ans, dim=-1, keepdim=True)
    mean.add_(eps)
    torch.rsqrt(mean, out=mean)
    torch.mul(x, mean, out=ans)
    ans.mul_(w)


def test(
    handle,
    device,
    y_shape,
    x_shape,
    w_shape,
    y_stride,
    x_stride,
    w_dtype=InfiniDtype.F32,
    dtype=InfiniDtype.F16,
    sync=None,
):
    w_dtype = w_dtype if w_dtype else dtype
    print(
        f"Testing RMS_Norm on {InfiniDeviceNames[device]} with y_shape:{y_shape} x_shape:{x_shape} w_shape:{w_shape}"
        f" y_stride:{y_stride} x_stride:{x_stride} w_dtype:{InfiniDtypeNames[w_dtype]} dtype:{InfiniDtypeNames[dtype]}"
    )

    y = TestTensor(y_shape, y_stride, dtype, device, mode="ones")
    x = TestTensor(x_shape, x_stride, dtype, device, scale=0.01)
    w = TestTensor(w_shape, None, w_dtype, device)

    eps = 1e-6
    rms_norm(y.torch_tensor(), x.torch_tensor(), w.torch_tensor(), eps)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()

    check_error(
        LIBINFINIOP.infiniopCreateRMSNormDescriptor(
            handle,
            ctypes.byref(descriptor),
            y.descriptor,
            x.descriptor,
            w.descriptor,
            eps,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [x, y, w]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetRMSNormWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, y.device)

    def lib_rms_norm():
        check_error(
            LIBINFINIOP.infiniopRMSNorm(
                descriptor,
                workspace.data(),
                workspace_size.value,
                y.data(),
                x.data(),
                w.data(),
                None,
            )
        )

    lib_rms_norm()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: rms_norm(y.torch_tensor(), x.torch_tensor(), w.torch_tensor(), eps), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_rms_norm(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyRMSNormDescriptor(descriptor))


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
