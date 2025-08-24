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
_TEST_CASES = [
    # batch_size, in_features, out_features, has_bias
    (1, 4, 2, True),
    (2, 8, 4, True),
    (1, 4, 2, False),
    (4, 16, 8, True),
    (8, 32, 16, False),
    (16, 128, 64, True),
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-2},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-3},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 5e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


# PyTorch implementation for linear
def linear_pytorch(input_tensor, weight_tensor, bias_tensor=None):
    return torch.nn.functional.linear(input_tensor, weight_tensor, bias_tensor)


# The argument list should be (lib, handle, torch_device, <param list>, dtype)
# The <param list> should keep the same order as the one specified in _TEST_CASES
def test(
    handle,
    device,
    batch_size,
    in_features,
    out_features,
    has_bias,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing Linear on {InfiniDeviceNames[device]} with batch_size:{batch_size}, "
        f"in_features:{in_features}, out_features:{out_features}, has_bias:{has_bias}, dtype:{InfiniDtypeNames[dtype]}"
    )

    # Initialize tensors
    input_shape = (batch_size, in_features)
    weight_shape = (out_features, in_features)
    output_shape = (batch_size, out_features)
    bias_shape = (out_features,) if has_bias else None

    input_tensor = TestTensor(input_shape, None, dtype, device)
    weight_tensor = TestTensor(weight_shape, None, dtype, device)
    output_tensor = TestTensor(output_shape, None, dtype, device, mode="zeros")
    ans_tensor = TestTensor(output_shape, None, dtype, device, mode="zeros")
    
    bias_tensor = TestTensor(bias_shape, None, dtype, device) if has_bias else None

    # Compute the PyTorch reference result
    def torch_linear():
        bias_torch = bias_tensor.torch_tensor() if has_bias else None
        result = linear_pytorch(
            input_tensor.torch_tensor(),
            weight_tensor.torch_tensor(),
            bias_torch
        )
        ans_tensor.torch_tensor().copy_(result)

    torch_linear()

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    bias_desc = bias_tensor.descriptor if has_bias else None
    check_error(
        LIBINFINIOP.infiniopCreateLinearDescriptor(
            handle,
            ctypes.byref(descriptor),
            output_tensor.descriptor,
            input_tensor.descriptor,
            weight_tensor.descriptor,
            bias_desc,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [input_tensor, weight_tensor, output_tensor]:
        tensor.destroy_desc()
    if has_bias:
        bias_tensor.destroy_desc()

    # Get workspace size and create workspace
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetLinearWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    # Execute infiniop linear operator
    def lib_linear():
        bias_data = bias_tensor.data() if has_bias else None
        check_error(
            LIBINFINIOP.infiniopLinear(
                descriptor,
                workspace.data(),
                workspace_size.value,
                output_tensor.data(),
                input_tensor.data(),
                weight_tensor.data(),
                bias_data,
                None,
            )
        )

    lib_linear()

    # Validate results
    if sync is not None:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    torch.testing.assert_close(
        output_tensor.torch_tensor(),
        ans_tensor.torch_tensor(),
        atol=atol,
        rtol=rtol,
    )

    # Profile operation if enabled
    if PROFILE:
        profile_operation(
            torch_linear, lib_linear, NUM_PRERUN, NUM_ITERATIONS, sync
        )

    # Clean up
    check_error(LIBINFINIOP.infiniopDestroyLinearDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)