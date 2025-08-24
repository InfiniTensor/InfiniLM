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


# PyTorch implementation for linear backwards
def linear_backwards_pytorch(grad_output, input_tensor, weight_tensor, bias_tensor=None):
    """Compute gradients using PyTorch autograd"""
    
    # Enable gradients
    input_tensor = input_tensor.clone().detach().requires_grad_(True)
    weight_tensor = weight_tensor.clone().detach().requires_grad_(True)
    if bias_tensor is not None:
        bias_tensor = bias_tensor.clone().detach().requires_grad_(True)
    
    # Forward pass
    output = torch.nn.functional.linear(input_tensor, weight_tensor, bias_tensor)
    
    # Backward pass
    output.backward(grad_output)
    
    grad_input = input_tensor.grad
    grad_weight = weight_tensor.grad
    grad_bias = bias_tensor.grad if bias_tensor is not None else None
    
    return grad_input, grad_weight, grad_bias


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
        f"Testing LinearBackwards on {InfiniDeviceNames[device]} with batch_size:{batch_size}, "
        f"in_features:{in_features}, out_features:{out_features}, has_bias:{has_bias}, dtype:{InfiniDtypeNames[dtype]}"
    )

    # Initialize tensors
    input_shape = (batch_size, in_features)
    weight_shape = (out_features, in_features)
    output_shape = (batch_size, out_features)
    bias_shape = (out_features,) if has_bias else None

    # Forward pass tensors
    input_tensor = TestTensor(input_shape, None, dtype, device)
    weight_tensor = TestTensor(weight_shape, None, dtype, device)
    bias_tensor = TestTensor(bias_shape, None, dtype, device) if has_bias else None
    
    # Gradient tensors
    grad_output_tensor = TestTensor(output_shape, None, dtype, device)
    grad_input_tensor = TestTensor(input_shape, None, dtype, device, mode="zeros")
    grad_weight_tensor = TestTensor(weight_shape, None, dtype, device, mode="zeros")
    grad_bias_tensor = TestTensor(bias_shape, None, dtype, device, mode="zeros") if has_bias else None
    
    # Reference tensors for PyTorch computation
    ans_grad_input = TestTensor(input_shape, None, dtype, device, mode="zeros")
    ans_grad_weight = TestTensor(weight_shape, None, dtype, device, mode="zeros")
    ans_grad_bias = TestTensor(bias_shape, None, dtype, device, mode="zeros") if has_bias else None

    # Compute the PyTorch reference result
    def torch_linear_backwards():
        bias_torch = bias_tensor.torch_tensor() if has_bias else None
        grad_input_ref, grad_weight_ref, grad_bias_ref = linear_backwards_pytorch(
            grad_output_tensor.torch_tensor(),
            input_tensor.torch_tensor(),
            weight_tensor.torch_tensor(),
            bias_torch
        )
        
        ans_grad_input.torch_tensor().copy_(grad_input_ref)
        ans_grad_weight.torch_tensor().copy_(grad_weight_ref)
        if has_bias and grad_bias_ref is not None:
            ans_grad_bias.torch_tensor().copy_(grad_bias_ref)

    torch_linear_backwards()

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    bias_desc = bias_tensor.descriptor if has_bias else None
    grad_bias_desc = grad_bias_tensor.descriptor if has_bias else None
    
    check_error(
        LIBINFINIOP.infiniopCreateLinearBackwardsDescriptor(
            handle,
            ctypes.byref(descriptor),
            grad_input_tensor.descriptor,
            grad_weight_tensor.descriptor,
            grad_bias_desc,
            grad_output_tensor.descriptor,
            input_tensor.descriptor,
            weight_tensor.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [input_tensor, weight_tensor, grad_output_tensor, grad_input_tensor, grad_weight_tensor]:
        tensor.destroy_desc()
    if has_bias:
        bias_tensor.destroy_desc()
        grad_bias_tensor.destroy_desc()

    # Get workspace size and create workspace
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetLinearBackwardsWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    # Execute infiniop linear backwards operator
    def lib_linear_backwards():
        bias_data = bias_tensor.data() if has_bias else None
        grad_bias_data = grad_bias_tensor.data() if has_bias else None
        check_error(
            LIBINFINIOP.infiniopLinearBackwards(
                descriptor,
                workspace.data(),
                workspace_size.value,
                grad_input_tensor.data(),
                grad_weight_tensor.data(),
                grad_bias_data,
                grad_output_tensor.data(),
                input_tensor.data(),
                weight_tensor.data(),
                None,
            )
        )

    lib_linear_backwards()

    # Validate results
    if sync is not None:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    
    # Check grad_input
    torch.testing.assert_close(
        grad_input_tensor.torch_tensor(),
        ans_grad_input.torch_tensor(),
        atol=atol,
        rtol=rtol,
    )
    
    # Check grad_weight
    torch.testing.assert_close(
        grad_weight_tensor.torch_tensor(),
        ans_grad_weight.torch_tensor(),
        atol=atol,
        rtol=rtol,
    )
    
    # Check grad_bias if present
    if has_bias:
        torch.testing.assert_close(
            grad_bias_tensor.torch_tensor(),
            ans_grad_bias.torch_tensor(),
            atol=atol,
            rtol=rtol,
        )

    # Profile operation if enabled
    if PROFILE:
        profile_operation(
            torch_linear_backwards, lib_linear_backwards, NUM_PRERUN, NUM_ITERATIONS, sync
        )

    # Clean up
    check_error(LIBINFINIOP.infiniopDestroyLinearBackwardsDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)