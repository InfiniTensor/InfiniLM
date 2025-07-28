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
	InfiniDeviceEnum,
    infiniopOperatorDescriptor_t,
)

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES = [
    # num_tokens, k, hidden_dim
    (16, 4, 128),
    (32, 2, 256),
    (1, 8, 512),
    (64, 8, 1024),
]

# The current CUDA implementation for moe_combine only supports F32
_TENSOR_DTYPES = [InfiniDtype.F32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def moe_combine_pytorch(permuted_input, gating_weights, aux_info, output):
    """
    PyTorch reference implementation for the MoE combine operation.
    """
    output.zero_()
    num_tokens_mul_k, _ = permuted_input.shape

    original_token_indices = aux_info[:, 0].long()
    gating_val_indices = aux_info[:, 1].long()
    
    num_tokens, k = gating_weights.shape
    gating_val_indices_row = gating_val_indices // k
    gating_val_indices_col = gating_val_indices % k

    weights = gating_weights[gating_val_indices_row, gating_val_indices_col]
    weights = weights.unsqueeze(1)

    output.index_add_(0, original_token_indices, permuted_input * weights)


def test(
    handle,
    device,
    num_tokens,
    k,
    hidden_dim,
    dtype=InfiniDtype.F32,
    sync=None,
):
    print(
        f"Testing MoECombine on {InfiniDeviceNames[device]} with num_tokens:{num_tokens} k:{k} hidden_dim:{hidden_dim} "
        f"dtype:{InfiniDtypeNames[dtype]}"
    )

    permuted_input = TestTensor(
        (num_tokens * k, hidden_dim), None, dtype, device, scale=0.1
    )
    gating_weights = TestTensor((num_tokens, k), None, dtype, device, scale=0.1)

    # Generate plausible aux_info based on a simulated TopK output
    original_token_pos = torch.arange(num_tokens).repeat_interleave(k)
    gating_val_pos = torch.arange(num_tokens * k)
    
    aux_info_torch = torch.stack([original_token_pos, gating_val_pos], dim=1)

    aux_info = TestTensor.from_torch(aux_info_torch, InfiniDtype.I32, device)
    
    output = TestTensor((num_tokens, hidden_dim), None, dtype, device, mode="zeros")
    ans = torch.zeros_like(output.torch_tensor())

    moe_combine_pytorch(
        permuted_input.torch_tensor(),
        gating_weights.torch_tensor(),
        aux_info.torch_tensor(),
        ans,
    )

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    
    # NOTE: The public API name might differ. Using infiniopCreateMoECombineDescriptor as a placeholder.
    check_error(
        LIBINFINIOP.infiniopCreateMoECombineDescriptor(
            handle,
            ctypes.byref(descriptor),
            permuted_input.descriptor,
            gating_weights.descriptor,
            aux_info.descriptor,
            output.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [permuted_input, gating_weights, aux_info, output]:
        tensor.destroy_desc()
    
    # MoECombine has no workspace.
    
    def lib_moe_combine():
        check_error(
            LIBINFINIOP.infiniopMoECombine(
                descriptor,
                output.data(),
                permuted_input.data(),
                gating_weights.data(),
                aux_info.data(),
                None,
            )
        )

    lib_moe_combine()
    
    if sync is not None:
        sync()
    
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(output.actual_tensor(), ans, atol=atol, rtol=rtol)
    assert torch.allclose(output.actual_tensor(), ans, atol=atol, rtol=rtol)

    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: moe_combine_pytorch(
                permuted_input.torch_tensor(),
                gating_weights.torch_tensor(),
                aux_info.torch_tensor(),
                ans,
            ),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib", lambda: lib_moe_combine(), device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroyMoECombineDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # only support nvidia now
    for device in get_test_devices(args):
        if device == InfiniDeviceEnum.NVIDIA:
            test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m") 