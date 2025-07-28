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
    # num_tokens, k, hidden_dim, num_experts
    (16, 2, 128, 8),
    (32, 2, 256, 16),
    (1, 8, 512, 16),
    (64, 4, 1024, 32),
]

# The current CUDA implementation for moe_dispatch only supports F32
_TENSOR_DTYPES = [InfiniDtype.F32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def check_dispatch_output(
    input_tensor, indices, permuted_output, aux_info, num_experts
):
    """
    Checks the output of the MoE dispatch operation, accounting for the
    non-deterministic nature of the CUDA kernel.
    """
    num_tokens, hidden_dim = input_tensor.shape
    _, k = indices.shape

    # Number of tokens that are actually dispatched (expert index >= 0)
    num_dispatched_tokens = torch.sum(indices >= 0).item()

    # Sliced views of the output tensors for dispatched tokens
    dispatched_permuted_output = permuted_output[:num_dispatched_tokens]
    dispatched_aux_info = aux_info[:num_dispatched_tokens]

    # --- Check 1: Data Consistency ---
    # The data in permuted_output should correspond to the correct original token's data.
    original_token_indices = dispatched_aux_info[:, 0].long()
    expected_permuted_output = input_tensor[original_token_indices]
    assert torch.allclose(
        dispatched_permuted_output, expected_permuted_output
    ), "Data consistency check failed: permuted output data does not match input data."

    # --- Check 2: Expert Assignment Correctness ---
    # The set of tokens dispatched to each expert must be correct.

    # 2a. Build the expected mapping from expert to a list of original token indices
    # from the input 'indices' tensor.
    expected_expert_to_tokens = {i: [] for i in range(num_experts)}
    for token_idx in range(num_tokens):
        for i in range(k):
            expert_idx = indices[token_idx, i].item()
            if expert_idx >= 0:
                expected_expert_to_tokens[expert_idx].append(token_idx)

    # 2b. Build the actual mapping from the kernel's output 'aux_info'.
    actual_expert_to_tokens = {i: [] for i in range(num_experts)}
    original_gating_pos = dispatched_aux_info[:, 1].long()
    gating_pos_token_idx = original_gating_pos // k
    gating_pos_k_idx = original_gating_pos % k
    experts_for_dispatched = indices[gating_pos_token_idx, gating_pos_k_idx]

    for i in range(num_dispatched_tokens):
        expert_idx = experts_for_dispatched[i].item()
        token_idx = original_token_indices[i].item()
        actual_expert_to_tokens[expert_idx].append(token_idx)

    # 2c. Compare the sorted lists of token indices for each expert.
    for expert_idx in range(num_experts):
        assert sorted(expected_expert_to_tokens[expert_idx]) == sorted(
            actual_expert_to_tokens[expert_idx]
        ), f"Token set for expert {expert_idx} does not match."

    return True


def test(
    handle,
    device,
    num_tokens,
    k,
    hidden_dim,
    num_experts,
    dtype=InfiniDtype.F32,
    sync=None,
):
    print(
        f"Testing MoEDispatch on {InfiniDeviceNames[device]} with num_tokens:{num_tokens} k:{k} "
        f"hidden_dim:{hidden_dim} num_experts:{num_experts} dtype:{InfiniDtypeNames[dtype]}"
    )

    input_tensor = TestTensor(
        (num_tokens, hidden_dim), None, dtype, device, scale=0.1
    )
    # Generate random expert assignments for each token's top-k choices
    # A small chance of -1 to test invalid expert handling
    indices_torch = torch.randint(
        -1, num_experts, (num_tokens, k), dtype=torch.int32
    )
    indices = TestTensor.from_torch(indices_torch, InfiniDtype.I32, device)

    permuted_output = TestTensor(
        (num_tokens * k, hidden_dim), None, dtype, device, mode="zeros"
    )
    aux_info = TestTensor((num_tokens * k, 2), None, InfiniDtype.I32, device, mode="zeros")

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()

    check_error(
        LIBINFINIOP.infiniopCreateMoEDispatchDescriptor(
            handle,
            ctypes.byref(descriptor),
            ctypes.c_int(num_experts),
            input_tensor.descriptor,
            indices.descriptor,
            permuted_output.descriptor,
            aux_info.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [input_tensor, indices, permuted_output, aux_info]:
        tensor.destroy_desc()

    # MoEDispatch has no workspace.
    
    def lib_moe_dispatch():
        check_error(
            LIBINFINIOP.infiniopMoEDispatch(
                descriptor,
                input_tensor.data(),
                indices.data(),
                permuted_output.data(),
                aux_info.data(),
                None,
            )
        )

    lib_moe_dispatch()
    
    if sync is not None:
        sync()

    assert check_dispatch_output(
        input_tensor.torch_tensor(),
        indices.torch_tensor(),
        permuted_output.actual_tensor(),
        aux_info.actual_tensor(),
        num_experts,
    )

    if PROFILE:
        profile_operation(
            "    lib", lambda: lib_moe_dispatch(), device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroyMoEDispatchDescriptor(descriptor))


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