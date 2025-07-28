import torch
import ctypes
from ctypes import c_uint64, c_int32
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

# ==============================================================================
# Strategies Enum based on src/infiniop/ops/topk/info.h
DEEPSEEK_V3 = 0
STANDARD_SOFTMAX = 1

# Test cases: (shape, k, strategy, n_group, topk_group, has_bias)
_TEST_CASES = [
    # STANDARD_SOFTMAX cases (n_group and topk_group are unused, pass 0)
    ((16, 128), 4, STANDARD_SOFTMAX, 0, 0, False),
    ((16, 128), 4, STANDARD_SOFTMAX, 0, 0, True),
    ((1, 256), 8, STANDARD_SOFTMAX, 0, 0, True),
    ((32, 64), 64, STANDARD_SOFTMAX, 0, 0, False), # k=num_experts
    # DEEPSEEK_V3 cases
    ((16, 128), 4, DEEPSEEK_V3, 8, 2, False),
    ((16, 128), 4, DEEPSEEK_V3, 8, 2, True),
    ((32, 256), 8, DEEPSEEK_V3, 16, 4, True),
    ((1, 64), 2, DEEPSEEK_V3, 4, 1, False),
]

# x types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-2, "rtol": 1e-2},
    InfiniDtype.BF16: {"atol": 1e-1, "rtol": 1e-1},
    InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

# ==============================================================================
#  Reference Implementations
# ==============================================================================
def standard_topk_router_pytorch(logits, k, bias=None):
    if bias is not None:
        logits = logits + bias
    scores = torch.softmax(logits.float(), dim=-1).to(logits.dtype)
    topk_weights, topk_indices = torch.topk(scores, k, dim=-1)
    return topk_weights, topk_indices.int()


def deepseek_v3_topk_router_pytorch(logits, k, n_group, topk_group, bias=None):
    if bias is not None:
        logits = logits + bias
    scores = torch.sigmoid(logits.float()).to(logits.dtype)

    num_tokens, num_experts = scores.shape
    experts_per_group = num_experts // n_group
    
    grouped_scores = scores.view(num_tokens, n_group, experts_per_group)
    top2_scores, _ = torch.topk(grouped_scores, 2, dim=-1)
    group_scores = top2_scores.sum(dim=-1)

    _, topk_group_indices = torch.topk(group_scores, topk_group, dim=-1)

    mask = torch.zeros_like(scores, dtype=torch.bool)
    # This loop is slow but clear for a reference implementation.
    for i in range(num_tokens):
        for group_idx in topk_group_indices[i]:
            start = group_idx * experts_per_group
            end = start + experts_per_group
            mask[i, start:end] = True

    masked_scores = scores.masked_fill(~mask, -float("inf"))
    _, topk_indices = torch.topk(masked_scores, k, dim=-1)
    
    topk_weights_unnormalized = torch.gather(scores, 1, topk_indices)
    
    norm_sum = topk_weights_unnormalized.sum(dim=-1, keepdim=True)
    norm_sum = torch.where(norm_sum > 1e-20, norm_sum, torch.ones_like(norm_sum))
    topk_weights_normalized = topk_weights_unnormalized / norm_sum
    
    return topk_weights_normalized, topk_indices.int()

# ==============================================================================
#  Test Function
# ==============================================================================
def test(
    handle,
    device,
    x_shape,
    k,
    strategy,
    n_group,
    topk_group,
    has_bias,
    dtype=InfiniDtype.F16,
    sync=None,
):
    strategy_name = "DEEPSEEK_V3" if strategy == DEEPSEEK_V3 else "STANDARD_SOFTMAX"
    print(
        f"Testing TopK ({strategy_name}) on {InfiniDeviceNames[device]} with x_shape:{x_shape} k:{k} dtype:{InfiniDtypeNames[dtype]} has_bias:{has_bias}"
    )

    num_tokens, num_experts = x_shape
    output_shape = (num_tokens, k)

    x = TestTensor(x_shape, None, dtype, device)
    bias = TestTensor((num_experts,), None, dtype, device) if has_bias else None
    output_val = TestTensor(output_shape, None, dtype, device, mode="zeros")
    output_ind = TestTensor(output_shape, None, InfiniDtype.I32, device, mode="zeros")

    if strategy == DEEPSEEK_V3:
        torch_values, torch_indices = deepseek_v3_topk_router_pytorch(
            x.torch_tensor(), k, n_group, topk_group, bias.torch_tensor() if bias else None
        )
    else: # STANDARD_SOFTMAX
        torch_values, torch_indices = standard_topk_router_pytorch(
            x.torch_tensor(), k, bias.torch_tensor() if bias else None
        )

    if sync is not None:
        sync()

    op_desc = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateTopKDescriptor(
            handle,
            ctypes.byref(op_desc),
            x.desc(),
            output_val.desc(),
            output_ind.desc(),
            bias.desc() if bias else None,
            c_int32(k),
            c_int32(strategy),
            c_int32(n_group),
            c_int32(topk_group),
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    tensors_to_invalidate = [x, output_val, output_ind]
    if has_bias:
        tensors_to_invalidate.append(bias)
    for tensor in tensors_to_invalidate:
        tensor.destroy_desc()

    workspace_size = LIBINFINIOP.infiniopGetTopKWorkspaceSize(op_desc)
    workspace = TestWorkspace(workspace_size, x.device)

    def lib_topk():
        check_error(
            LIBINFINIOP.infiniopTopKCalculate(
                op_desc,
                x.data(),
                output_val.data(),
                output_ind.data(),
                bias.data() if bias else None,
                workspace.data(),
                None, # stream
            )
        )

    lib_topk()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        print("Values:")
        debug(output_val.actual_tensor(), torch_values, atol=atol, rtol=rtol)
        print("Indices:")
        debug(output_ind.actual_tensor(), torch_indices, atol=0, rtol=0)

    assert torch.allclose(output_val.actual_tensor(), torch_values, atol=atol, rtol=rtol)
    assert torch.equal(output_ind.actual_tensor(), torch_indices.to(torch.int32))

    # Profiling workflow
    if PROFILE:
        # fmt: off
        # profile_operation("PyTorch", lambda: ..., device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_topk(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyTopKDescriptor(op_desc))


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