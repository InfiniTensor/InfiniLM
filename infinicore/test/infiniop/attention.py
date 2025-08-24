from ctypes import c_uint64
import ctypes
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
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

import torch


def causal_softmax(x):
    type = x.dtype
    mask = torch.tril(torch.ones_like(x), diagonal=-1).flip(dims=[-2, -1])
    y = x.clone()
    masked = torch.where(mask == 1, -torch.inf, y.to(torch.float32))
    return torch.nn.functional.softmax(masked, dim=-1).to(type)


def attention(q, k, v, k_cache, v_cache, pos):
    type = q.dtype

    n_q_head = q.shape[0]
    n_kv_head = k.shape[0]

    # Concatenate key and value caches
    k_cache = k_cache[:, :pos, :]  # (n_kv_head, pos, head_dim)
    v_cache = v_cache[:, :pos, :]  # (n_kv_head, pos, head_dim)
    k = torch.cat([k_cache, k], dim=1)  # (n_kv_head, total_seq_len, head_dim)
    v = torch.cat([v_cache, v], dim=1)  # (n_kv_head, total_seq_len, head_dim)

    total_seq_len = k.shape[1]

    head_dim = v.shape[-1]

    if n_q_head != n_kv_head:
        q = q.reshape(
            n_kv_head, -1, head_dim
        )  # (n_kv_head, n_group * seq_len, head_dim)

    # Scaled dot-product attention
    attn_scores = (
        torch.einsum("hqd,hkd->hqk", q.to(torch.float32), k.to(torch.float32))
        .to(type)
        .reshape(n_q_head, -1, total_seq_len)
    )  # (n_q_head, seq_len, total_seq_len)
    attn_scores = attn_scores / (head_dim**0.5)

    attn_weights = causal_softmax(attn_scores).reshape(
        n_kv_head, -1, total_seq_len
    )  # (n_kv_head, seq_len, total_seq_len)

    # Weighted sum of values
    attn_output = (
        torch.einsum(
            "hqk,hkd->hqd", attn_weights.to(torch.float32), v.to(torch.float32)
        )
        .to(type)
        .reshape(n_q_head, -1, head_dim)
        .permute(1, 0, 2)
    )  # ([seq_len, n_q_head, head_dim])

    return attn_output


def test(
    handle,
    device,
    n_q_head,
    n_kv_head,
    seq_len,
    head_dim,
    pos,
    k_cache_buf_len,
    v_cache_buf_len,
    q_stride=None,
    k_stride=None,
    v_stride=None,
    k_cache_stride=None,
    v_cache_stride=None,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing Attention on {InfiniDeviceNames[device]} with n_q_head:{n_q_head} n_kv_head:{n_kv_head} seq_len:{seq_len} head_dim:{head_dim} pos:{pos} "
        f"dtype:{InfiniDtypeNames[dtype]} q_stride:{q_stride} k_stride:{k_stride} v_stride:{v_stride} k_cache_stride:{k_cache_stride} v_cache_stride:{v_cache_stride}"
    )

    out = TestTensor([seq_len, n_q_head, head_dim], None, dtype, device, mode="zeros")
    q = TestTensor([n_q_head, seq_len, head_dim], q_stride, dtype, device, scale=0.1)
    k = TestTensor([n_kv_head, seq_len, head_dim], k_stride, dtype, device, scale=0.1)
    v = TestTensor([n_kv_head, seq_len, head_dim], v_stride, dtype, device, scale=0.1)
    k_cache = TestTensor(
        [n_kv_head, k_cache_buf_len, head_dim], k_cache_stride, dtype, device, scale=0.1
    )
    v_cache = TestTensor(
        [n_kv_head, v_cache_buf_len, head_dim], v_cache_stride, dtype, device, scale=0.1
    )

    def torch_attention():
        return attention(
            q.torch_tensor(),
            k.torch_tensor(),
            v.torch_tensor(),
            k_cache.torch_tensor(),
            v_cache.torch_tensor(),
            pos,
        )

    ans = torch_attention()

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateAttentionDescriptor(
            handle,
            ctypes.byref(descriptor),
            out.descriptor,
            q.descriptor,
            k.descriptor,
            v.descriptor,
            k_cache.descriptor,
            v_cache.descriptor,
            pos,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [out, q, k, v, k_cache, v_cache]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetAttentionWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, out.device)

    def lib_attention():
        check_error(
            LIBINFINIOP.infiniopAttention(
                descriptor,
                workspace.data(),
                workspace_size.value,
                out.data(),
                q.data(),
                k.data(),
                v.data(),
                k_cache.data(),
                v_cache.data(),
                None,
            )
        )

    lib_attention()

    # Validate results
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(out.actual_tensor(), ans, atol=atol, rtol=rtol)
    assert torch.allclose(out.actual_tensor(), ans, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch_attention(), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_attention(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyAttentionDescriptor(descriptor))


if __name__ == "__main__":
    _TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32]

    # Tolerance map for different data types
    _TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-4, "rtol": 1e-2},
        InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-3},
    }

    DEBUG = False
    PROFILE = False
    NUM_PRERUN = 10
    NUM_ITERATIONS = 1000
    test_cases = [
        # prefill
        (
            32,  # n_q_head
            4,  # n_kv_head
            5,  # seq_len
            64,  # head_dim
            0,  # pos
            2048,  # k_cache_buf_len
            2048,  # v_cache_buf_len
            [64, 2560, 1],  # q_stride
            [64, 2560, 1],  # k_stride
            [64, 2560, 1],  # v_stride
            [64, 11264, 1],  # k_cache_stride
            [64, 11264, 1],  # v_cache_stride
        ),
        # decode
        (
            32,  # n_q_head
            4,  # n_kv_head
            1,  # seq_len
            64,  # head_dim
            3,  # pos
            2048,  # k_cache_buf_len
            2048,  # v_cache_buf_len
            [64, 2560, 1],  # q_stride
            [64, 2560, 1],  # k_stride
            [64, 2560, 1],  # v_stride
            [64, 11264, 1],  # k_cache_stride
            [64, 11264, 1],  # v_cache_stride
        ),
        # for test
        (
            8,  # n_q_head
            4,  # n_kv_head
            2,  # seq_len
            16,  # head_dim
            1,  # pos
            8,  # k_cache_buf_len
            8,  # v_cache_buf_len
            None,  # q_stride
            None,  # k_stride
            None,  # v_stride
            None,  # k_cache_stride
            None,  # v_cache_stride
        ),
        (
            28,  # n_q_head
            28,  # n_kv_head
            15,  # seq_len
            128,  # head_dim
            0,  # pos
            2048,  # k_cache_buf_len
            2048,  # v_cache_buf_len
            [128, 10752, 1],  # q_stride
            [128, 10752, 1],  # k_stride
            [128, 10752, 1],  # v_stride
            [128, 3584, 1],  # k_cache_stride
            [128, 3584, 1],  # v_cache_stride
        ),
    ]
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(device, test, test_cases, _TENSOR_DTYPES)
    print("\033[92mTest passed!\033[0m")
