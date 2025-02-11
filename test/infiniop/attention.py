from ctypes import POINTER, Structure, c_int32, c_uint64, c_void_p, c_float, c_bool
import ctypes
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from operatorspy import (
    open_lib,
    to_tensor,
    CTensor,
    DeviceEnum,
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    create_handle,
    destroy_handle,
    check_error,
    rearrange_tensor,
    create_workspace,
)

from operatorspy.tests.test_utils import get_args
import torch
import torch.nn.functional as F


class AttentionDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopAttentionDescriptor_t = POINTER(AttentionDescriptor)


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
    lib,
    handle,
    torch_device,
    n_q_head,
    n_kv_head,
    seq_len,
    head_dim,
    pos,
    k_cache_buf_len,
    v_cache_buf_len,
    dtype=torch.float16,
    q_stride=None,
    k_stride=None,
    v_stride=None,
    k_cache_stride=None,
    v_cache_stride=None,
):
    print(
        f"Testing Attention on {torch_device} with n_q_head:{n_q_head} n_kv_head:{n_kv_head} seq_len:{seq_len} head_dim:{head_dim} pos:{pos} "
        f"dtype:{dtype} q_stride:{q_stride} k_stride:{k_stride} v_stride:{v_stride} k_cache_stride:{k_cache_stride} v_cache_stride:{v_cache_stride}"
    )

    out = torch.zeros([seq_len, n_q_head, head_dim], dtype=dtype, device=torch_device)
    q = torch.rand([n_q_head, seq_len, head_dim], dtype=dtype).to(torch_device) * 0.1
    k = torch.rand([n_kv_head, seq_len, head_dim], dtype=dtype).to(torch_device) * 0.1
    v = torch.rand([n_kv_head, seq_len, head_dim], dtype=dtype).to(torch_device) * 0.1
    k_cache = (
        torch.rand([n_kv_head, k_cache_buf_len, head_dim], dtype=dtype).to(torch_device)
        * 0.1
    )
    v_cache = (
        torch.rand([n_kv_head, v_cache_buf_len, head_dim], dtype=dtype).to(torch_device)
        * 0.1
    )

    ans = attention(q, k, v, k_cache, v_cache, pos)

    if q_stride is not None:
        q = rearrange_tensor(q, q_stride)
    if k_stride is not None:
        k = rearrange_tensor(k, k_stride)
    if v_stride is not None:
        v = rearrange_tensor(v, v_stride)
    if k_cache_stride is not None:
        k_cache = rearrange_tensor(k_cache, k_cache_stride)
    if v_cache_stride is not None:
        v_cache = rearrange_tensor(v_cache, v_cache_stride)

    out_tensor = to_tensor(out, lib)
    q_tensor = to_tensor(q, lib)
    k_tensor = to_tensor(k, lib)
    v_tensor = to_tensor(v, lib)
    k_cache_tensor = to_tensor(k_cache, lib)
    v_cache_tensor = to_tensor(v_cache, lib)

    descriptor = infiniopAttentionDescriptor_t()
    check_error(
        lib.infiniopCreateAttentionDescriptor(
            handle,
            ctypes.byref(descriptor),
            out_tensor.descriptor,
            q_tensor.descriptor,
            k_tensor.descriptor,
            v_tensor.descriptor,
            k_cache_tensor.descriptor,
            v_cache_tensor.descriptor,
            pos,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    out_tensor.descriptor.contents.invalidate()
    q_tensor.descriptor.contents.invalidate()
    k_tensor.descriptor.contents.invalidate()
    v_tensor.descriptor.contents.invalidate()
    k_cache_tensor.descriptor.contents.invalidate()
    v_cache_tensor.descriptor.contents.invalidate()

    workspace_size = c_uint64(0)
    check_error(
        lib.infiniopGetAttentionWorkspaceSize(descriptor, ctypes.byref(workspace_size))
    )
    workspace = create_workspace(workspace_size.value, out.device)

    check_error(
        lib.infiniopAttention(
            descriptor,
            workspace.data_ptr() if workspace is not None else None,
            workspace_size.value,
            out_tensor.data,
            q_tensor.data,
            k_tensor.data,
            v_tensor.data,
            k_cache_tensor.data,
            v_cache_tensor.data,
            None,
        )
    )

    assert torch.allclose(out, ans, atol=1e-4, rtol=1e-2)

    check_error(lib.infiniopDestroyAttentionDescriptor(descriptor))


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)

    for (
        n_q_head,
        n_kv_head,
        seq_len,
        head_dim,
        pos,
        k_cache_buf_len,
        v_cache_buf_len,
        dtype,
        q_stride,
        k_stride,
        v_stride,
        k_cache_stride,
        v_cache_stride,
    ) in test_cases:
        test(
            lib,
            handle,
            "cpu",
            n_q_head,
            n_kv_head,
            seq_len,
            head_dim,
            pos,
            k_cache_buf_len,
            v_cache_buf_len,
            dtype,
            q_stride,
            k_stride,
            v_stride,
            k_cache_stride,
            v_cache_stride,
        )

    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)

    for (
        n_q_head,
        n_kv_head,
        seq_len,
        head_dim,
        pos,
        k_cache_buf_len,
        v_cache_buf_len,
        dtype,
        q_stride,
        k_stride,
        v_stride,
        k_cache_stride,
        v_cache_stride,
    ) in test_cases:
        test(
            lib,
            handle,
            "cuda",
            n_q_head,
            n_kv_head,
            seq_len,
            head_dim,
            pos,
            k_cache_buf_len,
            v_cache_buf_len,
            dtype,
            q_stride,
            k_stride,
            v_stride,
            k_cache_stride,
            v_cache_stride,
        )

    destroy_handle(lib, handle)


def test_bang(lib, test_cases):
    import torch_mlu

    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)

    for (
        n_q_head,
        n_kv_head,
        seq_len,
        head_dim,
        pos,
        k_cache_buf_len,
        v_cache_buf_len,
        dtype,
        q_stride,
        k_stride,
        v_stride,
        k_cache_stride,
        v_cache_stride,
    ) in test_cases:
        test(
            lib,
            handle,
            "mlu",
            n_q_head,
            n_kv_head,
            seq_len,
            head_dim,
            pos,
            k_cache_buf_len,
            v_cache_buf_len,
            dtype,
            q_stride,
            k_stride,
            v_stride,
            k_cache_stride,
            v_cache_stride,
        )

    destroy_handle(lib, handle)


if __name__ == "__main__":
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
            torch.float16,  # dtype
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
            torch.float16,  # dtype
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
            torch.float16,  # dtype
            None,  # q_stride
            None,  # k_stride
            None,  # v_stride
            None,  # k_cache_stride
            None,  # v_cache_stride
        ),
    ]
    args = get_args()
    lib = open_lib()

    lib.infiniopCreateAttentionDescriptor.restype = c_int32
    lib.infiniopCreateAttentionDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopAttentionDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_uint64,
    ]

    lib.infiniopGetAttentionWorkspaceSize.restype = c_int32
    lib.infiniopGetAttentionWorkspaceSize.argtypes = [
        infiniopAttentionDescriptor_t,
        POINTER(c_uint64),
    ]

    lib.infiniopAttention.restype = c_int32
    lib.infiniopAttention.argtypes = [
        infiniopAttentionDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyAttentionDescriptor.restype = c_int32
    lib.infiniopDestroyAttentionDescriptor.argtypes = [
        infiniopAttentionDescriptor_t,
    ]

    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    if args.bang:
        test_bang(lib, test_cases)
    if not (args.cpu or args.cuda or args.bang):
        test_cpu(lib, test_cases)
    print("\033[92mTest passed!\033[0m")
