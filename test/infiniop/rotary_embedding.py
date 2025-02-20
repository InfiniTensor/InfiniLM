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
    # (t_shape, t_strides)
        ((1, 32, 128), None),
        ((1, 32, 64), None),
        # 昇腾暂不满足这个用例，最后一维度 <=32 会有问题，可能与其核心
        # 接口 GatherMask 的内部实现相关，目前 48 64 128 都可以支持
        ((4, 1, 32), None),
        ((1, 32, 128), None),
        ((3, 32, 128), (8000, 200, 1)),
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



class RoPEDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopRoPEDescriptor_t = POINTER(RoPEDescriptor)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[0], x.shape[-1])
    shape = [d if i == 0 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def rotary_embedding(t, pos, theta, torch_device):
    dh = t.shape[2]
    assert dh % 2 == 0, "Embedding dimension must be even."
    t_even = t[..., 0::2]  # [seq_len, n_head, dh // 2]
    t_odd = t[..., 1::2]  # [seq_len, n_head, dh // 2]
    freqs = (1.0 / (theta ** (torch.arange(0, dh, 2).float() / dh))).to(torch_device)
    freqs = torch.outer(pos, freqs)  # [seq_len, dh // 2]
    cos = torch.cos(freqs).unsqueeze(1)  # [seq_len, 1, dh // 2]
    sin = torch.sin(freqs).unsqueeze(1)  # [seq_len, 1, dh // 2]

    t_out_even = t_even * cos - t_odd * sin
    t_out_odd = t_even * sin + t_odd * cos

    t_out = torch.empty_like(t)
    t_out[..., 0::2] = t_out_even
    t_out[..., 1::2] = t_out_odd

    return t_out


def sin_cos_table(max_seq_len, dim, torch_device, theta):
    pos = torch.arange(
        0, max_seq_len, dtype=torch.float32, device=torch.device(torch_device)
    )
    freqs = (1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))).to(
        torch_device
    )
    # (a0, a1, a2) -> (a0, a0, a1, a1, a2, a2)
    freqs = torch.repeat_interleave(freqs, repeats=2)
    angles = torch.outer(pos, freqs)
    return torch.sin(angles), torch.cos(angles)


def test(
    lib, 
    handle, 
    torch_device, 
    shape, 
    strides=None, 
    dtype=torch.float16
):
    print(
        f"Testing Rotary Positional Embedding on {torch_device} with shape:{shape} strides:{strides} and dtype:{dtype}"
    )

    t = torch.rand(shape, dtype=dtype)

    t = rearrange_if_needed(t, strides)

    posTmp = torch.arange(0, t.shape[0]).to(torch_device)
    pos = torch.zeros(2 * posTmp.shape[0], dtype=torch.int32)
    for i in range(posTmp.shape[0]):
        pos[2 * i] = posTmp[i]
        pos[2 * i + 1] = 0
    pos = pos.to(torch_device)
    theta = 1e4

    ans = rotary_embedding(t, posTmp, theta, torch_device)

    descriptor = infiniopRoPEDescriptor_t()
    # 2x table length for test
    sin_table, cos_table = sin_cos_table(t.shape[0] * 2, t.shape[2], t.device, theta)

    t_tensor, sin_table_tensor, cos_table_tensor = [to_tensor(tensor, lib) for tensor in [t, sin_table, cos_table]]
    
    pos_tensor = to_tensor(pos[: t.shape[0]], lib)
    pos_tensor.descriptor.contents.dtype = InfiniDtype.U64
    

    if torch_device == "npu":
        torch.npu.synchronize()

    check_error(
        lib.infiniopCreateRoPEDescriptor(
            handle,
            byref(descriptor),
            t_tensor.descriptor,
            pos_tensor.descriptor,
            sin_table_tensor.descriptor,
            cos_table_tensor.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [t_tensor, pos_tensor, sin_table_tensor, cos_table_tensor]:
        tensor.descriptor.contents.invalidate()

    workspace_size = c_uint64(0)
    check_error(
        lib.infiniopGetRoPEWorkspaceSize(descriptor, ctypes.byref(workspace_size))
    )
    workspace = create_workspace(workspace_size.value, t.device)

    def lib_rope():
        check_error(
            lib.infiniopRoPE(
                descriptor,
                workspace.data_ptr() if workspace is not None else None,
                workspace_size.value,
                t_tensor.data,
                pos_tensor.data,
                sin_table_tensor.data,
                cos_table_tensor.data,
                None,
            )
        )

    lib_rope()
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(t, ans, atol=atol, rtol=rtol)
    assert torch.allclose(t, ans, atol=atol, rtol=rtol)
    
    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: rotary_embedding(t, posTmp, theta, torch_device),
            torch_device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib", lambda: lib_rope(), torch_device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(lib.infiniopDestroyRoPEDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateRoPEDescriptor.restype = c_int32
    lib.infiniopCreateRoPEDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopRoPEDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopGetRoPEWorkspaceSize.restype = c_int32
    lib.infiniopGetRoPEWorkspaceSize.argtypes = [
        infiniopRoPEDescriptor_t,
        POINTER(c_uint64),
    ]
    lib.infiniopRoPE.restype = c_int32
    lib.infiniopRoPE.argtypes = [
        infiniopRoPEDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyRoPEDescriptor.restype = c_int32
    lib.infiniopDestroyRoPEDescriptor.argtypes = [
        infiniopRoPEDescriptor_t,
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
