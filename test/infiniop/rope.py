import torch
import ctypes
from ctypes import POINTER, Structure, c_int32, c_uint64, c_void_p
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
    synchronize_device,
)
from enum import Enum, auto

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # (shape, x_strides, y_strides)
    ((1, 32, 128), None, None),
    ((10, 32, 64), None, None),
    # 昇腾暂不满足这个用例，最后一维度 <=32 会有问题，可能与其核心
    # 接口 GatherMask 的内部实现相关，目前 48 64 128 都可以支持
    ((4, 1, 32), (64, 64, 1), None),
    ((11, 33, 128), None, (8000, 200, 1)),
    ((3, 32, 128), (8000, 200, 1), (7000, 128, 1)),
]

# Data types used for testing
_TENSOR_DTYPES = [torch.float16, torch.float32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    torch.float16: {"atol": 1e-3, "rtol": 1e-2},
    torch.float32: {"atol": 1e-4, "rtol": 1e-3},
}


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_X = auto()


_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE_X,
]

_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


class RoPEDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopRoPEDescriptor_t = POINTER(RoPEDescriptor)


def rotary_embedding(t, sin, cos, torch_device):
    dh = t.shape[2]
    dt = t.dtype
    assert dh % 2 == 0, "Embedding dimension must be even."
    t_even = t[..., 0::2]  # [seq_len, n_head, dh // 2]
    t_odd = t[..., 1::2]  # [seq_len, n_head, dh // 2]
    cos = cos.unsqueeze(1)  # [seq_len, 1, dh // 2]
    sin = sin.unsqueeze(1)  # [seq_len, 1, dh // 2]
    if torch_device == "cpu":
        (t_even, t_odd, cos, sin) = (
            t_even.float(),
            t_odd.float(),
            cos.float(),
            sin.float(),
        )

    t_out_even = t_even * cos - t_odd * sin
    t_out_odd = t_even * sin + t_odd * cos

    t_out = torch.empty_like(t)
    t_out[..., 0::2] = t_out_even
    t_out[..., 1::2] = t_out_odd

    return t_out.to(dt).to(torch_device)


def sin_cos_table(pos, dim, torch_device, theta, dtype):
    assert dim % 2 == 0, "Embedding dimension must be even."
    freqs = (1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))).to(
        torch_device
    )
    angles = torch.outer(pos, freqs)
    return torch.sin(angles).to(dtype), torch.cos(angles).to(dtype)


def test(
    lib,
    handle,
    torch_device,
    shape,
    x_strides=None,
    y_strides=None,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=torch.float32,
):
    if inplace == Inplace.INPLACE_X:
        y_strides = x_strides
    print(
        f"Testing Rotary Positional Embedding on {torch_device} with shape:{shape} x_strides:{x_strides} y_strides:{y_strides} and dtype:{dtype} inplace:{inplace}"
    )

    x = torch.rand(shape, dtype=dtype).to(torch_device)
    x = rearrange_if_needed(x, x_strides)
    if inplace == Inplace.INPLACE_X:
        y = x
    else:
        y = torch.rand(shape, dtype=dtype).to(torch_device)
        y = rearrange_if_needed(y, y_strides)
    theta = 1e5
    pos = torch.arange(0, x.shape[0], dtype=torch.int32).to(torch_device)
    sin_table, cos_table = sin_cos_table(pos, x.shape[2], x.device, theta, dtype)

    ans = rotary_embedding(x, sin_table, cos_table, torch_device)

    descriptor = infiniopRoPEDescriptor_t()
    x_tensor, pos_tensor, sin_table_tensor, cos_table_tensor = [
        to_tensor(tensor, lib, force_unsigned=True)
        for tensor in [x, pos, sin_table, cos_table]
    ]
    if inplace == Inplace.INPLACE_X:
        y_tensor = x_tensor
    else:
        y_tensor = to_tensor(y, lib)

    if torch_device == "npu":
        synchronize_device(torch_device)

    check_error(
        lib.infiniopCreateRoPEDescriptor(
            handle,
            ctypes.byref(descriptor),
            y_tensor.descriptor,
            x_tensor.descriptor,
            pos_tensor.descriptor,
            sin_table_tensor.descriptor,
            cos_table_tensor.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [y_tensor, x_tensor, pos_tensor, sin_table_tensor, cos_table_tensor]:
        tensor.destroyDesc(lib)

    workspace_size = c_uint64(0)
    check_error(
        lib.infiniopGetRoPEWorkspaceSize(descriptor, ctypes.byref(workspace_size))
    )
    workspace = create_workspace(workspace_size.value, x.device)

    def lib_rope():
        check_error(
            lib.infiniopRoPE(
                descriptor,
                workspace.data_ptr() if workspace is not None else None,
                workspace_size.value,
                y_tensor.data,
                x_tensor.data,
                pos_tensor.data,
                sin_table_tensor.data,
                cos_table_tensor.data,
                None,
            )
        )

    lib_rope()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y, ans, atol=atol, rtol=rtol)
    assert torch.allclose(y, ans, atol=atol, rtol=rtol)

    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: rotary_embedding(x, pos, theta, torch_device),
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
