import torch
import ctypes
from ctypes import c_uint64
import math
from torch.nn import functional as F
import sys # 导入 sys 模块

# --- 诊断打印 #1: 检查脚本是否开始执行 ---
print("--- Python script started ---", flush=True)

try:
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
        InfiniDtype,
        InfiniDtypeNames,
        InfiniDeviceNames,
        InfiniDeviceEnum,
        infiniopOperatorDescriptor_t,
    )
    # --- 诊断打印 #2: 检查库是否成功导入 ---
    print("--- libinfiniop imported successfully ---", flush=True)
except ImportError as e:
    print(f"--- FAILED to import libinfiniop: {e} ---", flush=True)
    sys.exit(1)


# --- Test Case Configuration ---
_TEST_CASES = [
    (1, 64, 8, 8, 32),
    (4, 128, 16, 16, 64),
    (1, 256, 16, 4, 64),
    (2, 512, 32, 8, 128),
    (1, 1024, 8, 2, 64),
]

_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-2, "rtol": 1e-2},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
}

# --- Script Control Flags ---
DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def gqa_pytorch(q, k, v, y_tensor):
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    if num_q_heads != num_kv_heads:
        repeats = num_q_heads // num_kv_heads
        k = k.unsqueeze(2).repeat(1, 1, repeats, 1, 1).reshape(q.shape)
        v = v.unsqueeze(2).repeat(1, 1, repeats, 1, 1).reshape(q.shape)
    output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    y_tensor.copy_(output)


def test(
    handle,
    device,
    batch_size,
    seq_len,
    num_q_heads,
    num_kv_heads,
    head_size,
    tensor_dtype=InfiniDtype.F16,
    sync=None,
    q_stride=None,
    k_stride=None,
    v_stride=None
):
    # 添加 flush=True 确保立即输出
    print(
        f"Testing GQA on {InfiniDeviceNames[device]} with B={batch_size}, S={seq_len}, H_q={num_q_heads}, H_kv={num_kv_heads}, D'={head_size}, dtype={InfiniDtypeNames[tensor_dtype]}",
        flush=True
    )

    q_shape = (batch_size, num_q_heads, seq_len, head_size)
    k_shape = (batch_size, num_kv_heads, seq_len, head_size)
    v_shape = (batch_size, num_kv_heads, seq_len, head_size)
    y_shape = q_shape

    q = TestTensor(q_shape, q_stride,dt=tensor_dtype, device=device)
    k = TestTensor(k_shape, k_stride,dt=tensor_dtype, device=device)
    v = TestTensor(v_shape, v_stride,dt=tensor_dtype, device=device)
    y_ground_truth = TestTensor(y_shape, None,dt=tensor_dtype, device=device)
    y_actual = TestTensor(y_shape, None,dt=tensor_dtype, device=device)

    gqa_pytorch(q.torch_tensor(), k.torch_tensor(), v.torch_tensor(), y_ground_truth.torch_tensor())
    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateGQADescriptor(
            handle,
            ctypes.byref(descriptor),
            q.descriptor,
            k.descriptor,
            v.descriptor,
            y_actual.descriptor,
        )
    )

    def lib_gqa():
        check_error(
            LIBINFINIOP.infiniopGQA(
                descriptor,
                q.data(),
                k.data(),
                v.data(),
                y_actual.data(),
                None,
            )
        )

    lib_gqa()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, tensor_dtype)
    if DEBUG:
        debug(y_actual.torch_tensor(), y_ground_truth.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(y_actual.torch_tensor(), y_ground_truth.torch_tensor(), atol=atol, rtol=rtol)

    if PROFILE:
        profile_operation("PyTorch", lambda: gqa_pytorch(q.torch_tensor(), k.torch_tensor(), v.torch_tensor(), y_ground_truth.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lib_gqa, device, NUM_PRERUN, NUM_ITERATIONS)

    check_error(LIBINFINIOP.infiniopDestroyGQADescriptor(descriptor))


if __name__ == "__main__":
    try:
        args = get_args()
        print(f"--- Arguments parsed: {args} ---", flush=True)

        DEBUG = args.debug
        PROFILE = args.profile
        NUM_PRERUN = args.num_prerun
        NUM_ITERATIONS = args.num_iterations

        devices_to_test = get_test_devices(args)
        # --- 诊断打印 #3: 检查找到了哪些设备 ---
        print(f"--- Devices found by get_test_devices: {devices_to_test} ---", flush=True)
        
        if not devices_to_test:
            print("--- WARNING: No test devices found. Exiting. ---", flush=True)

        for device in devices_to_test:
            print(f"--- Starting tests for device: {InfiniDeviceNames.get(device, 'Unknown')} ---", flush=True)
            if device == InfiniDeviceEnum.NVIDIA:
                test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

        print("\033[92mAll GQA tests passed!\033[0m", flush=True)

    except Exception as e:
        # --- 诊断打印 #4: 捕获任何未预料到的错误 ---
        print(f"\n--- An unexpected error occurred: {e} ---", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
