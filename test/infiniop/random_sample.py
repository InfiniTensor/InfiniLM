import torch
import ctypes
from ctypes import POINTER, Structure, c_int32, c_uint64, c_void_p, c_float
from libinfiniop import (
    InfiniDtype,
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    open_lib,
    to_tensor,
    get_test_devices,
    check_error,
    create_workspace,
    test_operator,
    get_args,
    debug_all,
    get_tolerance,
    profile_operation,
    synchronize_device,
)

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES = [
    # voc, random_val, topp, topk, temperature
    (512, 0.8, 0.8, 3, 0.5),
    (4096, 0.05, 0.9, 5, 1.0),
    (16384, 0.15, 0.85, 10, 2.0),
    (512, 0.08, 0, 3, 0.5),
    (4096, 0.5, 0.9, 1, 1.0),
    (16384, 0.15, 0, 1, 2.0),
    (16384, 0.15, 0, 1, 2.0),
    (32000, 0.08, 0.8, 50, 1.0),
    (32000, 0.08, 1.0, 25, 1.0),
    # (119696, 0.01, 1.0, 100, 1.0),
]

# Data types used for testing
_TENSOR_DTYPES = [torch.float16]

_TOLERANCE_MAP = {
    torch.float16: {"atol": 0, "rtol": 0},
}


DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


class RandomSampleDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopRandomSampleDescriptor_t = POINTER(RandomSampleDescriptor)


def random_sample(data, random_val, topp, topk, voc, temperature):
    if topp > 0 and topk > 1:
        sorted_vals, sorted_indices = torch.sort(data, descending=True)
        
        scaled_vals = (sorted_vals - sorted_vals[0]) / temperature
        probs = torch.softmax(scaled_vals, dim=0)
        cum_probs = torch.cumsum(probs, dim=0)
        
        k_index = min(topk, voc) - 1
        threshold = min(cum_probs[k_index], topp) * random_val
        
        idx = torch.searchsorted(cum_probs, threshold)
        return sorted_indices[idx]
    
    return torch.argmax(data)


def test(
    lib,
    handle,
    torch_device,
    voc,
    random_val,
    topp,
    topk,
    temperature,
    dtype=torch.float16,
    sync=None
):
    print(
        f"Testing RandomSample on {torch_device} with voc:{voc} random_val:{random_val} topp:{topp} topk:{topk} temperature:{temperature} dtype:{dtype}"
    )

    data = torch.arange(voc).float() * 0.0001
    _perm = torch.randperm(voc)
    data = data[_perm].to(dtype).to(torch_device)

    ans = random_sample(
        data, random_val, topp, topk, voc, temperature
    )  # 这个函数在device速度可能会很慢，可以通过data.to("cpu")方式加快计算过程

    indices = torch.zeros([], dtype=torch.int64).to(torch_device)

    x_tensor, indices_tensor = [to_tensor(tensor, lib) for tensor in [data, indices]]

    indices_tensor.descriptor.contents.dt = InfiniDtype.U64  # treat int64 as uint64

    if sync is not None:
        sync()

    descriptor = infiniopRandomSampleDescriptor_t()
    check_error(
        lib.infiniopCreateRandomSampleDescriptor(
            handle,
            ctypes.byref(descriptor),
            indices_tensor.descriptor,
            x_tensor.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [x_tensor, indices_tensor]:
        tensor.destroyDesc(lib)

    workspace_size = c_uint64(0)
    check_error(
        lib.infiniopGetRandomSampleWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = create_workspace(workspace_size.value, torch_device)

    def lib_random_sample():
        check_error(
            lib.infiniopRandomSample(
                descriptor,
                workspace.data_ptr() if workspace is not None else None,
                workspace_size.value,
                indices_tensor.data,
                x_tensor.data,
                random_val,
                topp,
                topk,
                temperature,
                None,
            )
        )

    lib_random_sample()

    if torch_device == "npu":
        synchronize_device(torch_device)

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug_all(
            (indices.type(ans.dtype), data[indices]),
            (ans, data[ans]),
            "or",
            atol=atol,
            rtol=rtol,
        )
    assert indices.type(ans.dtype) == ans or data[ans] == data[indices]

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: random_sample(
                data, random_val, topp, topk, voc, temperature
            ), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_random_sample(), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(lib.infiniopDestroyRandomSampleDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    lib = open_lib()

    lib.infiniopCreateRandomSampleDescriptor.restype = c_int32
    lib.infiniopCreateRandomSampleDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopRandomSampleDescriptor_t),
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetRandomSampleWorkspaceSize.restype = c_int32
    lib.infiniopGetRandomSampleWorkspaceSize.argtypes = [
        infiniopRandomSampleDescriptor_t,
        POINTER(c_uint64),
    ]

    lib.infiniopRandomSample.restype = c_int32
    lib.infiniopRandomSample.argtypes = [
        infiniopRandomSampleDescriptor_t,
        c_void_p,
        c_uint64,
        c_uint64,
        c_void_p,
        c_float,
        c_float,
        c_int32,
        c_float,
        c_void_p,
    ]

    lib.infiniopDestroyRandomSampleDescriptor.restype = c_int32
    lib.infiniopDestroyRandomSampleDescriptor.argtypes = [
        infiniopRandomSampleDescriptor_t,
    ]

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(lib, device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
