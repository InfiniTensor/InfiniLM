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
_TENSOR_DTYPES = [torch.float16, torch.float32]


PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


class RandomSampleDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopRandomSampleDescriptor_t = POINTER(RandomSampleDescriptor)


def random_sample(data, random_val, topp, topk, voc, temperature, torch_device):
    indices = torch.zeros([topk], dtype=torch.int64)
    dataNp = data.clone().detach()
    sorted_indices = torch.arange(voc)

    for i in range(topk):
        for j in range(i + 1, voc):
            if dataNp[i] < dataNp[j]:
                tmp = dataNp[i].clone().detach()
                dataNp[i] = dataNp[j].clone().detach()
                dataNp[j] = tmp

                tmpInd = sorted_indices[i].clone().detach()
                sorted_indices[i] = sorted_indices[j].clone().detach()
                sorted_indices[j] = tmpInd

    # sorted_indices = torch.argsort(dataNp, descending=True)
    indices = sorted_indices[:topk]

    dataNp = dataNp[sorted_indices]

    globalM = dataNp[0]
    dataNp = (dataNp - globalM) / temperature
    dataNp = torch.softmax(dataNp.float(), dim=0)
    sum_s = 0
    for end in range(topk):
        sum_s += dataNp[end]
        if sum_s >= topp:
            break
    if end < topk - 1:
        end += 1
    else:
        end = topk

    sum_s = 0
    for i in range(end):
        sum_s += dataNp[i]
    random_val *= sum_s

    sum_s = 0
    for i in range(end):
        sum_s += dataNp[i]
        if random_val < sum_s:
            return indices[i]


def random_sample_0(data):
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
    x_dtype=torch.float16,
):
    print(f"Testing RandomSample on {torch_device} with voc:{voc} dtype:{x_dtype}")
    data = torch.arange(voc).float() * 0.0001
    _perm = torch.randperm(voc)
    data = data[_perm].to(x_dtype).to(torch_device)
    if topp > 0 and topk > 1:
        ans = random_sample(
            data.to("cpu"), random_val, topp, topk, voc, temperature, "cpu"
        )
    else:
        ans = random_sample_0(data)
    indices = torch.zeros([1], dtype=torch.int64).to(torch_device)
    x_tensor = to_tensor(data, lib)
    indices_tensor = to_tensor(indices, lib)
    indices_tensor.descriptor.contents.dt = U64  # treat int64 as uint64

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
        tensor.descriptor.contents.invalidate()

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
    if torch_device == "npu":
        torch.npu.synchronize()

    assert indices[0].type(ans.dtype) == ans or data[ans] == data[indices[0]]
    
    # Profiling workflow
    if PROFILE:
        # fmt: off
        if topp > 0 and topk > 1:
            profile_operation("PyTorch", lambda: random_sample(
                data.to("cpu"), random_val, topp, topk, voc, temperature, "cpu"
            ), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        else:
            profile_operation("PyTorch", lambda: random_sample_0(data), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        
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

    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(lib, device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
