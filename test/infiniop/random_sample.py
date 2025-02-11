from ctypes import POINTER, Structure, c_int32, c_uint64, c_void_p, c_float
import ctypes
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from operatorspy import (
    open_lib,
    to_tensor,
    DeviceEnum,
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    create_handle,
    destroy_handle,
    check_error,
    rearrange_tensor,
    create_workspace,
    U64,
)

from operatorspy.tests.test_utils import get_args
import torch


class RandomSampleDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopRandomSampleDescriptor_t = POINTER(RandomSampleDescriptor)


def random_sample(data, random_val, topp, topk, voc, temperature, torch_device):
    indices = torch.zeros([topk], dtype = torch.int64)
    dataNp = data.clone().detach()
    sorted_indices = torch.arange(voc)
    
    for i in range(topk):
        for j in range(i + 1, voc):
            if(dataNp[i] < dataNp[j]):
                tmp = dataNp[i].clone().detach()
                dataNp[i] = dataNp[j].clone().detach()
                dataNp[j] = tmp

                tmpInd = sorted_indices[i].clone().detach()
                sorted_indices[i] = sorted_indices[j].clone().detach()
                sorted_indices[j] = tmpInd
                
    #sorted_indices = torch.argsort(dataNp, descending=True)
    indices = sorted_indices[:topk] 
    
    dataNp = dataNp[sorted_indices]
    
    globalM = dataNp[0]
    dataNp = (dataNp - globalM) / temperature
    dataNp = torch.softmax(dataNp.float(), dim = 0)
    sum_s = 0
    for end in range(topk):
        sum_s += dataNp[end]
        if(sum_s >= topp):
            break
    if(end < topk - 1):
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
        if(random_val < sum_s):
            return indices[i]

def random_sample_0(data):
    return torch.argmax(data)

def test(lib, handle, torch_device, voc, random_val, topp, topk, temperature, x_dtype=torch.float16):
    print(
        f"Testing RandomSample on {torch_device} with voc:{voc} dtype:{x_dtype}"
    )
    data = torch.arange(voc).float() * 0.0001
    _perm = torch.randperm(voc)
    data = data[_perm].to(x_dtype).to(torch_device)
    if(topp > 0 and topk > 1):
        ans = random_sample(data.to("cpu"), random_val, topp, topk, voc, temperature, "cpu")
    else:
        ans = random_sample_0(data)
    indices = torch.zeros([1], dtype=torch.int64).to(torch_device)
    x_tensor = to_tensor(data, lib)
    indices_tensor = to_tensor(indices, lib)
    indices_tensor.descriptor.contents.dt = U64  # treat int64 as uint64

    descriptor = infiniopRandomSampleDescriptor_t()
    check_error(
        lib.infiniopCreateRandomSampleDescriptor(
            handle, ctypes.byref(descriptor), indices_tensor.descriptor, x_tensor.descriptor
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x_tensor.descriptor.contents.invalidate()
    indices_tensor.descriptor.contents.invalidate()

    workspace_size = c_uint64(0)
    check_error(
        lib.infiniopGetRandomSampleWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = create_workspace(workspace_size.value, torch_device) 
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
    check_error(lib.infiniopDestroyRandomSampleDescriptor(descriptor))

def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for (voc, random_val, topp, topk, temperature) in test_cases:
        test(lib, handle, "cpu", voc, random_val, topp, topk, temperature)
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for (voc, random_val, topp, topk, temperature) in test_cases:
        test(lib, handle, "cuda", voc, random_val, topp, topk, temperature)
    destroy_handle(lib, handle)


def test_bang(lib, test_cases):
    import torch_mlu

    device = DeviceEnum.DEVICE_BANG
    handle = create_handle(lib, device)
    for (voc, random_val, topp, topk, temperature) in test_cases:
        test(lib, handle, "mlu", voc, random_val, topp, topk, temperature)
    destroy_handle(lib, handle)


def test_ascend(lib, test_cases):
    import torch_npu
    device = DeviceEnum.DEVICE_ASCEND
    handle = create_handle(lib, device)
    for (voc, random_val, topp, topk, temperature) in test_cases:
        test(lib, handle, "npu", voc, random_val, topp, topk, temperature)
    destroy_handle(lib, handle)


if __name__ == "__main__":
    test_cases = [
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

    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    if args.bang:
        test_bang(lib, test_cases)
    if args.ascend:
        test_ascend(lib, test_cases)
    if not (args.cpu or args.cuda or args.bang or args.ascend):
        test_cpu(lib, test_cases)
    print("\033[92mTest passed!\033[0m")
