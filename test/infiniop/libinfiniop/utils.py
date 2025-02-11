import ctypes
from .datatypes import *
from .liboperators import infiniopTensorDescriptor_t, CTensor, infiniopHandle_t


def check_error(status):
    if status != 0:
        raise Exception("Error code " + str(status))


def to_tensor(tensor, lib):
    """
    Convert a PyTorch tensor to a library Tensor(descriptor, data).
    """
    import torch

    ndim = tensor.ndimension()
    shape = (ctypes.c_size_t * ndim)(*tensor.shape)
    strides = (ctypes.c_int64 * ndim)(*(tensor.stride()))
    data_ptr = tensor.data_ptr()
    # fmt: off
    dt = (
        InfiniDtype.I8 if tensor.dtype == torch.int8 else
        InfiniDtype.I16 if tensor.dtype == torch.int16 else
        InfiniDtype.I32 if tensor.dtype == torch.int32 else
        InfiniDtype.I64 if tensor.dtype == torch.int64 else
        InfiniDtype.U8 if tensor.dtype == torch.uint8 else
        InfiniDtype.F16 if tensor.dtype == torch.float16 else
        InfiniDtype.BF16 if tensor.dtype == torch.bfloat16 else
        InfiniDtype.F32 if tensor.dtype == torch.float32 else
        InfiniDtype.F64 if tensor.dtype == torch.float64 else
        # TODO: These following types may not be supported by older 
        # versions of PyTorch.
        InfiniDtype.U16 if tensor.dtype == torch.uint16 else
        InfiniDtype.U32 if tensor.dtype == torch.uint32 else
        InfiniDtype.U64 if tensor.dtype == torch.uint64 else
        None
    )
    # fmt: on
    assert dt is not None
    # Create TensorDecriptor
    tensor_desc = infiniopTensorDescriptor_t()
    lib.infiniopCreateTensorDescriptor(
        ctypes.byref(tensor_desc), ndim, shape, strides, dt
    )
    # Create Tensor
    return CTensor(tensor_desc, data_ptr)

def create_workspace(size, torch_device):
    if size == 0:
        return None
    import torch
    return torch.zeros(size=(size,), dtype=torch.uint8, device=torch_device)

def create_handle(lib, device, id=0):
    handle = infiniopHandle_t()
    check_error(lib.infiniopCreateHandle(ctypes.byref(handle), device, id))
    return handle


def destroy_handle(lib, handle):
    check_error(lib.infiniopDestroyHandle(handle))


def rearrange_tensor(tensor, new_strides):
    """
    Given a PyTorch tensor and a list of new strides, return a new PyTorch tensor with the given strides.
    """
    import torch

    shape = tensor.shape

    new_size = [0] * len(shape)
    left = 0
    right = 0
    for i in range(len(shape)):
        if new_strides[i] > 0:
            new_size[i] = (shape[i] - 1) * new_strides[i] + 1
            right += new_strides[i] * (shape[i] - 1)
        else:  # TODO: Support negative strides in the future
            # new_size[i] = (shape[i] - 1) * (-new_strides[i]) + 1
            # left += new_strides[i] * (shape[i] - 1)
            raise ValueError("Negative strides are not supported yet")

    # Create a new tensor with zeros
    new_tensor = torch.zeros(
        (right - left + 1,), dtype=tensor.dtype, device=tensor.device
    )

    # Generate indices for original tensor based on original strides
    indices = [torch.arange(s) for s in shape]
    mesh = torch.meshgrid(*indices, indexing="ij")

    # Flatten indices for linear indexing
    linear_indices = [m.flatten() for m in mesh]

    # Calculate new positions based on new strides
    new_positions = sum(
        linear_indices[i] * new_strides[i] for i in range(len(shape))
    ).to(tensor.device)
    offset = -left
    new_positions += offset

    # Copy the original data to the new tensor
    new_tensor.view(-1).index_add_(0, new_positions, tensor.view(-1))
    new_tensor.set_(new_tensor.untyped_storage(), offset, shape, tuple(new_strides))

    return new_tensor
