import ctypes

import numpy as np

import infinicore.device
import infinicore.dtype
from infinicore.lib import _infinicore

from .utils import (
    infinicore_to_numpy_dtype,
    numpy_to_infinicore_dtype,
    to_infinicore_dtype,
)


class Tensor:
    # Public attributes describing the Tensor
    _underlying: _infinicore.Tensor
    _torch_ref: "torch.Tensor"  # noqa: F821
    shape: list[int]
    dtype: infinicore.dtype
    device: infinicore.device

    def __init__(self, underlying, *, _torch_ref=None):
        """An internal method. Please do not use this directly."""

        self._underlying = underlying
        self._torch_ref = _torch_ref

    def __getattr__(self, name):
        # Lazily construct and cache an attribute.
        # such as, self.shape, self.dtype, self.device .
        if name == "shape":
            setattr(self, name, getattr(self._underlying, name))
        elif name == "dtype":
            setattr(self, name, infinicore.dtype(getattr(self._underlying, name)))
        elif name == "device":
            setattr(
                self,
                name,
                infinicore.device._from_infinicore_device(
                    getattr(self._underlying, name)
                ),
            )
        else:
            raise AttributeError(
                "{!r} object has no attribute {!r}".format(__name__, name)
            )

        return getattr(self, name)

    @property
    def ndim(self):
        return self._underlying.ndim

    def data_ptr(self):
        return self._underlying.data_ptr()

    def size(self, dim=None):
        if dim is None:
            return self.shape

        return self.shape[dim]

    def stride(self, dim=None):
        if dim is None:
            return self._underlying.strides

        return self._underlying.strides[dim]

    def numel(self):
        return self._underlying.numel()

    def is_contiguous(self):
        return self._underlying.is_contiguous()

    def is_pinned(self):
        return self._underlying.is_pinned()

    def copy_(self, src):
        self._underlying.copy_(src._underlying)

    def to(self, *args, **kwargs):
        return Tensor(
            self._underlying.to(*tuple(arg._underlying for arg in args), **kwargs)
        )

    def contiguous(self):
        return Tensor(self._underlying.contiguous())

    def as_strided(self, size, stride):
        return Tensor(self._underlying.as_strided(size, stride))

    def permute(self, dims):
        return Tensor(self._underlying.permute(dims))

    def view(self, shape):
        return Tensor(self._underlying.view(shape))

    def squeeze(self, dim):
        return infinicore.squeeze(self, dim)

    def unsqueeze(self, dim):
        return infinicore.unsqueeze(self, dim)

    def debug(self, filename=None):
        """Print tensor data or save to file for debugging

        Args:
            filename: Optional filename to save raw binary data. If None, prints to stdout.
        """
        if filename is None:
            self._underlying.debug()
        else:
            self._underlying.debug(filename)

    def __add__(self, other):
        return infinicore.add(self, other)

    def __iadd__(self, other):
        infinicore.add(self, other, out=self)
        return self

    def __matmul__(self, other):
        return infinicore.matmul(self, other)

    def __mul__(self, other):
        return infinicore.mul(self, other)

    def narrow(self, dim, start, length):
        return infinicore.narrow(self, dim, start, length)


def empty(size, *, dtype=None, device=None, pin_memory=False):
    return Tensor(
        _infinicore.empty(size, dtype._underlying, device._underlying, pin_memory)
    )


def empty_like(input, *, dtype=None, device=None):
    if dtype is None:
        dtype = input.dtype

    if device is None:
        device = input.device

    return empty(input.size(), dtype=dtype, device=device)


def strided_empty(size, strides, *, dtype=None, device=None, pin_memory=False):
    return Tensor(
        _infinicore.strided_empty(
            size, strides, dtype._underlying, device._underlying, pin_memory
        )
    )


def zeros(size, *, dtype=None, device=None, pin_memory=False):
    return Tensor(
        _infinicore.zeros(size, dtype._underlying, device._underlying, pin_memory)
    )


def ones(size, *, dtype=None, device=None, pin_memory=False):
    return Tensor(
        _infinicore.ones(size, dtype._underlying, device._underlying, pin_memory)
    )


def from_blob(data_ptr, size, *, dtype=None, device=None):
    return Tensor(
        _infinicore.from_blob(data_ptr, size, dtype._underlying, device._underlying)
    )


def strided_from_blob(data_ptr, size, strides, *, dtype=None, device=None):
    return Tensor(
        _infinicore.strided_from_blob(
            data_ptr, size, strides, dtype._underlying, device._underlying
        )
    )


def from_torch(torch_tensor) -> Tensor:
    infini_type = to_infinicore_dtype(torch_tensor.dtype)
    infini_device = infinicore.device(torch_tensor.device.type, 0)
    return Tensor(
        _infinicore.from_blob(
            torch_tensor.data_ptr(),
            list(torch_tensor.shape),
            dtype=infini_type._underlying,
            device=infini_device._underlying,
        ),
        _torch_ref=torch_tensor,
    )


def from_numpy(
    np_array,
    *,
    dtype: infinicore.dtype = None,
    device: infinicore.device = None,
) -> Tensor:
    """Convert a NumPy ndarray to an infinicore Tensor.

    Args:
        np_array: NumPy ndarray to convert to tensor
        dtype: Optional infinicore dtype. If None, inferred from numpy array
        device: Optional infinicore device. If None, defaults to CPU device

    Returns:
        Tensor: An infinicore tensor created from the numpy array

    Raises:
        TypeError: If input data is not a numpy ndarray
        ValueError: If input array is empty

    Note:
        NumPy arrays can only be created on CPU. For CUDA devices, data is first
        created on CPU, then copied to the target device.
    """
    # Input validation
    if not isinstance(np_array, np.ndarray):
        raise TypeError(
            f"Input data must be a np.ndarray, got {type(np_array).__name__}"
        )

    if np_array.size == 0:
        raise ValueError("Input array cannot be empty")

    # Determine target numpy dtype
    # If dtype is specified, convert it to numpy dtype first
    if dtype is not None:
        np_dtype = infinicore_to_numpy_dtype(dtype)
        # Create a copy with the target dtype if dtype doesn't match
        # Use copy=True to ensure we don't modify the original array
        if np_dtype != np_array.dtype:
            np_array = np_array.astype(np_dtype, copy=True)
        # Ensure C-contiguous layout
        elif not np_array.flags.c_contiguous:
            np_array = np.ascontiguousarray(np_array)
    else:
        # Ensure C-contiguous layout
        if not np_array.flags.c_contiguous:
            np_array = np.ascontiguousarray(np_array)

    # Infer infinicore dtype if not provided
    infini_type = (
        dtype if dtype is not None else numpy_to_infinicore_dtype(np_array.dtype)
    )

    # Default to CPU device if not provided
    infini_device = device if device is not None else infinicore.device("cpu", 0)
    cpu_device = infinicore.device("cpu", 0)

    # Create a temporary tensor on CPU using from_blob to reference numpy array
    # This allows us to copy data without keeping numpy array reference
    data_ptr = np_array.ctypes.data_as(ctypes.c_void_p).value
    temp_tensor = Tensor(
        _infinicore.from_blob(
            data_ptr,
            list(np_array.shape),
            dtype=infini_type._underlying,
            device=cpu_device._underlying,
        )
    )

    # Always create the result tensor on CPU first, then copy data
    # This ensures we have a proper copy of the data
    result = empty(list(np_array.shape), dtype=infini_type, device=cpu_device)
    result.copy_(temp_tensor)

    # If target device is not CPU, move the tensor to the target device
    # The temporary tensor and numpy array will be garbage collected
    # since we don't keep references to them
    if infini_device.type != "cpu":
        result = result.to(infini_device)

    return result


def from_list(data, *, dtype=None, device=None) -> Tensor:
    """Convert a Python list to an infinicore Tensor.

    Args:
        data: Python list or nested list to convert to tensor
        dtype: Optional infinicore dtype. If None, inferred from numpy array
        device: Optional infinicore device. If None, defaults to CPU device

    Returns:
        Tensor: An infinicore tensor created from the list data

    Raises:
        TypeError: If input data is not a list or tuple
        ValueError: If input data is empty

    Note:
        NumPy arrays can only be created on CPU. For CUDA devices, data is first
        created on CPU, then copied to the target device.
        This function internally converts the list to a numpy array and calls from_numpy.
    """
    # Input validation
    if not isinstance(data, (list, tuple)):
        raise TypeError(
            f"Input data must be a list or tuple, got {type(data).__name__}"
        )

    if not data:
        raise ValueError("Input data cannot be empty")

    # Determine target numpy dtype
    # If dtype is specified, convert it to numpy dtype first
    # This ensures the numpy array has the correct dtype from the start
    if dtype is not None:
        np_dtype = infinicore_to_numpy_dtype(dtype)
    else:
        np_dtype = None  # Let numpy infer

    # Convert Python list to numpy array with correct dtype
    # NumPy arrays can only be created on CPU
    # Use np.array(..., copy=True, order='C') to efficiently:
    # - Convert data type (if dtype is specified)
    # - Create a copy (ensuring data ownership)
    # - Ensure C-contiguous memory layout
    if np_dtype is not None:
        np_array = np.array(data, dtype=np_dtype, copy=True, order="C")
    else:
        np_array = np.array(data, copy=True, order="C")

    # Reuse from_numpy to create the tensor
    # This avoids code duplication and ensures consistent behavior
    return from_numpy(np_array, dtype=dtype, device=device)
