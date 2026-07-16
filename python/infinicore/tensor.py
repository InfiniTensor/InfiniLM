from infinicore.device import device as Device
from infinicore.dtype import dtype as DType
from infinicore.lib import _infinicore


class Tensor:
    def __init__(self, underlying, *, owner=None):
        self._underlying = underlying
        self._owner = owner

    @property
    def shape(self):
        return self._underlying.shape

    @property
    def strides(self):
        return self._underlying.strides

    @property
    def ndim(self):
        return self._underlying.ndim

    @property
    def dtype(self):
        return DType(self._underlying.dtype)

    @property
    def device(self):
        return Device._from_underlying(self._underlying.device)

    def data_ptr(self):
        return self._underlying.data_ptr()

    def size(self, dim=None):
        return self.shape if dim is None else self._underlying.size(dim)

    def stride(self, dim=None):
        return self.strides if dim is None else self._underlying.stride(dim)

    def numel(self):
        return self._underlying.numel()

    def is_contiguous(self):
        return self._underlying.is_contiguous()

    def is_pinned(self):
        return self._underlying.is_pinned()

    def copy_(self, source):
        self._underlying.copy_(source._underlying)
        return self

    def to(self, target=None, *, device=None):
        if target is not None and device is not None:
            raise TypeError("device must be provided once")
        target = device if device is not None else target
        if target is None:
            return self
        return Tensor(self._underlying.to(Device(target)._underlying), owner=self)

    def to_numpy(self):
        """Return an owning, C-contiguous NumPy copy of this tensor."""
        import ctypes

        import numpy as np

        from infinicore.utils import infinicore_to_numpy_dtype

        source = self if self.device.type == "cpu" else self.to(Device("cpu"))
        if not source.is_contiguous():
            source = source.contiguous()

        result = np.empty(source.shape, dtype=infinicore_to_numpy_dtype(source.dtype))
        if result.nbytes:
            ctypes.memmove(result.ctypes.data, source.data_ptr(), result.nbytes)
        return result

    def contiguous(self):
        return Tensor(self._underlying.contiguous(), owner=self._owner)

    def as_strided(self, size, stride):
        return Tensor(self._underlying.as_strided(size, stride), owner=self._owner)

    def narrow(self, dim, start, length):
        return Tensor(self._underlying.narrow(dim, start, length), owner=self._owner)

    def permute(self, dims):
        return Tensor(self._underlying.permute(dims), owner=self._owner)

    def view(self, shape):
        return Tensor(self._underlying.view(shape), owner=self._owner)

    def squeeze(self, dim):
        return Tensor(self._underlying.squeeze(dim), owner=self._owner)

    def unsqueeze(self, dim):
        return Tensor(self._underlying.unsqueeze(dim), owner=self._owner)

    def debug(self, filename=None):
        if filename is None:
            return self._underlying.debug()
        return self._underlying.debug(filename)

    def __add__(self, other):
        from infinicore.ops import add

        return add(self, other)

    def __iadd__(self, other):
        from infinicore.ops import add

        return add(self, other, out=self)

    def __matmul__(self, other):
        from infinicore.ops import matmul

        return matmul(self, other)

    def __repr__(self):
        return repr(self._underlying)


def _normalize_factory_args(dtype, device):
    from infinicore import float32

    return (
        float32 if dtype is None else dtype,
        Device("cpu") if device is None else Device(device),
    )


def empty(size, *, dtype=None, device=None, pin_memory=False):
    dtype, device = _normalize_factory_args(dtype, device)
    return Tensor(
        _infinicore.empty(size, dtype._underlying, device._underlying, pin_memory)
    )


def empty_like(input, *, dtype=None, device=None):
    return empty(
        input.shape,
        dtype=input.dtype if dtype is None else dtype,
        device=input.device if device is None else device,
    )


def strided_empty(size, strides, *, dtype=None, device=None, pin_memory=False):
    dtype, device = _normalize_factory_args(dtype, device)
    return Tensor(
        _infinicore.strided_empty(
            size, strides, dtype._underlying, device._underlying, pin_memory
        )
    )


def zeros(size, *, dtype=None, device=None, pin_memory=False):
    dtype, device = _normalize_factory_args(dtype, device)
    return Tensor(
        _infinicore.zeros(size, dtype._underlying, device._underlying, pin_memory)
    )


def from_blob(data_ptr, size, *, dtype, device):
    return Tensor(
        _infinicore.from_blob(
            data_ptr, size, dtype._underlying, Device(device)._underlying
        )
    )


def from_torch(torch_tensor, *, device=None):
    import torch

    from infinicore.context import get_device, sync_stream
    from infinicore.utils import to_infinicore_dtype

    owner = torch_tensor.detach().contiguous()
    if owner.device.type not in ("cpu", "cuda"):
        raise ValueError(
            "from_torch() currently supports CPU and CUDA-compatible tensors"
        )

    index = 0 if owner.device.index is None else owner.device.index

    if device is None:
        if owner.device.type == "cuda":
            current = get_device()
            if current.type not in ("cpu", "cuda"):
                raise ValueError(
                    "Torch reports a CUDA-compatible device without its vendor; "
                    "pass device= explicitly"
                )
            target = Device("nvidia", index)
        else:
            target = Device(owner.device.type, index)
    elif isinstance(device, str) and ":" not in device:
        target = Device(device, index)
    else:
        target = Device(device)

    if target.index != index:
        raise ValueError("device index must match the source Torch tensor")
    if owner.device.type == "cuda":
        if target.type not in ("cuda", "metax", "iluvatar", "hygon"):
            raise ValueError("device is not compatible with a Torch CUDA tensor")
    elif target.type != owner.device.type:
        raise ValueError("device type must match the source Torch tensor")

    dtype = to_infinicore_dtype(owner.dtype)
    borrowed = Tensor(
        _infinicore.from_blob(
            owner.data_ptr(),
            list(owner.shape),
            dtype._underlying,
            target._underlying,
        ),
        owner=owner,
    )
    result = empty(list(owner.shape), dtype=dtype, device=target)

    if owner.device.type == "cuda":
        torch.cuda.synchronize(owner.device)

    result.copy_(borrowed)
    if owner.device.type == "cuda":
        sync_stream()
    return result


def from_numpy(array, *, dtype=None, device=None):
    import numpy as np

    from infinicore.utils import (
        infinicore_to_numpy_dtype,
        numpy_to_infinicore_dtype,
    )

    source = np.asarray(array)
    if dtype is None:
        dtype = numpy_to_infinicore_dtype(source.dtype)
    numpy_dtype = infinicore_to_numpy_dtype(dtype)
    owner = np.ascontiguousarray(array, dtype=numpy_dtype)

    result = Tensor(_infinicore._from_numpy_copy(owner, dtype._underlying))
    cpu = Device("cpu")
    target = cpu if device is None else Device(device)
    return result if target == cpu else result.to(target)


def from_list(data, *, dtype):
    if dtype is None:
        raise TypeError("from_list() requires dtype")
    return Tensor(_infinicore.from_list(data, dtype._underlying))
