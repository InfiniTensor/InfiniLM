from ..lib import _infinicore
from ..tensor import Tensor


def _unwrap(value):
    return value._underlying


def add(input, other):
    return Tensor(_infinicore.add(_unwrap(input), _unwrap(other)))


def cat(tensors, dim=0):
    underlying = [_unwrap(tensor) for tensor in tensors]
    return Tensor(_infinicore.cat(underlying, dim))
