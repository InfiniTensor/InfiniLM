from ..lib import _infinicore
from ..tensor import Tensor


def _unwrap(value):
    return value._underlying


def add(input, other, *, out=None):
    if out is None:
        return Tensor(_infinicore.add(_unwrap(input), _unwrap(other)))
    _infinicore.add_(_unwrap(out), _unwrap(input), _unwrap(other))
    return out


def cat(tensors, dim=0, *, out=None):
    underlying = [_unwrap(tensor) for tensor in tensors]
    if out is None:
        return Tensor(_infinicore.cat(underlying, dim))
    _infinicore.cat_(_unwrap(out), underlying, dim)
    return out


def matmul(input, other, *, alpha=1.0, out=None):
    if out is None:
        return Tensor(_infinicore.matmul(_unwrap(input), _unwrap(other), alpha))
    _infinicore.matmul_(_unwrap(out), _unwrap(input), _unwrap(other), alpha)
    return out
