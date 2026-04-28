from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def inner(input, other, *, out=None):
    if out is None:
        return Tensor(_infinicore.inner(input._underlying, other._underlying))

    _infinicore.inner_(out._underlying, input._underlying, other._underlying)

    return out
