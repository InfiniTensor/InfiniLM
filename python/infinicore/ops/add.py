from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def add(input, other, *, out=None):
    if out is None:
        return Tensor(_infinicore.add(input._underlying, other._underlying))

    _infinicore.add_(out._underlying, input._underlying, other._underlying)

    return out
