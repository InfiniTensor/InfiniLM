from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def fmin(input, other, *, out=None):
    if out is None:
        return Tensor(_infinicore.fmin(input._underlying, other._underlying))

    _infinicore.fmin_(out._underlying, input._underlying, other._underlying)
    return out
