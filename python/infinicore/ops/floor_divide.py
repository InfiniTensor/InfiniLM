from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def floor_divide(input, other, *, out=None):
    if out is None:
        return Tensor(_infinicore.floor_divide(input._underlying, other._underlying))

    _infinicore.floor_divide_(out._underlying, input._underlying, other._underlying)

    return out
