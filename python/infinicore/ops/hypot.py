from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def hypot(input, other, *, out=None):
    if out is None:
        return Tensor(_infinicore.hypot(input._underlying, other._underlying))
    _infinicore.hypot_(out._underlying, input._underlying, other._underlying)

    return out
