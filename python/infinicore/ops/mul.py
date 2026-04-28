from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def mul(input, other, *, out=None):
    if out is None:
        return Tensor(_infinicore.mul(input._underlying, other._underlying))

    _infinicore.mul_(out._underlying, input._underlying, other._underlying)

    return out
