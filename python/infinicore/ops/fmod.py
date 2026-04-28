from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def fmod(input, other, *, out=None):
    if out is None:
        return Tensor(_infinicore.fmod(input._underlying, other._underlying))

    _infinicore.fmod_(out._underlying, input._underlying, other._underlying)

    return out
