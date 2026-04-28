from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def atanh(input, *, out=None):
    if out is None:
        return Tensor(_infinicore.atanh(input._underlying))

    _infinicore.atanh_(out._underlying, input._underlying)

    return out
