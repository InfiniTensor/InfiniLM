from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def floor(input, *, out=None):
    if out is None:
        return Tensor(_infinicore.floor(input._underlying))

    _infinicore.floor_(out._underlying, input._underlying)

    return out
