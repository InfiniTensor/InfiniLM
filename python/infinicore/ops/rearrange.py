from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def rearrange(input, other, *, out=None):
    if out is None:
        return Tensor(_infinicore.rearrange(input._underlying))

    _infinicore.rearrange_(out._underlying, input._underlying)

    return out
