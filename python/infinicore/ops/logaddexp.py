from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def logaddexp(input, other, *, out=None):
    if out is None:
        return Tensor(_infinicore.logaddexp(input._underlying, other._underlying))

    _infinicore.logaddexp_(out._underlying, input._underlying, other._underlying)

    return out
