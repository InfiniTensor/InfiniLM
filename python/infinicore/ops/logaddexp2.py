from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def logaddexp2(input, other, *, out=None):
    if out is None:
        return Tensor(_infinicore.logaddexp2(input._underlying, other._underlying))

    _infinicore.logaddexp2_(out._underlying, input._underlying, other._underlying)

    return out
