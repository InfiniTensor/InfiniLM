from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def equal(input, other, *, out=None):
    if out is None:
        return Tensor(_infinicore.equal(input._underlying, other._underlying))

    _infinicore.equal_(out._underlying, input._underlying, other._underlying)
    return out
