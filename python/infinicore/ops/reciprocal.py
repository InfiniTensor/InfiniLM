from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def reciprocal(input, *, out=None):
    if out is None:
        return Tensor(_infinicore.reciprocal(input._underlying))

    _infinicore.reciprocal_(out._underlying, input._underlying)

    return out
