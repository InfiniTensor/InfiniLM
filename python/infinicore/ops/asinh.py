from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def asinh(input, *, out=None):
    if out is None:
        return Tensor(_infinicore.asinh(input._underlying))

    _infinicore.asinh_(out._underlying, input._underlying)

    return out
