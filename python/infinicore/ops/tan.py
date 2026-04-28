from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def tan(input, *, out=None):
    if out is None:
        return Tensor(_infinicore.tan(input._underlying))

    _infinicore.tan_(out._underlying, input._underlying)

    return out
