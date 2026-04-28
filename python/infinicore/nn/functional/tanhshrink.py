from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def tanhshrink(input, *, out=None):
    if out is None:
        return Tensor(_infinicore.tanhshrink(input._underlying))

    _infinicore.tanhshrink_(out._underlying, input._underlying)

    return out
