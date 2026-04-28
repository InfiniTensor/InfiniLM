from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def asin(input: Tensor, *, out=None):
    """Arcsin activation function."""
    if out is None:
        return Tensor(_infinicore.asin(input._underlying))

    _infinicore.asin_(out._underlying, input._underlying)
    return out
