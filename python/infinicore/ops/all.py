from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def all(input, dim=None, keepdim=False, out=None):
    if out is None:
        return Tensor(_infinicore.all(input._underlying, dim, keepdim))

    _infinicore.all_(out._underlying, input._underlying, dim, keepdim)

    return out
