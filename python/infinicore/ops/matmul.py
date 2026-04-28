from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def matmul(input, other, *, alpha=1.0, out=None):
    if out is None:
        return Tensor(_infinicore.matmul(input._underlying, other._underlying, alpha))

    _infinicore.matmul_(out._underlying, input._underlying, other._underlying, alpha)

    return out
