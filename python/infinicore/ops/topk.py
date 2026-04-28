from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def topk(input, k, dim, largest=True, sorted=True, out=None):
    if out is None:
        values, indices = _infinicore.topk(input._underlying, k, dim, largest, sorted)
        return Tensor(values), Tensor(indices)

    _infinicore.topk_(out._underlying, input._underlying, k, dim, largest, sorted)

    return out
