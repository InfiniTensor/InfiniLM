from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def index_copy(input, dim, index, source, *, out=None):
    if out is None:
        return Tensor(
            _infinicore.index_copy(
                input._underlying, dim, index._underlying, source._underlying
            )
        )

    _infinicore.index_copy_(
        out._underlying, input._underlying, dim, index._underlying, source._underlying
    )

    return out
