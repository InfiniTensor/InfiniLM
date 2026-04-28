from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def index_add(input, dim, index, source, *, alpha=1.0, out=None):
    if out is None:
        return Tensor(
            _infinicore.index_add(
                input._underlying, dim, index._underlying, source._underlying, alpha
            )
        )
    _infinicore.index_add_(
        out._underlying,
        input._underlying,
        dim,
        index._underlying,
        source._underlying,
        alpha,
    )

    return out
