from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def softsign(input, *, out=None):
    if out is None:
        return Tensor(_infinicore.softsign(input._underlying))

    _infinicore.softsign_(out._underlying, input._underlying)

    return out
