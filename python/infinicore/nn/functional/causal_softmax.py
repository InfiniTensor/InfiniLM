from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def causal_softmax(input: Tensor, out=None) -> Tensor:
    r"""Apply a causal softmax function."""

    if out is None:
        return Tensor(_infinicore.causal_softmax(input._underlying))

    _infinicore.causal_softmax_(out._underlying, input._underlying)

    return out
