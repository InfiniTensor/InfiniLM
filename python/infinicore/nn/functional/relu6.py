from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def relu6(input: Tensor, inplace: bool = False, *, out: Tensor | None = None) -> Tensor:
    if inplace:
        _infinicore.relu6_(input._underlying, input._underlying)
        return input

    if out is None:
        return Tensor(_infinicore.relu6(input._underlying))

    _infinicore.relu6_(out._underlying, input._underlying)
    return out
