from __future__ import annotations

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def selu(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    if out is None:
        return Tensor(_infinicore.selu(input._underlying))

    _infinicore.selu_(out._underlying, input._underlying)
    return out
