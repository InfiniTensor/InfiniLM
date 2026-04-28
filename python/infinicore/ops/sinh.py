from __future__ import annotations

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def sinh(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    if out is None:
        return Tensor(_infinicore.sinh(input._underlying))

    _infinicore.sinh_(out._underlying, input._underlying)
    return out
