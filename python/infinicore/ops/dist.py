from __future__ import annotations

from typing import Optional

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def dist(input: Tensor, other: Tensor, p: float = 2.0, *, out: Optional[Tensor] = None):
    if out is None:
        return Tensor(_infinicore.dist(input._underlying, other._underlying, p))

    _infinicore.dist_(out._underlying, input._underlying, other._underlying, p)
    return out
