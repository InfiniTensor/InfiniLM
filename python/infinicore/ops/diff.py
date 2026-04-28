from __future__ import annotations

from typing import Optional

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def diff(input: Tensor, n: int = 1, dim: int = -1, *, out: Optional[Tensor] = None):
    if out is None:
        return Tensor(_infinicore.diff(input._underlying, n, dim))

    _infinicore.diff_(out._underlying, input._underlying, n, dim)
    return out
