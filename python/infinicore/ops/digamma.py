from __future__ import annotations

from typing import Optional

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def digamma(input: Tensor, *, out: Optional[Tensor] = None):
    if out is None:
        return Tensor(_infinicore.digamma(input._underlying))

    _infinicore.digamma_(out._underlying, input._underlying)
    return out
