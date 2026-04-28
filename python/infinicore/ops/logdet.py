from __future__ import annotations

from typing import Optional

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def logdet(input: Tensor, *, out: Optional[Tensor] = None):
    if out is None:
        return Tensor(_infinicore.logdet(input._underlying))

    _infinicore.logdet_(out._underlying, input._underlying)
    return out
