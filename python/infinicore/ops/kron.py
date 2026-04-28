from __future__ import annotations

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def kron(a: Tensor, b: Tensor) -> Tensor:
    return Tensor(_infinicore.kron(a._underlying, b._underlying))
