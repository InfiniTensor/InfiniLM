from __future__ import annotations

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def block_diag(*tensors: Tensor) -> Tensor:
    if not tensors:
        raise ValueError("block_diag expects at least one input tensor")

    underlying = [t._underlying for t in tensors]
    return Tensor(_infinicore.block_diag(underlying))
