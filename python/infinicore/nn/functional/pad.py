from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def pad(
    input: Tensor,
    pad: Sequence[int],
    mode: str = "constant",
    value: float = 0.0,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    pad_list = list(pad)
    if out is None:
        return Tensor(_infinicore.pad(input._underlying, pad_list, mode, value))

    _infinicore.pad_(out._underlying, input._underlying, pad_list, mode, value)
    return out
