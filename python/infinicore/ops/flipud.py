from typing import Optional

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def flipud(input: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    r"""Flip array in the up/down direction.

    Flips the entries in axis 0 (preserving the shape).

    Args:
        input (Tensor): the input tensor.
        out (Tensor, optional): the output tensor.

    Returns:
        Tensor: The flipped tensor.
    """
    if not input.is_contiguous():
        input = input.contiguous()
    if out is not None:
        _infinicore.flipud_(out._underlying, input._underlying)
        return out
    return Tensor(_infinicore.flipud(input._underlying))
