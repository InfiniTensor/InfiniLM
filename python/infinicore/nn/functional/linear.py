from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

__all__ = ["linear"]


def linear(input: Tensor, weight: Tensor, bias=None, *, out=None) -> Tensor:
    r"""Applies a linear transformation to the incoming data: y=xA^T+b."""

    if out is None:
        return Tensor(
            _infinicore.linear(
                input._underlying,
                weight._underlying,
                None if bias is None else bias._underlying,
            )
        )

    _infinicore.linear_(
        out._underlying,
        input._underlying,
        weight._underlying,
        None if bias is None else bias._underlying,
    )
    return out
