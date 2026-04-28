from typing import Optional

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def triplet_margin_with_distance_loss(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    *,
    margin: float = 1.0,
    swap: bool = False,
    reduction: str = "mean",
    out: Optional[Tensor] = None,
) -> Tensor:
    r"""Calculates the triplet margin loss for a given triplet of tensors.
    The loss is defined as: L(a, p, n) = max(d(a, p) - d(a, n) + margin, 0)
    """

    if not anchor.is_contiguous():
        anchor = anchor.contiguous()
    if not positive.is_contiguous():
        positive = positive.contiguous()
    if not negative.is_contiguous():
        negative = negative.contiguous()

    reduction_map = {"none": 0, "mean": 1, "sum": 2}
    if reduction not in reduction_map:
        raise ValueError(f"Invalid reduction mode: {reduction}")

    reduction_val = reduction_map[reduction]

    if out is not None:
        if not isinstance(out, Tensor):
            raise ValueError("out must be a Tensor")

        _infinicore.triplet_margin_with_distance_loss_(
            out._underlying,
            anchor._underlying,
            positive._underlying,
            negative._underlying,
            margin,
            swap,
            reduction_val,
        )
        return out

    ret = _infinicore.triplet_margin_with_distance_loss(
        anchor._underlying,
        positive._underlying,
        negative._underlying,
        margin,
        swap,
        reduction_val,
    )

    return Tensor(ret)
