from typing import Optional

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

_REDUCTION_MODES = {
    "none": 0,
    "mean": 1,
    "sum": 2,
}


def triplet_margin_loss(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    margin: float = 1.0,
    p: float = 2,
    eps: float = 1e-6,
    swap: bool = False,
    reduction: str = "mean",
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    r"""Creates a criterion that measures the triplet loss given an input
    tensors x1, x2, x3 and a margin with a value greater than 0.
    """

    if not anchor.is_contiguous():
        anchor = anchor.contiguous()
    if not positive.is_contiguous():
        positive = positive.contiguous()
    if not negative.is_contiguous():
        negative = negative.contiguous()

    if reduction not in _REDUCTION_MODES:
        raise ValueError(f"{reduction} is not a valid value for reduction")
    reduction_val = _REDUCTION_MODES[reduction]

    if out is not None:
        _infinicore.triplet_margin_loss_(
            out._underlying,
            anchor._underlying,
            positive._underlying,
            negative._underlying,
            margin,
            int(p),
            eps,
            swap,
            reduction_val,
        )
        return out

    return Tensor(
        _infinicore.triplet_margin_loss(
            anchor._underlying,
            positive._underlying,
            negative._underlying,
            margin,
            int(p),
            eps,
            swap,
            reduction_val,
        )
    )
