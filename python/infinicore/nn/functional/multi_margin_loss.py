from typing import Optional

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

_REDUCTION_MODES = {
    "none": 0,
    "mean": 1,
    "sum": 2,
}


def multi_margin_loss(
    input: Tensor,
    target: Tensor,
    p: int = 1,
    margin: float = 1.0,
    weight: Optional[Tensor] = None,
    reduction: str = "mean",
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    r"""Creates a criterion that optimizes a multi-class classification hinge
    loss (margin-based loss) between input x and output y.
    """

    if not input.is_contiguous():
        input = input.contiguous()
    if not target.is_contiguous():
        target = target.contiguous()

    weight_underlying = None
    if weight is not None:
        if not weight.is_contiguous():
            weight = weight.contiguous()
        weight_underlying = weight._underlying

    # 解析 reduction 参数
    if reduction not in _REDUCTION_MODES:
        raise ValueError(f"{reduction} is not a valid value for reduction")
    reduction_val = _REDUCTION_MODES[reduction]
    if out is not None:
        _infinicore.multi_margin_loss_(
            out._underlying,
            input._underlying,
            target._underlying,
            weight_underlying,
            p,
            margin,
            reduction_val,
        )
        return out

    return Tensor(
        _infinicore.multi_margin_loss(
            input._underlying,
            target._underlying,
            weight_underlying,
            p,
            margin,
            reduction_val,
        )
    )
