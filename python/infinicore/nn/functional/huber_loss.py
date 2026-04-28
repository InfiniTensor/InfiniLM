from typing import Optional

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

_REDUCTION_MODES = {
    "none": 0,
    "mean": 1,
    "sum": 2,
}


def huber_loss(
    input: Tensor,
    target: Tensor,
    delta: float = 1.0,
    reduction: str = "mean",
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    r"""Creates a criterion that uses a squared term if the absolute
    element-wise error falls below delta and a delta-scaled L1 term otherwise.
    """

    if not input.is_contiguous():
        input = input.contiguous()
    if not target.is_contiguous():
        target = target.contiguous()

    # 解析 reduction 参数
    if reduction not in _REDUCTION_MODES:
        raise ValueError(f"{reduction} is not a valid value for reduction")
    reduction_val = _REDUCTION_MODES[reduction]

    if out is not None:
        _infinicore.huber_loss_(
            out._underlying, input._underlying, target._underlying, delta, reduction_val
        )
        return out

    return Tensor(
        _infinicore.huber_loss(
            input._underlying, target._underlying, delta, reduction_val
        )
    )
