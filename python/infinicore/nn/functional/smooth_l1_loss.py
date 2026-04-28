from typing import Optional

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

_REDUCTION_MODES = {
    "none": 0,
    "mean": 1,
    "sum": 2,
}


def smooth_l1_loss(
    input: Tensor,
    target: Tensor,
    beta: float = 1.0,
    reduction: str = "mean",
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    r"""Creates a criterion that uses a squared term if the absolute
    element-wise error falls below beta and an L1 term otherwise.

    Args:
        input (Tensor): the input tensor.
        target (Tensor): the target tensor.
        beta (float, optional): The threshold at which to change between L1 and L2 loss.
            The value must be non-negative. Default: 1.0.
        reduction (str, optional): Specifies the reduction to apply to the output:
            'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of elements in the output,
            'sum': the output will be summed. Default: 'mean'.
        out (Tensor, optional): the output tensor.

    Returns:
        Tensor: The loss value.
    """

    if not input.is_contiguous():
        input = input.contiguous()
    if not target.is_contiguous():
        target = target.contiguous()
    if reduction not in _REDUCTION_MODES:
        raise ValueError(f"{reduction} is not a valid value for reduction")
    reduction_val = _REDUCTION_MODES[reduction]
    if out is not None:
        _infinicore.smooth_l1_loss_(
            out._underlying, input._underlying, target._underlying, beta, reduction_val
        )
        return out
    return Tensor(
        _infinicore.smooth_l1_loss(
            input._underlying, target._underlying, beta, reduction_val
        )
    )
