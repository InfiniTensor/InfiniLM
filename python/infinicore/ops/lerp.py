from typing import Optional, Union

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def lerp(
    start: Tensor,
    end: Tensor,
    weight: Union[Tensor, float],
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    r"""Does a linear interpolation of two tensors start and end based on a scalar or tensor weight.

    output = start + weight * (end - start)
    """

    # 检查输入 Tensor 的连续性
    if not start.is_contiguous():
        start = start.contiguous()
    if not end.is_contiguous():
        end = end.contiguous()

    # 处理 weight 参数：可能是 Tensor 也可能是标量
    weight_arg = weight
    if isinstance(weight, Tensor):
        if not weight.is_contiguous():
            weight = weight.contiguous()
        weight_arg = weight._underlying
    elif isinstance(weight, (float, int)):
        weight_arg = float(weight)
    else:
        raise TypeError(f"weight must be a Tensor or float, got {type(weight)}")

    # In-place / 输出到指定 Tensor
    if out is not None:
        _infinicore.lerp_(
            out._underlying, start._underlying, end._underlying, weight_arg
        )
        return out

    # 返回新 Tensor
    return Tensor(_infinicore.lerp(start._underlying, end._underlying, weight_arg))
