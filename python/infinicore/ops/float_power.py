from typing import Optional

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def float_power(
    input: Tensor, exponent: float, *, out: Optional[Tensor] = None
) -> Tensor:
    r"""Computes the power of each element in input with the given exponent.

    .. math::
        \text{out}_i = \text{input}_i^{\text{exponent}}

    Args:
        input (Tensor): the input tensor.
        exponent (float): the exponent value.
        out (Tensor, optional): the output tensor.

    Returns:
        Tensor: The result tensor.
    """

    # 1. 确保输入内存连续 (Contiguous check)
    if not input.is_contiguous():
        input = input.contiguous()

    # 2. 分发计算
    # 如果用户提供了 output tensor，调用底层的 in-place/explicit 接口
    if out is not None:
        if not out.is_contiguous():
            raise RuntimeError("Output tensor must be contiguous")

        _infinicore.float_power_(out._underlying, input._underlying, exponent)
        return out

    # 否则调用底层的 functional 接口，返回新 Tensor
    return Tensor(_infinicore.float_power(input._underlying, exponent))
