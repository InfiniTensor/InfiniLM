from typing import Optional

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def ldexp(input: Tensor, other: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    r"""Multiplies input by 2 raised to the power of other.

    Args:
        input (Tensor): The input tensor (mantissa).
        other (Tensor): The exponent tensor.
    """

    # 1. 确保输入内存连续
    if not input.is_contiguous():
        input = input.contiguous()
    if not other.is_contiguous():
        other = other.contiguous()

    # 2. 处理 Explicit Output (out=...)
    if out is not None:
        if not isinstance(out, Tensor):
            raise ValueError("out must be a Tensor")

        _infinicore.ldexp_(out._underlying, input._underlying, other._underlying)
        return out

    # 3. 处理 Functional 调用
    ret = _infinicore.ldexp(input._underlying, other._underlying)

    # 4. 封装返回结果
    return Tensor(ret)
