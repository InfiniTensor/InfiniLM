from typing import Optional

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def take(input: Tensor, indices: Tensor, *, out: Optional[Tensor] = None) -> Tensor:
    r"""Returns a new tensor with the elements of input at the given indices.
    The input tensor is treated as if it were viewed as a 1-D tensor.
    The result tensor has the same shape as the indices tensor.

    Args:
        input (Tensor): the input tensor.
        indices (Tensor): the indices into tensor, must be an Int or Long tensor.
        out (Tensor, optional): the output tensor.

    Returns:
        Tensor: A new tensor with the elements of input at the given indices.
    """
    if not input.is_contiguous():
        input = input.contiguous()

    # 如果用户提供了 output tensor，调用底层的 in-place/explicit 接口
    if out is not None:
        _infinicore.take_(out._underlying, input._underlying, indices._underlying)
        return out
    return Tensor(_infinicore.take(input._underlying, indices._underlying))
