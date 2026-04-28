from typing import List, Optional, Tuple, Union

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def unfold(
    input: Tensor,
    kernel_size: Union[int, Tuple[int, ...], List[int]],
    dilation: Union[int, Tuple[int, ...], List[int]] = 1,
    padding: Union[int, Tuple[int, ...], List[int]] = 0,
    stride: Union[int, Tuple[int, ...], List[int]] = 1,
    *,
    out: Optional[Tensor] = None,
) -> Tensor:
    r"""Extracts sliding local blocks from a batched input tensor.

    Also known as im2col. The output tensor contains the flattened blocks.

    Args:
        input (Tensor): The input tensor.
        kernel_size (int or tuple): The size of the sliding blocks.
        dilation (int or tuple, optional): The parameter that controls the stride of elements within the neighborhood. Default: 1.
        padding (int or tuple, optional): Implicit zero padding to be added on both sides of input. Default: 0.
        stride (int or tuple, optional): The stride of the sliding blocks. Default: 1.
        out (Tensor, optional): The output tensor.
    """

    if not input.is_contiguous():
        input = input.contiguous()

    # Helper to ensure parameters are iterable (assuming 2D spatial dims for single int)
    def _pair(x):
        return (x, x) if isinstance(x, int) else x

    k_val = _pair(kernel_size)
    d_val = _pair(dilation)
    p_val = _pair(padding)
    s_val = _pair(stride)

    if out is not None:
        _infinicore.unfold_(
            out._underlying, input._underlying, k_val, d_val, p_val, s_val
        )
        return out

    return Tensor(_infinicore.unfold(input._underlying, k_val, d_val, p_val, s_val))
