from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def adaptive_max_pool1d(
    input: Tensor,
    output_size: int,
    *,
    out=None,
) -> Tensor:
    r"""Applies a 1D adaptive max pooling over an input signal composed of
    several input planes.

    The output size is H_out. The algorithm used is fairly simple:

    .. math::
        \text{start} = \left\lfloor \frac{i \cdot L_{in}}{L_{out}} \right\rfloor

        \text{end} = \left\lceil \frac{(i + 1) \cdot L_{in}}{L_{out}} \right\rceil

    where :math:`L_{in}` is the size of the input dimension, and :math:`L_{out}` is the size of the output dimension.

    Args:
        input (Tensor): Input tensor of shape (N, C, L_in)
        output_size (int): The target output size (L_out)
        out (Tensor, optional): Output tensor.

    Returns:
        Tensor: The result of the adaptive max pooling operation.
    """

    if out is None:
        return Tensor(_infinicore.adaptive_max_pool1d(input._underlying, output_size))

    _infinicore.adaptive_max_pool1d_(out._underlying, input._underlying, output_size)

    return out
