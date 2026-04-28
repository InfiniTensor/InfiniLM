from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def sum(input, dim=None, keepdim=False, out=None):
    """
    Sum the elements of the input tensor along the given dimensions.

    Args:
        input (Tensor): The input tensor.
        out (Tensor, optional): The output tensor.

    Returns:
        Tensor: The output tensor.

    Example:
        >>> import infinicore
        >>> input = infinicore.tensor([[1, 2, 3], [4, 5, 6]])
        >>> output = infinicore.sum(input)
        >>> print(output)
        tensor([15])
    """
    if out is None:
        return Tensor(_infinicore.sum(input._underlying, dim, keepdim))

    _infinicore.sum_(out._underlying, input._underlying, dim, keepdim)

    return out
