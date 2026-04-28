from typing import List

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def adaptive_avg_pool3d(x: Tensor, output_size: List[int] = {1, 1, 1}) -> Tensor:
    r"""Applies a 3D adaptive average pooling over an input signal composed of several input planes.

    Args:
        x (Tensor): The input tensor of shape (N, C, D, H, W)
        output_size (List[int]): The target output size of the form (d, h, w)

    Returns:
        Tensor: The pooled output tensor
    """
    return Tensor(_infinicore.adaptive_avg_pool3d(x._underlying, output_size))
