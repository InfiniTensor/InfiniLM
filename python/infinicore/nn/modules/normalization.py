import numbers
from typing import Optional, Union

import infinicore
from infinicore.nn import functional as F

from ...tensor import Tensor
from ..parameter import InfiniCoreParameter as Parameter
from .module import InfiniCoreModule as Module


class RMSNorm(Module):
    r"""Applies Root Mean Square Layer Normalization over a mini-batch of inputs.
    The RMS is taken over the last dimensions, where normalized_shape is one-dimensional.

    Args:
        normalized_shape (int or list): input shape from an expected input of size  [* , normalized_shape[0]]
        this module will normalize over the last dimension.
        eps (float): a value added to the denominator for numerical stability.

    Shape:
        - Input: (N, *)
        - Output: (N, *) (same shape as input)
    """

    __constants__ = ["normalized_shape", "eps"]
    normalized_shape: tuple[int]
    eps: Optional[float]

    def __init__(
        self,
        normalized_shape: Union[int, list[int]],
        eps=1e-6,
        elementwise_affine=True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {
            "dtype": infinicore.float32 if dtype is None else dtype,
            "device": infinicore.device("cpu", 0) if device is None else device,
        }
        super().__init__()

        assert elementwise_affine, "elementwise_affine must be true."

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = [normalized_shape]

        self.normalized_shape = list(normalized_shape)
        self.eps = eps
        self.weight = Parameter(
            infinicore.empty(self.normalized_shape, **factory_kwargs)
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)

    def extra_repr(self) -> str:
        return "{normalized_shape}, eps={eps}".format(**self.__dict__)
