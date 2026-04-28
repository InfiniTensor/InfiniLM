import infinicore
from infinicore.nn import functional as F

from ...tensor import Tensor
from ..parameter import InfiniCoreParameter as Parameter
from .module import InfiniCoreModule as Module


class Linear(Module):
    r"""Applies an affine linear transformation to the incoming data: :math:`y = xA^T + b`.

    Args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
        bias (bool): If set to ``False``, the layer will not learn an additive bias.
                Default: ``True``

    Shape:
        - Input: :(*, H_in) where :math:`*` means any number of dimensions, H_in = in_features.
        - Output: :math:(*, H_out) where all but the last dimension are the same shape as the input and :math:H_out = out_features.

    Attributes:
        weight(Tensor): the weights of the module of shape (out_features, in_features).
        bias(Tensor):   the bias of the module of shape (out_features).
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {
            "device": infinicore.device("cpu", 0) if device is None else device,
            "dtype": infinicore.float32 if dtype is None else dtype,
        }

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(
            infinicore.empty([out_features, in_features], **factory_kwargs)
        )

        if bias:
            self.bias = Parameter(infinicore.empty([out_features], **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
