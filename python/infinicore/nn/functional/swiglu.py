from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def swiglu(input: Tensor, other: Tensor, *, out=None):
    r"""Apply the Swish-Gated Linear Unit (SwiGLU) function, element-wise."""

    if out is None:
        return Tensor(_infinicore.swiglu(input._underlying, other._underlying))

    _infinicore.swiglu_(out._underlying, input._underlying, other._underlying)

    return out
