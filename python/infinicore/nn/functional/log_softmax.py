from typing import Optional

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def log_softmax(input: Tensor, dim: int, *, out: Optional[Tensor] = None) -> Tensor:
    r"""Applies a softmax followed by a logarithm.
    While mathematically equivalent to log(softmax(x)), doing these two
    operations separately is slower and numerically unstable. This function
    uses an alternative formulation to compute the output and gradient correctly.
    """

    if not input.is_contiguous():
        input = input.contiguous()

    if out is not None:
        if not isinstance(out, Tensor):
            raise ValueError("out must be a Tensor")

        _infinicore.log_softmax_(out._underlying, input._underlying, dim)
        return out

    ret = _infinicore.log_softmax(input._underlying, dim)

    return Tensor(ret)
