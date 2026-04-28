from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def silu_and_mul(input: Tensor, out=None) -> Tensor:
    r"""Apply the SiLU and Mul (SwiGLU) function.

    Formula: output = SiLU(input_gate) * input_up
    Input shape: [..., 2*d], Output shape: [..., d]
    """

    if out is None:
        return Tensor(_infinicore.silu_and_mul(input._underlying))

    _infinicore.silu_and_mul_(out._underlying, input._underlying)

    return out
